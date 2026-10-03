"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Hoàn thiện từ bộ khung starter/inference.py. Liên hệ slide Day 2: TTA (trang 62-66, 75),
ensemble/EMA/soup (trang 67), độ phân giải kiểm tra (trang 68), temperature scaling (trang 69),
gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val; nhiệt độ T khớp
trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện (giữ nguyên như bộ khung):
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
Thêm:
    predict_views(model, loader, device, views_fn)   -> (filenames, y_true, [logits_view_k])
    model_soup(state_dicts)                          -> state_dict trung bình đều
    probs_to_logits(probs)                           -> log p (để temperature scaling sau TTA gộp prob)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


@torch.inference_mode()
def predict_views(model, loader, device, views_fn=None, amp: bool = False, half: bool = False):
    """Chạy model trên loader; views_fn(x) -> list K batch (None = 1 view). Trả về logit từng view."""
    model.eval()
    names, ys, outs = [], [], None
    for x, y, f in loader:
        x = x.to(device, non_blocking=True)
        if half:
            x = x.half()
        views = views_fn(x) if views_fn is not None else [x]
        if outs is None:
            outs = [[] for _ in views]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            for k, v in enumerate(views):
                outs[k].append(model(v).float().cpu())
        ys.append(torch.as_tensor(y))
        names.extend(f)
    return names, torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def predict_logits(model, loader, device, view=None, amp: bool = False, half: bool = False):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch trước khi đưa vào model (ví dụ view_hflip), hoặc None.
    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9]).
    """
    vf = (lambda x: [view(x)]) if view is not None else None
    names, y, outs = predict_views(model, loader, device, vf, amp=amp, half=half)
    return names, y, outs[0]


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) theo chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[3])


def views_hflip2(x):
    """TTA K = 2: ảnh gốc + bản lật ngang."""
    return [x, view_hflip(x)]


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop kích thước `crop` (4 góc + giữa) từ batch (N, C, H, W); flip=True thêm bản lật -> 10 view."""
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError(f"crop {crop} lớn hơn ảnh {h}x{w}")
    top, left = (h - crop) // 2, (w - crop) // 2
    crops = [x[..., :crop, :crop], x[..., :crop, w - crop:], x[..., h - crop:, :crop],
             x[..., h - crop:, w - crop:], x[..., top:top + crop, left:left + crop]]
    if flip:
        crops += [view_hflip(c) for c in crops]
    return crops


def views_multiscale(x, sizes):
    """Resize (bilinear, antialias) batch về từng kích thước trong `sizes`, trả về list các batch.

    CNN có global pooling nhận được mọi kích thước. ViT/DeiT được tạo với dynamic_img_size=True
    (model.py) nên cũng chạy được; Swin cần kích thước chia hết cho patch*window, nếu không sẽ lỗi:
    hàm gọi ngoài phải bắt lỗi và ghi "không áp dụng".
    """
    out = []
    for s in sizes:
        if tuple(x.shape[-2:]) == (s, s):
            out.append(x)
        else:
            out.append(F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True))
    return out


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Trả về xác suất (N, 9) đã chuẩn hoá (mỗi dòng cộng bằng 1).
    """
    arr = np.stack([np.asarray(l, dtype=np.float64) for l in logits_per_view])
    if space == "prob":
        p = np.mean([_softmax_np(a) for a in arr], axis=0)
    elif space == "logit":
        p = _softmax_np(arr.mean(0))
    else:
        raise ValueError(f"space={space!r} không hợp lệ (prob | logit)")
    return p / p.sum(1, keepdims=True)


def probs_to_logits(probs, eps: float = 1e-12):
    """log p: softmax(log p) = p, nên log p dùng được như logit cho temperature scaling."""
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), eps, None))


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file). Chi phí = số mô hình."""
    arr = np.stack([np.asarray(p, dtype=np.float64) for p in list_of_probs])
    p = arr.mean(0)
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels, max_iter: int = 200) -> float:
    """Tìm T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    Tối ưu log T bằng LBFGS (float64), khởi đầu từ tìm lưới thô trên [0.05, 20] để tránh cực tiểu
    xấu. Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 60))
    nll = [F.cross_entropy(z / t, y).item() for t in grid]
    log_t = torch.tensor([np.log(grid[int(np.argmin(nll))])], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T) dạng numpy (N, K)."""
    return _softmax_np(np.asarray(logits, dtype=np.float64) / float(T))


def model_soup(state_dicts):
    """Uniform soup: trung bình đều các state_dict cùng kiến trúc (Wortsman et al. 2022).

    Buffer số nguyên (num_batches_tracked) lấy của model đầu tiên.
    """
    out = {}
    for k, v in state_dicts[0].items():
        if v.is_floating_point():
            out[k] = sum(sd[k].float() for sd in state_dicts) / len(state_dicts)
        else:
            out[k] = v.clone()
    return out


def _fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """w' = gamma * w / sqrt(var + eps);  b' = beta + gamma * (b - mean) / sqrt(var + eps)."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode)
    fused = fused.to(conv.weight.device, conv.weight.dtype)
    with torch.no_grad():
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        fused.bias.copy_(bn.bias + (b - bn.running_mean) * scale)
    return fused


def _bn_replacement(bn: nn.Module) -> nn.Module:
    """BN thường -> Identity. BatchNormAct2d của timm (BN + drop + activation) -> giữ drop + act."""
    act = getattr(bn, "act", None)
    drop = getattr(bn, "drop", None)
    if act is None and drop is None:
        return nn.Identity()
    return nn.Sequential(drop if drop is not None else nn.Identity(), act if act is not None else nn.Identity())


def fuse_conv_bn(model, check: bool = True, img_size: int = 224, atol: float = 1e-3):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75).

    Cách tìm cặp: trong mỗi module cha, một Conv2d con đăng ký NGAY TRƯỚC một BatchNorm2d con có
    num_features == out_channels (đúng với ResNet/ResNeXt/EfficientNet/MobileNetV3 của timm). Vì
    thứ tự đăng ký không bảo đảm đúng luồng dữ liệu, hàm luôn KIỂM TRA SỐ: so đầu ra trước/sau trên
    một batch ngẫu nhiên và raise nếu sai số lớn nhất > atol. Trả về model đã gộp (bản sao, model gốc
    không đổi); số cặp đã gộp và sai số lớn nhất nằm ở thuộc tính `fused.fuse_info`.
    Kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm): trả lại bản sao, n_fused = 0.
    """
    ref = model.eval()
    fused = copy.deepcopy(ref).eval()
    n = 0
    for parent in fused.modules():
        names = list(parent._modules.keys())
        for a, b in zip(names, names[1:]):
            conv, bn = parent._modules[a], parent._modules[b]
            if (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)
                    and bn.num_features == conv.out_channels and bn.track_running_stats):
                parent._modules[a] = _fuse(conv, bn)
                parent._modules[b] = _bn_replacement(bn)
                n += 1
    info = {"n_fused": n, "max_abs_diff": 0.0}
    if check and n:
        dev = next(ref.parameters()).device
        x = torch.randn(4, 3, img_size, img_size, device=dev)
        with torch.inference_mode():
            a, b = ref(x).float(), fused(x).float()
        diff = float((a - b).abs().max())
        info["max_abs_diff"] = diff
        if diff > atol:
            raise RuntimeError(f"gộp BN sai: sai số lớn nhất {diff:.3e} > {atol}")
    fused.bn_fused = n > 0
    fused.fuse_info = info
    return fused
