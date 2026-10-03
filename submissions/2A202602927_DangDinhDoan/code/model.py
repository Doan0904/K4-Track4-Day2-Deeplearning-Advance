"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Hoàn thiện từ bộ khung starter/model.py.

Giao diện (giữ nguyên như bộ khung):
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
Thêm:
    set_train_mode(model)  : model.train() nhưng giữ backbone đóng băng ở eval (BN không cập nhật)
    weight_info(model)     : tag trọng số timm thực sự được tải, mean/std chuẩn hoá
"""
from __future__ import annotations

# Tên đầy đủ kèm tag trọng số timm (GHI vào results.xlsx). Chọn toàn bộ là trọng số chỉ huấn luyện
# trên ImageNet-1k để so sánh backbone công bằng hơn (không trộn in12k/in22k).
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50.a1_in1k",
    "resnext50": "resnext50_32x4d.a1h_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100.ra_in1k",      # mạng nhẹ
    "dinov2_small": "vit_small_patch14_dinov2.lvd142m",  # bonus: đóng băng + linear probe
}


def _is_vit_like(name: str) -> bool:
    return any(k in name for k in ("vit_", "deit_", "dinov2"))


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", drop_path_rate: float = 0.0,
                img_size: int | None = None):
    """Tạo model phân loại 9 lớp bằng timm (timm tự thay head mới, khởi tạo ngẫu nhiên).

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    `name` có thể là khoá của SUGGESTED_BACKBONES hoặc tên timm (có/không tag).
    ViT/DeiT/DINOv2 được tạo với dynamic_img_size=True để chạy được nhiều độ phân giải khi
    suy luận (I04); ở đúng img_size gốc kết quả không đổi.
    """
    import timm

    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init={init!r} không hợp lệ")
    timm_name = SUGGESTED_BACKBONES.get(name, name)
    kw = dict(pretrained=pretrained and init != "scratch", num_classes=num_classes, drop_rate=drop_rate)
    if drop_path_rate:
        kw["drop_path_rate"] = drop_path_rate
    if _is_vit_like(timm_name):
        kw["dynamic_img_size"] = True
        if img_size is not None and "dinov2" in timm_name:
            kw["img_size"] = img_size  # DINOv2 mặc định 518; nội suy pos-embed về img_size
    model = timm.create_model(timm_name, **kw)
    model.timm_name = timm_name
    model.init_mode = init
    if init == "frozen":
        freeze_backbone(model)
    return model


def weight_info(model) -> dict:
    """Thông tin trọng số thực sự dùng (tag timm, mean/std, input size) để ghi vào config/xlsx."""
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    return {
        "timm_name": getattr(model, "timm_name", None),
        "architecture": cfg.get("architecture"),
        "tag": cfg.get("tag"),
        "hf_hub_id": cfg.get("hf_hub_id"),
        "mean": list(cfg.get("mean", (0.485, 0.456, 0.406))),
        "std": list(cfg.get("std", (0.229, 0.224, 0.225))),
        "input_size": list(cfg.get("input_size", (3, 224, 224))),
        "pretrained": getattr(model, "init_mode", "finetune") != "scratch",
    }


def _head_param_ids(model) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head (model.get_classifier()).

    BatchNorm của backbone đóng băng phải ở chế độ eval, nếu không running_mean/var vẫn bị cập
    nhật theo batch dù trọng số đứng yên. Train loop gọi set_train_mode(model) thay cho model.train().
    """
    head = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(); nếu backbone đóng băng thì đưa mọi module ngoài head về eval."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        head_modules = set(id(m) for m in model.get_classifier().modules())
        for m in model.modules():
            if id(m) not in head_modules:
                m.training = False  # không gọi m.eval(): eval() đệ quy sẽ tắt cả head


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành các nhóm như slide Day 2, trang 52.

    - backbone, ndim > 1                         : lr_backbone, weight_decay
    - backbone, norm/bias (ndim <= 1) + pos_embed/cls_token (model.no_weight_decay())
                                                 : lr_backbone, weight_decay = 0
    - head mới, trọng số                         : lr_head, weight_decay
    - head mới, bias                             : lr_head, weight_decay = 0 (bias không decay)
    Bỏ qua tham số requires_grad == False. Nhóm rỗng bị bỏ.
    """
    head = _head_param_ids(model)
    skip_names = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    groups = {
        "backbone_decay": {"params": [], "lr": lr_backbone, "weight_decay": weight_decay},
        "backbone_no_decay": {"params": [], "lr": lr_backbone, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": lr_head, "weight_decay": weight_decay},
        "head_no_decay": {"params": [], "lr": lr_head, "weight_decay": 0.0},
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or name in skip_names or name.split(".")[-1] in skip_names
        part = "head" if id(p) in head else "backbone"
        groups[f"{part}_{'no_decay' if no_decay else 'decay'}"]["params"].append(p)
    out = []
    for gname, g in groups.items():
        if g["params"]:
            g["name"] = gname
            out.append(g)
    return out


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (MAC = FLOPs / 2).

    Công cụ: torch.utils.flop_counter.FlopCounterMode (đếm conv, matmul, attention/SDPA ở mức aten),
    chạy trên CPU với một ảnh, chế độ eval. Phép tính element-wise (BN, activation) không được đếm,
    giống quy ước của fvcore/slide; số có thể lệch vài phần trăm so với công cụ khác.
    """
    import copy

    import torch
    from torch.utils.flop_counter import FlopCounterMode

    m = copy.deepcopy(model).float().cpu().eval()
    x = torch.zeros(1, 3, img_size, img_size)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        m(x)
    return counter.get_total_flops() / 2 / 1e9


def has_batchnorm(model) -> bool:
    import torch.nn as nn
    return any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
