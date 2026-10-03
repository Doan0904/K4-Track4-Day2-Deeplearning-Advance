"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Hoàn thiện từ bộ khung starter/losses.py. Liên hệ slide Day 2: label smoothing (trang 56),
focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện (giữ nguyên như bộ khung):
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
Các kiểm tra (focal gamma=0 == CE, label smoothing eps=0 == CE, CutMix lam = diện tích thật...)
nằm ở test_code.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`:

      - "ce"          : nn.CrossEntropyLoss()
      - "ls"          : LabelSmoothingCE(smoothing)               (tự cài đặt)
      - "focal"       : FocalLoss(gamma, alpha)
      - "ce_weighted" : nn.CrossEntropyLoss(weight=weight)       (weight từ class_weights(train))
    kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        w = kw.get("weight")
        if w is None:
            raise ValueError("ce_weighted cần weight=class_weights(số ảnh mỗi lớp của train)")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32))
    raise ValueError(f"loss={kind!r} không hợp lệ (ce | ls | focal | ce_weighted)")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Tự cài đặt: loss = (1 - eps) * NLL(y) + eps * mean_k(-log p_k). Kết quả trùng với
    nn.CrossEntropyLoss(label_smoothing=eps) (đã kiểm tra trong test_code.py); eps = 0 cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target.view(-1, 1)).squeeze(1)
        smooth = -logp.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth).mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    alpha: None hoặc vector trọng số theo lớp (độ dài K). Lấy trung bình theo batch.
    gamma = 0 và alpha = None cho đúng cross-entropy (test_code.py kiểm tra sai số < 1e-6).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_t = logp.gather(1, target.view(-1, 1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t).clamp(min=0) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: w_c ∝ 1 / n_c, chuẩn hoá về trung bình 1
    - beta > 0: class-balanced, w_c = (1 - beta) / (1 - beta ** n_c) (Cui et al. arXiv:1901.05555),
      chuẩn hoá tổng trọng số về số lớp (tức cũng là trung bình 1)
    """
    n = np.asarray(counts, dtype=np.float64)
    if (n <= 0).any():
        raise ValueError("mọi lớp phải có ít nhất 1 ảnh trong train")
    if beta and beta > 0:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
    else:
        w = 1.0 / n
    w = w / w.sum() * len(n)
    return torch.tensor(w, dtype=torch.float32)


def rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator):
    """Hộp CutMix có diện tích (1 - lam) * H * W, tâm ngẫu nhiên, cắt theo biên ảnh."""
    cut = np.sqrt(1.0 - lam)
    ch, cw = int(round(h * cut)), int(round(w * cut))
    cy, cx = int(rng.integers(h)), int(rng.integers(w))
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch - ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw - cw // 2, 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix", rng: np.random.Generator | None = None):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha); perm là hoán vị ngẫu nhiên của batch (torch RNG -> theo seed)
    - mode="mixup":  x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": dán một hộp của x[perm] vào x, rồi đặt lại lam = 1 - (diện tích hộp THẬT sau
      khi bị cắt ở biên) / (H * W)  (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]
    """
    if rng is None:
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        y1, y2, x1, x2 = rand_bbox(h, w, lam, rng)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / float(h * w)
    else:
        raise ValueError(f"mode={mode!r} không hợp lệ (mixup | cutmix)")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    Nếu `targets` là tensor nhãn thường thì gọi thẳng criterion. Accuracy trên batch đã trộn không
    còn nghĩa bình thường; đánh giá bằng val.
    """
    if isinstance(targets, (tuple, list)):
        y_a, y_b, lam = targets
        return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
    return criterion(logits, targets)
