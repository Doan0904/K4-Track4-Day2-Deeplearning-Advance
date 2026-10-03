"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Hoàn thiện từ bộ khung starter/benchmark.py.

Quy tắc đo áp dụng ở mọi hàm:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn đo
  - >= 50 lần đo (mặc định 100), báo cáo p50, p95, p99 và mean
  - ghi rõ GPU, dtype (fp32 / amp / fp16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - KHÔNG tính tiền xử lý (giải mã JPEG, resize, chuẩn hoá trên CPU): đầu vào là tensor đã nằm trên
    GPU. Với TTA, việc tạo các view (lật, cắt, resize) chạy trên GPU và ĐƯỢC tính.
"""
from __future__ import annotations

import time

import numpy as np


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian `fn()` (không tham số), trả về mili-giây: p50/p95/p99/mean/std/min/max."""
    if iters < 50:
        raise ValueError("cần >= 50 lần đo (GUIDE.md mục 4.1)")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = np.empty(iters)
    for i in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times[i] = (time.perf_counter() - t0) * 1000.0
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "mean": float(times.mean()),
            "std": float(times.std(ddof=1)), "min": float(times.min()), "max": float(times.max()),
            "n": iters, "warmup": warmup}


def _env(device: str) -> dict:
    import torch
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return {"gpu": gpu, "torch": torch.__version__,
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}


def _prepare(model, dtype: str, device: str):
    import copy

    import torch
    m = copy.deepcopy(model).to(device).eval()
    if dtype == "fp16":
        m = m.half()
    elif dtype not in ("fp32", "amp"):
        raise ValueError(f"dtype={dtype!r} không hợp lệ (fp32 | amp | fp16)")
    in_dtype = torch.float16 if dtype == "fp16" else torch.float32
    return m, in_dtype


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, bn_fused: bool = False, label: str = "") -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict ghi thẳng được vào sheet `Latency` của results.xlsx.
    dtype: "fp32" | "amp" (autocast fp16) | "fp16" (model.half()).
    """
    import torch

    m, in_dtype = _prepare(model, dtype, device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=in_dtype)
    sync = torch.cuda.synchronize if device.startswith("cuda") else None

    def fn():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=dtype == "amp"):
            m(x)

    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    return {"config": label, **_env(device), "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "bn_fused": bn_fused, "includes_preprocessing": False,
            **{k: r[k] for k in ("p50", "p95", "p99", "mean", "std", "n", "warmup")},
            "images_per_s": batch_size / (r["p50"] / 1000.0)}


def tta_latency(model, k_views: int = 2, views_fn=None, img_size: int = 224, input_size: int | None = None,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 100,
                mode: str = "stacked", label: str = "") -> dict:
    """Độ trễ TTA ở batch 1 (một ảnh, K view). Đo thật để so với K * p50 của 1 view (slide trang 63).

    - views_fn(x) -> list K batch; mặc định K bản sao (chỉ để đo chi phí K lượt).
    - input_size: kích thước ảnh đầu vào trước khi tạo view (ví dụ 256 cho 5-crop 224).
    - mode="stacked": ghép K view thành một batch K rồi chạy một forward (cách nhanh nhất ở batch 1).
      mode="sequential": K forward liên tiếp, mỗi lần 1 view.
    Thời gian tạo view trên GPU và softmax/trung bình được tính; tiền xử lý CPU không tính.
    """
    import torch

    m, in_dtype = _prepare(model, dtype, device)
    x = torch.randn(1, 3, input_size or img_size, input_size or img_size, device=device, dtype=in_dtype)
    views_fn = views_fn or (lambda t: [t] * k_views)
    sync = torch.cuda.synchronize if device.startswith("cuda") else None

    def fn():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=dtype == "amp"):
            views = views_fn(x)
            if mode == "stacked" and len({tuple(v.shape[-2:]) for v in views}) == 1:
                out = m(torch.cat(views, 0)).float().softmax(-1).mean(0)
            else:
                out = torch.stack([m(v).float().softmax(-1) for v in views]).mean(0)
            return out

    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    return {"config": label, **_env(device), "dtype": dtype, "batch": 1, "img_size": img_size,
            "k_views": k_views, "mode": mode, "bn_fused": False, "includes_preprocessing": False,
            **{k: r[k] for k in ("p50", "p95", "p99", "mean", "std", "n", "warmup")},
            "images_per_s": 1.0 / (r["p50"] / 1000.0)}


def multi_model_latency(models, img_size=224, dtype: str = "fp32", device: str = "cuda",
                        warmup: int = 10, iters: int = 100, label: str = "") -> dict:
    """Độ trễ ensemble ở batch 1: chạy lần lượt mọi model trên cùng ảnh rồi trung bình xác suất.

    `img_size` là một số (mọi model cùng kích thước) hoặc list độ dài len(models): mỗi model chạy ở đúng
    kích thước nó được huấn luyện (ví dụ Swin-T cố định 224, ConvNeXt công thức T15 dùng 256).
    """
    import torch

    sizes = [img_size] * len(models) if isinstance(img_size, int) else list(img_size)
    assert len(sizes) == len(models)
    prepared = [_prepare(m, dtype, device)[0] for m in models]
    in_dtype = torch.float16 if dtype == "fp16" else torch.float32
    xs = [torch.randn(1, 3, s, s, device=device, dtype=in_dtype) for s in sizes]
    sync = torch.cuda.synchronize if device.startswith("cuda") else None

    def fn():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=dtype == "amp"):
            return torch.stack([m(x).float().softmax(-1) for m, x in zip(prepared, xs)]).mean(0)

    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    return {"config": label, **_env(device), "dtype": dtype, "batch": 1, "img_size": "/".join(map(str, sizes)),
            "k_models": len(models), "bn_fused": False, "includes_preprocessing": False,
            **{k: r[k] for k in ("p50", "p95", "p99", "mean", "std", "n", "warmup")},
            "images_per_s": 1.0 / (r["p50"] / 1000.0)}
