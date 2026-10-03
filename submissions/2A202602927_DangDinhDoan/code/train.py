"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Hoàn thiện từ bộ khung starter/train.py. MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H):
đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc.

Những gì một lần chạy lưu ra (run_dir = <out_dir>/<exp_id>/seed<k>/):
    config.json        cấu hình đầy đủ + tag trọng số timm + version thư viện + GPU
    history.csv        log theo epoch (loss/F1/acc train và val, ECE val, LR, thời gian)
    lr_trace.csv       LR theo từng bước (để vẽ warmup + cosine)
    val_outputs.npz    tên file, nhãn, logit val của checkpoint tốt nhất
    val_predictions.csv  định dạng eval.py (softmax)
    summary.json       tóm tắt: epoch tốt nhất, chỉ số val, thời gian/epoch, params, GMAC, độ trễ sơ bộ
    <ckpt_dir>/<exp_id>_seed<k>_best.pt   trọng số tốt nhất (KHÔNG commit)
    <curves_dir>/<exp_id>_<desc>.png      đường cong (mọi seed của exp_id vẽ chồng)
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _find_eval_dir() -> Path:
    """Thư mục chứa eval.py của repo gốc (đi ngược lên từ code/), hoặc biến môi trường LAB_EVAL_DIR."""
    env = os.environ.get("LAB_EVAL_DIR")
    if env:
        return Path(env)
    for p in [HERE, *HERE.parents]:
        if (p / "eval.py").exists() and (p / "RUBRIC.md").exists():
            return p
    raise FileNotFoundError("không tìm thấy eval.py của repo gốc; đặt LAB_EVAL_DIR")


EVAL_DIR = _find_eval_dir()
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from eval import compute_metrics, save_predictions  # noqa: E402  (eval.py gốc, không sửa)


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    desc: str = ""                    # mô tả ngắn cho tên ảnh curves/<exp_id>_<desc>.png
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"        # khoá của model.SUGGESTED_BACKBONES hoặc tên timm (kèm tag)
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | flipv
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    mix_prob: float = 1.0             # xác suất trộn mỗi batch khi mix != None
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0      # loss="ls" mà để 0 thì dùng 0.1
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None   # ce_weighted: None/0 = 1/n_c; > 0 = class-balanced
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    opt: str = "adamw"                # adamw | sgd
    momentum: float = 0.9             # chỉ dùng với sgd
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    clip_grad: float | None = None
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = True
    deterministic: bool = False       # True: cudnn.deterministic (chậm hơn)
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    cache_path: str | None = None     # file .npy của dataset.build_cache (tuỳ chọn, tăng tốc)
    out_dir: str = "runs"             # config.json, history.csv, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    ckpt_dir: str = "ckpt"            # checkpoint (không commit)
    curves_dir: str = "curves"        # ảnh đường cong (nộp cùng bài)
    # --- tiện ích ---
    bench_latency: bool = True        # đo độ trễ sơ bộ batch 1 (Bước 1) sau khi train
    max_train_steps: int | None = None  # chỉ để chạy thử nhanh (smoke test)
    eval_limit: int | None = None       # chỉ để chạy thử nhanh: dùng N ảnh val đầu
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def ckpt_path(cfg: Config, kind: str = "best") -> Path:
    return Path(cfg.ckpt_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{kind}.pt"


def curve_path(cfg: Config) -> Path:
    name = f"{cfg.exp_id}_{cfg.desc}" if cfg.desc else cfg.exp_id
    return Path(cfg.curves_dir) / f"{name}.png"


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Cố định random, numpy, torch (CPU và CUDA). Worker DataLoader được seed trong make_loader.

    Mức tái lập: cùng seed -> cùng khởi tạo head, cùng thứ tự batch, cùng augmentation. Với
    deterministic=False (mặc định, nhanh hơn) cuDNN được chọn thuật toán tự do (benchmark=True) và
    atomicAdd trên GPU, nên hai lần chạy cùng seed có thể lệch nhẹ ở chữ số cuối.
    """
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD + momentum) với nhóm tham số của model.param_groups (không decay norm/bias)."""
    import torch
    from model import param_groups

    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    for g in groups:
        g["initial_lr"] = g["lr"]
    if cfg.opt == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr_backbone, betas=(0.9, 0.999))
    if cfg.opt == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr_backbone, momentum=cfg.momentum, nesterov=True)
    raise ValueError(f"opt={cfg.opt!r} không hợp lệ (adamw | sgd)")


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Hệ số LR tại một bước: warmup tuyến tính (step+1)/warmup, rồi cosine từ 1 về 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55). Cập nhật THEO BƯỚC (mỗi iteration)."""
    import torch
    total = cfg.epochs * steps_per_epoch
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total, warm))


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    - Giữ một bản sao riêng `self.module` (eval mode) để đánh giá bằng trọng số EMA.
    - EMA áp dụng cho mọi tensor số thực trong state_dict, kể cả buffer BatchNorm
      (running_mean/var) giống timm ModelEmaV2; buffer số nguyên (num_batches_tracked) được chép.
    - Hệ số thực tế d_t = min(decay, (1 + t) / (10 + t)) để EMA không bị kéo về trọng số ban đầu
      ở những bước đầu.
    """

    def __init__(self, model, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.num_updates = 0

    def update(self, model) -> None:
        import torch
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            src = model.state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(src[k].detach().to(v.dtype), alpha=1.0 - d)
                else:
                    v.copy_(src[k])

    def copy_to(self, model) -> None:
        model.load_state_dict(self.module.state_dict())

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "num_updates": self.num_updates, "decay": self.decay}

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd["module"])
        self.num_updates = sd["num_updates"]


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, rng: np.random.Generator | None = None,
                    lr_trace: list | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "train_acc", "train_macro_f1", "lr", "steps"}.

    - set_train_mode: model.train() nhưng backbone đóng băng (init="frozen") vẫn ở eval
    - cfg.mix: mix_batch rồi mixed_loss; khi đó train_acc/F1 tính với nhãn y_a (chỉ tham khảo)
    - AMP (autocast fp16 + GradScaler), clip gradient nếu cfg.clip_grad, scheduler.step() mỗi bước
    - EMA cập nhật sau mỗi bước tối ưu
    """
    import torch
    from losses import mix_batch, mixed_loss
    from model import set_train_mode

    set_train_mode(model)
    rng = rng or np.random.default_rng(cfg.seed)
    tot_loss, n_seen, steps = 0.0, 0, 0
    preds, trues = [], []
    for x, y, _ in loader:
        if cfg.max_train_steps is not None and steps >= cfg.max_train_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        targets = y
        if cfg.mix and rng.random() < cfg.mix_prob:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix, rng)
        with torch.autocast(device.type, dtype=torch.float16, enabled=cfg.amp and device.type == "cuda"):
            logits = model(x)
            loss = mixed_loss(criterion, logits.float(), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss không hữu hạn ở bước {steps}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.clip_grad:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        if lr_trace is not None:
            lr_trace.append([g["lr"] for g in optimizer.param_groups])
        bs = y.size(0)
        tot_loss += loss.item() * bs
        n_seen += bs
        steps += 1
        preds.append(logits.detach().argmax(1).cpu())
        trues.append(y.cpu())
    p = torch.cat(preds).numpy()
    t = torch.cat(trues).numpy()
    m = compute_metrics(t, p, np.eye(9)[p])
    return {"train_loss": tot_loss / max(1, n_seen), "train_acc": m["top1"], "train_macro_f1": m["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"], "steps": steps}


def evaluate(model, loader, criterion, device, amp: bool = True, channels_last: bool = False):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader. Loss val luôn là CE không trọng số (truyền criterion = CE) để so
    sánh được giữa các thí nghiệm dùng loss train khác nhau.
    """
    import torch
    model.eval()
    names, ys, outs = [], [], []
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(x)
            outs.append(logits.float().cpu())
            ys.append(torch.as_tensor(y))
            names.extend(f)
    logits = torch.cat(outs)
    y = torch.cat(ys)
    loss = float(criterion(logits, y).item())
    return names, y.numpy(), logits.numpy(), loss


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def metrics_from_logits(y, logits) -> dict:
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    return compute_metrics(np.asarray(y), probs.argmax(1), probs)


def plot_curves(history, path: str | Path, title: str, lr_trace=None, steps_per_epoch: int | None = None) -> None:
    """Vẽ đường cong training -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    history: list[dict] (một seed) hoặc dict {seed: list[dict]} (nhiều seed, vẽ chồng).
    3 ô: (1) loss train/val, (2) macro-F1 val (+ EMA nếu có) và train, (3) LR theo bước.
    Đánh dấu epoch tốt nhất (macro-F1 val) của từng seed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = history if isinstance(history, dict) else {None: history}
    lr_traces = lr_trace if isinstance(lr_trace, dict) else {None: lr_trace}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (seed, hist) in enumerate(runs.items()):
        if not hist:
            continue
        c = colors[i % len(colors)]
        sfx = f" (seed {seed})" if seed is not None else ""
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], "-o", ms=3, color=c, label=f"train loss{sfx}")
        axes[0].plot(ep, [h["val_loss"] for h in hist], "--s", ms=3, color=c, label=f"val loss (CE){sfx}")
        axes[1].plot(ep, [h["val_macro_f1"] for h in hist], "-o", ms=3, color=c, label=f"val macro-F1{sfx}")
        if "val_macro_f1_ema" in hist[0]:
            axes[1].plot(ep, [h["val_macro_f1_ema"] for h in hist], "-^", ms=3, color=c, alpha=.8,
                         label=f"val macro-F1 EMA{sfx}")
        axes[1].plot(ep, [h["train_macro_f1"] for h in hist], ":", color=c, alpha=.7, label=f"train macro-F1{sfx}")
        sel = "val_macro_f1_ema" if "val_macro_f1_ema" in hist[0] else "val_macro_f1"
        best = max(range(len(hist)), key=lambda j: (hist[j][sel], -j))
        axes[1].axvline(hist[best]["epoch"], color=c, alpha=.25, lw=4)
    for seed, tr in lr_traces.items():
        if tr is None or len(tr) == 0:
            continue
        tr = np.asarray(tr)
        x = np.arange(1, len(tr) + 1)
        if steps_per_epoch:
            x = x / steps_per_epoch
        axes[2].plot(x, tr[:, 0], label="LR nhóm backbone" if tr.shape[1] > 1 else "LR")
        if tr.shape[1] > 1:
            axes[2].plot(x, tr[:, -1], label="LR nhóm head")
        break  # LR giống nhau giữa các seed
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    axes[1].set(xlabel="epoch", ylabel="macro-F1", title="Macro-F1 (vạch mờ = epoch được chọn)")
    axes[2].set(xlabel="epoch" if steps_per_epoch else "bước", ylabel="learning rate", title="LR theo bước")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    for a in axes:
        a.grid(alpha=.3)
        if a.lines:
            a.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_exp_curves(cfg: Config) -> Path:
    """Vẽ lại ảnh của exp_id với mọi seed đã chạy xong (đọc history.csv/lr_trace.csv của từng seed)."""
    import pandas as pd
    root = Path(cfg.out_dir) / cfg.exp_id
    hists, traces, spe = {}, {}, None
    for d in sorted(root.glob("seed*")):
        if (d / "history.csv").exists():
            s = int(d.name[4:])
            hists[s] = pd.read_csv(d / "history.csv").to_dict("records")
            if (d / "lr_trace.csv").exists():
                traces[s] = pd.read_csv(d / "lr_trace.csv").to_numpy()
            if (d / "summary.json").exists():
                spe = json.loads((d / "summary.json").read_text()).get("steps_per_epoch", spe)
    if len(hists) == 1:
        (s, h), = hists.items()
        hists, traces = {s: h}, {s: traces.get(s)}
    path = curve_path(cfg)
    bk = json.loads((root / f"seed{min(hists)}" / "config.json").read_text())["weights"]["timm_name"]
    plot_curves(hists, path, f"{cfg.exp_id} {cfg.desc} | {bk} | {len(hists)} seed", traces, spe)
    return path


def env_info() -> dict:
    import torch
    import torchvision
    try:
        import timm
        timm_v = timm.__version__
    except ImportError:
        timm_v = None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "timm": timm_v, "numpy": np.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "platform": platform.platform()}


def build_eval_loader(cfg: Config, df, mean, std, img_size: int | None = None, crop_pct: float | None = None,
                      batch_size: int | None = None):
    import dataset as D
    tf = D.eval_transform(img_size or cfg.img_size, mean, std, crop_pct or D.EVAL_CROP_PCT)
    return D.make_loader(df, cfg.images_dir, tf, batch_size or cfg.batch_size * 2, train=False,
                         num_workers=cfg.num_workers, cache_path=cfg.cache_path, seed=cfg.seed)


def load_trained(cfg: Config, device=None, kind: str = "best"):
    """Dựng lại model của một lần chạy và nạp checkpoint tốt nhất. Trả về (model, mean, std)."""
    import torch
    from model import build_model

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.backbone, pretrained=False, num_classes=9, drop_rate=cfg.drop_rate,
                        init="finetune", drop_path_rate=cfg.drop_path_rate, img_size=cfg.img_size)
    sd = torch.load(ckpt_path(cfg, kind), map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.to(device).eval()
    conf = json.loads((run_dir(cfg) / "config.json").read_text())
    return model, tuple(conf["weights"]["mean"]), tuple(conf["weights"]["std"])


def load_config(exp_id: str, seed: int, out_dir: str) -> Config:
    conf = json.loads((Path(out_dir) / exp_id / f"seed{seed}" / "config.json").read_text())
    fields = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in conf["config"].items() if k in fields})


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    1. set_seed; tạo run_dir(cfg); ghi config.json (cấu hình + tag trọng số + version thư viện)
       Nếu summary.json đã có (đã chạy xong) thì bỏ qua và trả lại tóm tắt cũ. Nếu có checkpoint
       `_last.pt` (phiên trước bị ngắt) thì chạy tiếp từ epoch đó.
    2. dataset.load_split + dataset.check_split (dừng nếu vi phạm S1-S6)
    3. train/val loader (test loader chỉ tạo khi cfg.save_test_predictions)
    4. model, criterion, optimizer, scheduler, scaler, EMA
    5. mỗi epoch: train_one_epoch -> evaluate(val) -> history; giữ checkpoint có MACRO-F1 VAL cao
       nhất (hòa thì giữ epoch sớm hơn). Có EMA thì chọn và lưu theo trọng số EMA.
    6. cuối: nạp checkpoint tốt nhất, lưu val logits + val_predictions.csv
    7. NẾU cfg.save_test_predictions: đánh giá test đúng MỘT lần, lưu logits + pred_path(cfg, "test")
    8. history.csv, đường cong, summary.json
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    import pandas as pd
    import torch
    import torch.nn as nn

    import dataset as D
    from losses import build_criterion, class_weights
    from model import build_model, count_gmacs, count_params, weight_info

    rd = run_dir(cfg)
    if (rd / "summary.json").exists():
        print(f"[{cfg.exp_id} seed{cfg.seed}] đã chạy xong, bỏ qua (xoá {rd}/summary.json để chạy lại)")
        return json.loads((rd / "summary.json").read_text())
    rd.mkdir(parents=True, exist_ok=True)
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed, cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    # 2. split + kiểm tra
    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    split_report = D.check_split(train_df, val_df, test_df, cfg.images_dir,
                                 Path(cfg.labels_dir) / "labels.csv", verbose=False)
    if cfg.eval_limit:
        val_df = val_df.iloc[: cfg.eval_limit].reset_index(drop=True)

    # 4a. model (trước transform để lấy mean/std của trọng số)
    model = build_model(cfg.backbone, pretrained=True, num_classes=9, drop_rate=cfg.drop_rate,
                        init=cfg.init, drop_path_rate=cfg.drop_path_rate, img_size=cfg.img_size)
    winfo = weight_info(model)
    mean, std = tuple(winfo["mean"]), tuple(winfo["std"])
    n_params = count_params(model)
    try:
        gmacs = count_gmacs(model, cfg.img_size)
    except Exception as e:  # noqa: BLE001
        print("không đếm được GMAC:", e)
        gmacs = float("nan")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    model.to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    (rd / "config.json").write_text(json.dumps({
        "config": dataclasses.asdict(cfg), "weights": winfo, "env": env_info(),
        "params_M": n_params, "trainable_params_M": n_trainable, "gmacs": gmacs,
        "split_check": {k: split_report[k] for k in ("n", "overlap", "union", "missing_files")},
    }, indent=2, ensure_ascii=False))

    # 3. loader
    train_tf = D.build_transforms(True, cfg.img_size, cfg.aug, mean, std)
    train_loader = D.make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size, train=True,
                                 sampler=cfg.sampler, num_workers=cfg.num_workers,
                                 cache_path=cfg.cache_path, seed=cfg.seed)
    val_loader = build_eval_loader(cfg, val_df, mean, std)
    steps_per_epoch = len(train_loader) if cfg.max_train_steps is None else min(len(train_loader), cfg.max_train_steps)

    # 4b. loss, optimizer, scheduler, scaler, EMA
    counts = np.bincount(train_df["Label"].to_numpy(), minlength=9)
    crit_kw = {"smoothing": cfg.label_smoothing or 0.1, "gamma": cfg.focal_gamma}
    if cfg.loss == "ce_weighted":
        crit_kw["weight"] = class_weights(counts, cfg.class_weight_beta or 0.0)
    criterion = build_criterion(cfg.loss, **crit_kw).to(device)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    rng = np.random.default_rng(cfg.seed + 12345)

    history, lr_trace = [], []
    best = {"f1": -1.0, "epoch": -1, "logits": None}
    start_epoch = 1
    last = ckpt_path(cfg, "last")
    if last.exists():  # chạy tiếp sau khi phiên bị ngắt
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        if ema is not None and st.get("ema"):
            ema.load_state_dict(st["ema"])
        history, lr_trace, best = st["history"], st["lr_trace"], st["best"]
        rng = st["rng_np"]
        torch.set_rng_state(st["rng_torch"])
        if torch.cuda.is_available() and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        start_epoch = st["epoch"] + 1
        print(f"[{cfg.exp_id} seed{cfg.seed}] chạy tiếp từ epoch {start_epoch}")

    # 5. vòng epoch
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device,
                             ema=ema, rng=rng, lr_trace=lr_trace)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.time() - t0
        t1 = time.time()
        _, yv, lv, vloss = evaluate(model, val_loader, val_criterion, device, cfg.amp, cfg.channels_last)
        mv = metrics_from_logits(yv, lv)
        row = {"epoch": epoch, **{k: tr[k] for k in ("train_loss", "train_acc", "train_macro_f1")},
               "val_loss": vloss, "val_top1": mv["top1"], "val_macro_f1": mv["macro_f1"],
               "val_balanced_acc": mv["balanced_acc"], "val_ece": mv["ece"],
               "lr_backbone_end": optimizer.param_groups[0]["lr"], "lr_head_end": optimizer.param_groups[-1]["lr"]}
        sel_f1, sel_logits, sel_state = mv["macro_f1"], lv, model
        if ema is not None:
            _, _, le, eloss = evaluate(ema.module, val_loader, val_criterion, device, cfg.amp, cfg.channels_last)
            me = metrics_from_logits(yv, le)
            row.update({"val_loss_ema": eloss, "val_top1_ema": me["top1"], "val_macro_f1_ema": me["macro_f1"],
                        "val_ece_ema": me["ece"]})
            sel_f1, sel_logits, sel_state = me["macro_f1"], le, ema.module
        row["time_train_s"] = t_train
        row["time_val_s"] = time.time() - t1
        history.append(row)
        if sel_f1 > best["f1"]:  # strict: hòa thì giữ epoch sớm hơn
            best = {"f1": sel_f1, "epoch": epoch, "logits": sel_logits}
            torch.save({"model": {k: v.detach().cpu() for k, v in sel_state.state_dict().items()},
                        "epoch": epoch, "val_macro_f1": sel_f1}, ckpt_path(cfg, "best"))
        print(f"[{cfg.exp_id} seed{cfg.seed}] ep {epoch:02d}/{cfg.epochs} loss {tr['train_loss']:.4f} "
              f"| val loss {vloss:.4f} F1 {mv['macro_f1']:.4f} top1 {mv['top1']:.4f}"
              + (f" | EMA F1 {row['val_macro_f1_ema']:.4f}" if ema is not None else "")
              + f" | {t_train:.0f}s+{row['time_val_s']:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None, "history": history,
                    "lr_trace": lr_trace, "best": best, "epoch": epoch, "rng_np": rng,
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   last)

    # 6. checkpoint tốt nhất -> logit val
    eval_model = ema.module if ema is not None else model
    eval_model.load_state_dict(torch.load(ckpt_path(cfg, "best"), map_location=device, weights_only=False)["model"])
    names_v, yv, lv, _ = evaluate(eval_model, val_loader, val_criterion, device, cfg.amp, cfg.channels_last)
    np.savez_compressed(rd / "val_outputs.npz", filenames=np.array(names_v), y=yv, logits=lv)
    save_predictions(rd / "val_predictions.csv", names_v, yv, _softmax(lv.astype(np.float64)))
    mv = metrics_from_logits(yv, lv)

    # 7. test: chỉ ở Bước 4, đúng một lần
    test_info = None
    if cfg.save_test_predictions:
        test_loader = build_eval_loader(cfg, test_df, mean, std)
        names_t, yt, lt, _ = evaluate(eval_model, test_loader, val_criterion, device, cfg.amp, cfg.channels_last)
        np.savez_compressed(rd / "test_outputs.npz", filenames=np.array(names_t), y=yt, logits=lt)
        test_info = str(save_predictions(pred_path(cfg, "test"), names_t, yt, _softmax(lt.astype(np.float64))))

    # độ trễ sơ bộ (Bước 1): batch 1, FP32, một lần đo 50 lượt sau 10 lượt warmup
    lat = None
    if cfg.bench_latency and device.type == "cuda":
        from benchmark import latency_report
        try:
            lat = latency_report(eval_model.to(memory_format=torch.contiguous_format), 1, cfg.img_size,
                                 "fp32", "cuda", warmup=10, iters=50)
        except Exception as e:  # noqa: BLE001
            print("không đo được độ trễ:", e)

    # 8. log, đường cong, tóm tắt
    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    lr_cols = [g.get("name", f"g{i}") for i, g in enumerate(optimizer.param_groups)]
    pd.DataFrame(lr_trace, columns=lr_cols).to_csv(rd / "lr_trace.csv", index=False)
    hard = {"Chinee Apple": 0, "Snake Weed": 7}
    summary = {
        "exp_id": cfg.exp_id, "desc": cfg.desc, "seed": cfg.seed, "backbone": cfg.backbone,
        "timm_name": winfo["timm_name"], "tag": winfo["tag"], "init": cfg.init,
        "best_epoch": best["epoch"], "epochs": cfg.epochs, "selected_on": "ema" if ema is not None else "raw",
        "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"], "val_balanced_acc": mv["balanced_acc"],
        "val_ece": mv["ece"], "val_nll": mv["nll"],
        "val_f1_per_class": [float(v) for v in mv["f1"]],
        "val_recall_per_class": [float(v) for v in mv["recall"]],
        **{f"val_f1_{k.replace(' ', '_').lower()}": float(mv["f1"][i]) for k, i in hard.items()},
        "best_raw_val_macro_f1": max(h["val_macro_f1"] for h in history),
        "best_ema_val_macro_f1": max(h["val_macro_f1_ema"] for h in history) if ema is not None else None,
        "params_M": n_params, "trainable_params_M": n_trainable, "gmacs": gmacs,
        "time_train_per_epoch_s": float(np.mean([h["time_train_s"] for h in history])),
        "time_val_per_epoch_s": float(np.mean([h["time_val_s"] for h in history])),
        "time_total_s": time.time() - t_start, "steps_per_epoch": steps_per_epoch,
        "latency_b1_fp32": lat, "test_predictions": test_info, "gpu": env_info()["gpu"],
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    try:
        summary["curve"] = str(plot_exp_curves(cfg))
    except Exception as e:  # noqa: BLE001
        print("không vẽ được đường cong:", e)
    (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if last.exists():
        last.unlink()
    print(f"[{cfg.exp_id} seed{cfg.seed}] XONG: best epoch {best['epoch']}, val macro-F1 {mv['macro_f1']:.4f}, "
          f"top1 {mv['top1']:.4f}, {summary['time_train_per_epoch_s']:.0f}s/epoch", flush=True)
    del model, ema, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _cast(value: str, type_str: str):
    t = str(type_str).replace(" ", "")
    opts = t.split("|")
    if value.lower() in ("none", "null") and "None" in opts:
        return None
    base = [o for o in opts if o != "None"][0]
    if base == "bool":
        if value.lower() in ("1", "true", "yes", "y"):
            return True
        if value.lower() in ("0", "false", "no", "n"):
            return False
        raise ValueError(f"không hiểu giá trị bool: {value!r}")
    if base == "int":
        return int(value)
    if base == "float":
        return float(value)
    return value


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    types = {f.name: f.type for f in dataclasses.fields(Config)}
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"override phải có dạng KEY=VALUE, nhận {p!r}")
        k, v = p.split("=", 1)
        k = k.strip()
        if k not in types:
            raise KeyError(f"Config không có field {k!r}; các field: {sorted(types)}")
        out[k] = _cast(v.strip(), types[k])
    return out


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ghi đè field của Config")
    ap.add_argument("--print-config", action="store_true", help="chỉ in cấu hình rồi thoát")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(dataclasses.asdict(cfg), indent=2, ensure_ascii=False))
    if args.print_config:
        return
    res = run(cfg)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
