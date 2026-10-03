"""experiments.py - danh sách thí nghiệm, luật chọn tự động (CHỈ dựa trên val), chạy nhiều GPU.

Notebook lab_day2.ipynb gọi các bước theo thứ tự:

    lab = Lab(out_root, images_dir, labels_dir, cache_path)
    lab.stage1_backbones()      # B01..B07 (+ B08 DINOv2 linear probe, bonus)    -> chọn backbone
    lab.select_backbone()
    lab.stage2_ablation()       # T00 x 3 seed + T01..T14 (mỗi lần khác T00 đúng 1 yếu tố)
    lab.select_combo(); lab.stage2_combo(); lab.select_recipe()               # T15 = kết hợp
    lab.stage3_inference()      # I00..I08 trên val + độ trễ                     -> chọn suy luận
    lab.select_inference()
    lab.stage4_final()          # F01 x 3 seed; test đúng MỘT lần mỗi seed; eval.py score/grade
    lab.stage5_bonus()          # lệch phân phối, Grad-CAM, ảnh lỗi (+ results.xlsx ở analysis.py)

Mọi quyết định ghi vào <out_root>/decisions.json kèm số liệu làm căn cứ; chạy lại notebook sẽ dùng
lại quyết định đã ghi (không chọn lại), trừ khi truyền force=... Không bước nào đọc nhãn test trước
stage4_final; stage4 chỉ tính dự đoán test cho các cấu hình đã chốt, rồi mới gọi eval.py.

Chạy song song: nếu có >= 2 GPU (Kaggle T4 x2), mỗi GPU một tiến trình worker, các worker lấy việc
từ cùng một hàng đợi (file claim). Một GPU thì chạy ngay trong tiến trình notebook.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train as TR  # noqa: E402  (cũng thêm thư mục eval.py vào sys.path)
from train import Config  # noqa: E402

# --------------------------------------------------------------------------- #
# Danh sách thí nghiệm
# --------------------------------------------------------------------------- #
BACKBONES = [  # (exp_id, khoá model.SUGGESTED_BACKBONES); ràng buộc GUIDE mục 2.1
    ("B01", "resnet50"),         # ResNet (mốc)
    ("B02", "resnext50"),        # ResNeXt
    ("B03", "convnext_tiny"),    # ConvNeXt
    ("B04", "deit_small"),       # transformer (DeiT)
    ("B05", "swin_tiny"),        # transformer (Swin)
    ("B06", "efficientnet_b0"),  # mạng nhẹ
    ("B07", "mobilenetv3"),      # mạng nhẹ
]
BONUS_BACKBONES = [("B08", "dinov2_small", {"init": "frozen"})]  # RUBRIC điểm thưởng: linear probe

# T01..T14: mỗi dòng khác T00 đúng MỘT yếu tố (nguyên tắc N1). slot = nhóm loại trừ nhau khi kết hợp.
ABLATIONS = [
    # exp_id, trục, slot, desc, overrides
    ("T01", "A", "init", "scratch", {"init": "scratch"}),
    ("T02", "A", "init", "frozen", {"init": "frozen"}),
    ("T03", "B", "aug", "color", {"aug": "color"}),
    ("T04", "B", "aug", "trivialaug", {"aug": "trivial"}),
    ("T05", "B", "aug", "vflip", {"aug": "flipv"}),
    ("T06", "B", "mix", "mixup", {"mix": "mixup", "mix_alpha": 0.2}),
    ("T07", "B", "mix", "cutmix", {"mix": "cutmix", "mix_alpha": 1.0}),
    ("T08", "C", "loss", "label_smoothing", {"loss": "ls", "label_smoothing": 0.1}),
    ("T09", "C", "loss", "focal", {"loss": "focal", "focal_gamma": 2.0}),
    ("T10", "C", "loss", "weighted_ce", {"loss": "ce_weighted"}),
    ("T11", "D", "sampler", "balanced_sampler", {"sampler": "balanced"}),
    ("T12", "E", "lr", "same_lr", {"lr_head": 1e-4}),
    ("T13", "F", "ema", "ema", {"ema_decay": 0.998}),
    ("T14", "G", "res", "res256", {"img_size": 256}),
]
AXIS_NAMES = {"A": "Khởi tạo", "B": "Augmentation", "C": "Loss", "D": "Cân bằng mẫu",
              "E": "LR/optimizer", "F": "Chính quy hoá (EMA)", "G": "Độ phân giải"}
SEEDS_FINAL = (0, 1, 2)
DEFAULT_NOISE = 0.005      # ngưỡng nhiễu macro-F1 khi chưa đo được std (1 seed)
TTA_MIN_GAIN = 0.002       # chỉ dùng TTA cho chung kết nếu tăng macro-F1 val ít nhất mức này
REALTIME_P95_MS = 100.0


def _json(path: Path, obj=None):
    if obj is None:
        return json.loads(Path(path).read_text()) if Path(path).exists() else None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default))
    return obj


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _metrics(y, probs) -> dict:
    from eval import compute_metrics
    probs = np.asarray(probs, dtype=np.float64)
    m = compute_metrics(np.asarray(y), probs.argmax(1), probs)
    return {k: m[k] for k in ("top1", "macro_f1", "balanced_acc", "ece", "nll")} | {
        "f1_chinee_apple": float(m["f1"][0]), "f1_snake_weed": float(m["f1"][7])}


# --------------------------------------------------------------------------- #
# Lab: giữ đường dẫn, chạy job, lưu quyết định
# --------------------------------------------------------------------------- #
class Lab:
    def __init__(self, out_root, images_dir, labels_dir, cache_path=None, n_gpus: int | None = None,
                 num_workers: int | None = None, smoke: bool = False, base_overrides: dict | None = None):
        import torch
        self.out = Path(out_root)
        self.images_dir, self.labels_dir = str(images_dir), str(labels_dir)
        self.cache_path = str(cache_path) if cache_path else None
        self.n_gpus = torch.cuda.device_count() if n_gpus is None else n_gpus
        self.cpus = os.cpu_count() or 2
        self.num_workers = num_workers
        self.smoke = smoke
        self.base_overrides = base_overrides or {}
        for d in ("runs", "ckpt", "curves", "predictions", "inference", "figures", "eval_out", "logs"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.dec_path = self.out / "decisions.json"

    # ---- cấu hình ---------------------------------------------------------- #
    def cfg(self, **kw) -> Config:
        base = dict(images_dir=self.images_dir, labels_dir=self.labels_dir, cache_path=self.cache_path,
                    out_dir=str(self.out / "runs"), pred_dir=str(self.out / "predictions"),
                    ckpt_dir=str(self.out / "ckpt"), curves_dir=str(self.out / "curves"),
                    num_workers=self.num_workers or max(2, self.cpus // max(1, self.n_gpus)))
        if self.smoke:
            base.update(epochs=2, max_train_steps=3, eval_limit=96, batch_size=16)
        base.update(self.base_overrides)
        base.update(kw)
        return Config(**base)

    def summary(self, exp_id: str, seed: int = 0) -> dict | None:
        return _json(self.out / "runs" / exp_id / f"seed{seed}" / "summary.json")

    @property
    def decisions(self) -> dict:
        return _json(self.dec_path) or {}

    def _decide(self, key: str, value: dict) -> dict:
        d = self.decisions
        d[key] = value
        _json(self.dec_path, d)
        return value

    # ---- chạy job ----------------------------------------------------------- #
    def run_jobs(self, jobs: list[dict], name: str) -> list[dict]:
        """Chạy danh sách job (mỗi job = kwargs của Config). Bỏ qua job đã xong (summary.json)."""
        todo = [j for j in jobs if self.summary(j["exp_id"], j.get("seed", 0)) is None]
        print(f"[{name}] {len(jobs)} job, {len(jobs) - len(todo)} đã xong, còn {len(todo)}; GPU = {self.n_gpus}")
        if todo:
            if self.n_gpus >= 2 and len(todo) >= 2:
                self._run_parallel(todo, name)
            else:
                for j in todo:
                    TR.run(self.cfg(**j))
        missing = [f"{j['exp_id']}_seed{j.get('seed', 0)}" for j in jobs
                   if self.summary(j["exp_id"], j.get("seed", 0)) is None]
        if missing:
            raise RuntimeError(f"[{name}] các job lỗi/chưa xong: {missing}; xem {self.out / 'logs'}")
        return [self.summary(j["exp_id"], j.get("seed", 0)) for j in jobs]

    def _run_parallel(self, jobs: list[dict], name: str) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        qdir = self.out / "logs" / f"queue_{name}_{stamp}"
        qdir.mkdir(parents=True)
        cfgs = [dataclasses.asdict(self.cfg(**j)) for j in jobs]
        _json(qdir / "jobs.json", cfgs)
        procs, logs = [], []
        n = min(self.n_gpus, len(jobs))
        for g in range(n):
            log = qdir / f"worker{g}.log"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
            procs.append(subprocess.Popen([sys.executable, str(HERE / "experiments.py"), "worker",
                                           "--queue", str(qdir)], env=env, cwd=str(HERE),
                                          stdout=open(log, "w"), stderr=subprocess.STDOUT))
            logs.append(log)
        pos = [0] * n
        while True:
            alive = [p.poll() is None for p in procs]
            for g, log in enumerate(logs):
                text = log.read_text(errors="replace")
                new = text[pos[g]:]
                pos[g] = len(text)
                for line in new.splitlines():
                    if "] ep " in line or "XONG" in line or "LỖI" in line or "Traceback" in line or "Error" in line:
                        print(f"[gpu{g}] {line}", flush=True)
            if not any(alive):
                break
            time.sleep(20)
        for g, p in enumerate(procs):
            if p.returncode != 0:
                print(f"[gpu{g}] worker thoát mã {p.returncode}; log: {logs[g]}")

    # ======================================================================= #
    # Bước 1: backbone
    # ======================================================================= #
    def backbone_jobs(self, bonus: bool = True) -> list[dict]:
        jobs = [dict(exp_id=e, desc=b, backbone=b, seed=0) for e, b in BACKBONES]
        if bonus:
            jobs += [dict(exp_id=e, desc=f"{b}_linear_probe", backbone=b, seed=0, **kw)
                     for e, b, kw in BONUS_BACKBONES]
        order = ["swin_tiny", "convnext_tiny", "resnext50", "deit_small", "resnet50", "dinov2_small",
                 "efficientnet_b0", "mobilenetv3"]  # job nặng trước để chia đều 2 GPU
        return sorted(jobs, key=lambda j: order.index(j["backbone"]) if j["backbone"] in order else 99)

    def stage1_backbones(self, bonus: bool = True) -> pd.DataFrame:
        self.run_jobs(self.backbone_jobs(bonus), "stage1")
        return self.backbone_table()

    def backbone_table(self) -> pd.DataFrame:
        rows = []
        for e, b in BACKBONES + [(x[0], x[1]) for x in BONUS_BACKBONES]:
            s = self.summary(e)
            if s is None:
                continue
            lat = s.get("latency_b1_fp32") or {}
            rows.append({"exp_id": e, "backbone": b, "timm_name": s["timm_name"], "tag": s["tag"],
                         "init": s["init"], "params_M": s["params_M"], "gmacs": s["gmacs"],
                         "val_macro_f1": s["val_macro_f1"], "val_top1": s["val_top1"],
                         "val_f1_chinee_apple": s["val_f1_chinee_apple"], "val_f1_snake_weed": s["val_f1_snake_weed"],
                         "best_epoch": s["best_epoch"], "time_train_per_epoch_s": s["time_train_per_epoch_s"],
                         "latency_b1_p50_ms": lat.get("p50"), "latency_b1_p95_ms": lat.get("p95")})
        return pd.DataFrame(rows)

    def select_backbone(self, force: str | None = None) -> dict:
        """Luật (chỉ val, 1 seed): lấy các backbone có macro-F1 val >= tốt nhất - DEFAULT_NOISE
        (1 seed nên không phân biệt được trong khoảng này), rồi chọn cái có thời gian train/epoch nhỏ nhất
        (rẻ nhất cho ~18 lần chạy ở Bước 2), hoà thì lấy độ trễ p95 thấp hơn. B08 (đóng băng) không xét."""
        if "backbone" in self.decisions and not force:
            return self.decisions["backbone"]
        t = self.backbone_table()
        t = t[t["exp_id"].isin([e for e, _ in BACKBONES])]
        best_f1 = t["val_macro_f1"].max()
        cand = t[t["val_macro_f1"] >= best_f1 - DEFAULT_NOISE].sort_values(
            ["time_train_per_epoch_s", "latency_b1_p95_ms"])
        pick = force or cand.iloc[0]["backbone"]
        row = t[t["backbone"] == pick].iloc[0]
        return self._decide("backbone", {
            "backbone": pick, "from_exp": row["exp_id"], "forced": bool(force),
            "rule": f"macro-F1 val >= max - {DEFAULT_NOISE} (1 seed), rồi thời gian train/epoch nhỏ nhất",
            "best_f1": float(best_f1), "candidates": cand["backbone"].tolist(),
            "table": t.round(5).to_dict("records")})

    # ======================================================================= #
    # Bước 2: công thức huấn luyện
    # ======================================================================= #
    def _bk(self) -> str:
        return self.decisions["backbone"]["backbone"]

    def ablation_jobs(self) -> list[dict]:
        bk = self._bk()
        jobs = [dict(exp_id="T00", desc="baseline", backbone=bk, seed=s) for s in SEEDS_FINAL]
        jobs += [dict(exp_id=e, desc=d, backbone=bk, seed=0, **ov) for e, _, _, d, ov in ABLATIONS]
        return jobs

    def stage2_ablation(self) -> pd.DataFrame:
        self.run_jobs(self.ablation_jobs(), "stage2")
        return self.training_table()

    def t00_stats(self) -> dict:
        f1 = [self.summary("T00", s)["val_macro_f1"] for s in SEEDS_FINAL if self.summary("T00", s)]
        t1 = [self.summary("T00", s)["val_top1"] for s in SEEDS_FINAL if self.summary("T00", s)]
        std = float(np.std(f1, ddof=1)) if len(f1) > 1 else float("nan")
        return {"f1": f1, "mean": float(np.mean(f1)), "std": std, "top1_mean": float(np.mean(t1)),
                "top1_std": float(np.std(t1, ddof=1)) if len(t1) > 1 else float("nan"),
                "noise": std if np.isfinite(std) and std > 0 else DEFAULT_NOISE}

    def training_table(self) -> pd.DataFrame:
        st = self.t00_stats()
        rows = []
        for s in SEEDS_FINAL:
            x = self.summary("T00", s)
            if x:
                rows.append(self._trow("T00", "-", "-", "baseline (công thức nền)", x, st))
        for e, ax, slot, d, ov in ABLATIONS + [("T15", "A-G", "combo", "combo", None)]:
            x = self.summary(e)
            if x is None:
                continue
            if ov is None:
                ov = self.decisions.get("combo", {}).get("overrides", {})
            diff = ", ".join(f"{k}={v}" for k, v in ov.items())
            rows.append(self._trow(e, ax, slot, diff, x, st))
        return pd.DataFrame(rows)

    def _trow(self, e, ax, slot, diff, x, st):
        delta = x["val_macro_f1"] - st["mean"]
        return {"exp_id": e, "desc": x["desc"], "backbone": x["backbone"], "axis": ax, "slot": slot,
                "diff_vs_T00": diff, "seed": x["seed"], "val_macro_f1": x["val_macro_f1"], "val_top1": x["val_top1"],
                "delta_vs_T00_mean": delta if e != "T00" else x["val_macro_f1"] - st["mean"],
                "T00_std": st["std"], "beyond_noise": bool(abs(delta) > st["noise"]) if e != "T00" else None,
                "val_f1_chinee_apple": x["val_f1_chinee_apple"], "val_f1_snake_weed": x["val_f1_snake_weed"],
                "val_ece": x["val_ece"], "best_epoch": x["best_epoch"],
                "time_train_per_epoch_s": x["time_train_per_epoch_s"],
                "best_raw_val_macro_f1": x.get("best_raw_val_macro_f1"),
                "best_ema_val_macro_f1": x.get("best_ema_val_macro_f1")}

    def select_combo(self, force: dict | None = None) -> dict:
        """Kết hợp T15: với mỗi slot, lấy giá trị có Δ (so với mean T00) lớn nhất NẾU Δ > std(T00).
        Trọng số lớp (T10) và sampler cân bằng (T11) cùng xử lý mất cân bằng: nếu cả hai thắng thì giữ cái
        Δ lớn hơn. Nếu ít hơn 2 yếu tố vượt nhiễu, bổ sung các yếu tố có Δ > 0 lớn nhất cho đủ 2 (ghi rõ),
        để vẫn kiểm tra được hiệu ứng cộng dồn (GUIDE mục 3.1 bước 4)."""
        if "combo" in self.decisions and not force:
            return self.decisions["combo"]
        st = self.t00_stats()
        cands = []
        for e, ax, slot, d, ov in ABLATIONS:
            x = self.summary(e)
            if x is not None:
                cands.append({"exp_id": e, "slot": slot, "desc": d, "overrides": ov,
                              "delta": x["val_macro_f1"] - st["mean"]})
        best_per_slot = {}
        for c in sorted(cands, key=lambda c: -c["delta"]):
            best_per_slot.setdefault(c["slot"], c)
        ranked = sorted(best_per_slot.values(), key=lambda c: -c["delta"])
        chosen = [c for c in ranked if c["delta"] > st["noise"]]
        note = "các yếu tố có Δ > std(T00)"
        if len(chosen) < 2:
            extra = [c for c in ranked if c not in chosen and c["delta"] > 0][: 2 - len(chosen)]
            if extra:
                note += f"; bổ sung {[c['exp_id'] for c in extra]} (Δ > 0 nhưng chưa vượt nhiễu) cho đủ 2 yếu tố"
            chosen += extra
        if len(chosen) < 2:
            extra = [c for c in ranked if c not in chosen][: 2 - len(chosen)]
            note += f"; không đủ yếu tố có Δ > 0, lấy {[c['exp_id'] for c in extra]} có Δ cao nhất"
            chosen += extra
        ids = {c["exp_id"] for c in chosen}
        if {"T10", "T11"} <= ids:
            drop = min((c for c in chosen if c["exp_id"] in ("T10", "T11")), key=lambda c: c["delta"])
            chosen.remove(drop)
            note += f"; bỏ {drop['exp_id']} vì trùng vai trò cân bằng lớp với yếu tố còn lại"
        if any(c["slot"] == "init" and c["overrides"].get("init") != "finetune" for c in chosen):
            pass  # vẫn cho phép (nếu scratch/frozen thắng thì kết quả hợp lệ để phân tích)
        ov = {}
        for c in chosen:
            ov.update(c["overrides"])
        if force:
            ov, note = dict(force), "chỉ định thủ công"
        return self._decide("combo", {"from": [c["exp_id"] for c in chosen], "overrides": ov, "note": note,
                                      "noise_threshold": st["noise"], "T00": st,
                                      "candidates": sorted(cands, key=lambda c: -c["delta"])})

    def stage2_combo(self) -> dict:
        ov = self.decisions["combo"]["overrides"]
        self.run_jobs([dict(exp_id="T15", desc="combo", backbone=self._bk(), seed=0, **ov)], "stage2_combo")
        return self.summary("T15")

    def select_recipe(self, force: str | None = None) -> dict:
        """Công thức chung kết = cái có macro-F1 val cao nhất trong {T00 (mean 3 seed), các T01..T14 có Δ > std,
        T15}. Nếu không có gì vượt T00 thì giữ công thức nền."""
        if "recipe" in self.decisions and not force:
            return self.decisions["recipe"]
        st = self.t00_stats()
        cand = [{"exp_id": "T00", "val_macro_f1": st["mean"], "overrides": {}}]
        for e, _, _, _, ov in ABLATIONS:
            x = self.summary(e)
            if x and x["val_macro_f1"] - st["mean"] > st["noise"]:
                cand.append({"exp_id": e, "val_macro_f1": x["val_macro_f1"], "overrides": ov})
        x = self.summary("T15")
        if x:
            cand.append({"exp_id": "T15", "val_macro_f1": x["val_macro_f1"],
                         "overrides": self.decisions["combo"]["overrides"]})
        best = max(cand, key=lambda c: c["val_macro_f1"])
        if force:
            best = next(c for c in cand if c["exp_id"] == force)
        return self._decide("recipe", {"exp_id": best["exp_id"], "overrides": best["overrides"],
                                       "backbone": self._bk(), "forced": bool(force), "candidates": cand,
                                       "rule": "max macro-F1 val trong {T00 mean, Δ > std(T00), T15}"})

    # ======================================================================= #
    # Bước 3: suy luận
    # ======================================================================= #
    def _recipe_cfg(self, seed: int = 0) -> Config:
        r = self.decisions["recipe"]
        return TR.load_config(r["exp_id"], seed, str(self.out / "runs"))

    def stage3_inference(self) -> pd.DataFrame:
        """Mọi thí nghiệm suy luận trên VAL (model của công thức đã chọn, seed 0) + bảng độ trễ."""
        import torch

        import dataset as D
        import inference as INF
        from benchmark import latency_report, multi_model_latency, tta_latency
        from model import has_batchnorm

        out_csv = self.out / "inference" / "inference.csv"
        if out_csv.exists():
            print("stage3 đã chạy, đọc lại", out_csv)
            return pd.read_csv(out_csv)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dname = "cuda" if dev.type == "cuda" else "cpu"
        iters = 50 if self.smoke else 100
        cfg = self._recipe_cfg(0)
        model, mean, std = TR.load_trained(cfg, dev)
        _, val_df, _ = D.load_split(cfg.labels_dir, cfg.fold)
        if cfg.eval_limit:
            val_df = val_df.iloc[: cfg.eval_limit].reset_index(drop=True)
        S = cfg.img_size
        full = int(round(S / D.EVAL_CROP_PCT))  # 256 khi S = 224
        loader = TR.build_eval_loader(cfg, val_df, mean, std, img_size=full, crop_pct=1.0, batch_size=64)

        def center(x, s=S):
            h = x.shape[-1]
            o = (h - s) // 2
            return x[..., o:o + s, o:o + s]

        def res_view(r):
            def f(x):
                big = int(round(r / D.EVAL_CROP_PCT))
                return center(INF.views_multiscale(x, [big])[0], r)
            return f

        bank_fns = {"center": lambda x: center(x), "center_flip": lambda x: INF.view_hflip(center(x))}
        for i, v in enumerate(INF.views_multicrop(torch.zeros(1, 3, full, full), S)[:4]):
            bank_fns[f"corner{i}"] = (lambda i: lambda x: INF.views_multicrop(x, S)[i])(i)
            bank_fns[f"corner{i}_flip"] = (lambda i: lambda x: INF.view_hflip(INF.views_multicrop(x, S)[i]))(i)
        bank_fns["full"] = lambda x: x
        bank_fns[f"full{full + 32}"] = lambda x: INF.views_multiscale(x, [full + 32])[0]
        for r in (S + 32, S + 64, S + 96):
            bank_fns[f"res{r}"] = res_view(r)
        # kiểm tra view nào model chạy được (Swin không nhận mọi kích thước)
        notes = {}
        x0 = next(iter(loader))[0][:2].to(dev)
        for k in list(bank_fns):
            try:
                with torch.inference_mode():
                    model(bank_fns[k](x0))
            except Exception as e:  # noqa: BLE001
                notes[k] = f"không áp dụng: {type(e).__name__}: {str(e)[:120]}"
                bank_fns.pop(k)
        keys = list(bank_fns)
        names, y, outs = INF.predict_views(model, loader, dev, lambda x: [bank_fns[k](x) for k in keys])
        bank = dict(zip(keys, outs))
        np.savez_compressed(self.out / "inference" / "val_view_bank.npz", filenames=np.array(names), y=y, **bank)

        # độ trễ cơ sở
        lat0 = latency_report(model, 1, S, "fp32", dname, iters=iters, label="I00 1-view fp32")
        lat0_b32 = latency_report(model, 32, S, "fp32", dname, iters=max(50, iters // 2), label="I00 b32 fp32")
        rows, lat_rows = [], [lat0, lat0_b32]
        ckpt = f"{cfg.exp_id}_seed0 ({cfg.backbone})"

        def add(eid, method, probs, k, lat=None, space="-", model_desc=ckpt, note="", thr=None):
            m = _metrics(y, probs)
            lat = lat or {}
            rows.append({"exp_id": eid, "method": method, "model": model_desc, "K": k, "space": space,
                         "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_ece": m["ece"],
                         "val_nll": m["nll"], "val_f1_chinee_apple": m["f1_chinee_apple"],
                         "val_f1_snake_weed": m["f1_snake_weed"],
                         "lat_p50_ms": lat.get("p50"), "lat_p95_ms": lat.get("p95"), "lat_p99_ms": lat.get("p99"),
                         "throughput_img_s": thr if thr is not None else lat.get("images_per_s"),
                         "rel_cost_vs_I00": (lat["p50"] / lat0["p50"]) if lat.get("p50") else None,
                         "note": note})

        P = lambda ks, sp="prob": INF.aggregate_views([bank[k] for k in ks], sp)  # noqa: E731
        add("I00", "1 view (center-crop)", P(["center"]), 1, lat0, thr=lat0_b32["images_per_s"],
            note=f"Resize {full} -> CenterCrop {S}, fp32")
        # TTA lật ngang
        l_flip = tta_latency(model, 2, INF.views_hflip2, S, dtype="fp32", device=dname, iters=iters,
                             label="I01 hflip K=2 stacked")
        l_flip_seq = tta_latency(model, 2, INF.views_hflip2, S, dtype="fp32", device=dname, iters=iters,
                                 mode="sequential", label="I01 hflip K=2 sequential")
        lat_rows += [l_flip, l_flip_seq]
        add("I01", "TTA lật ngang", P(["center", "center_flip"]), 2, l_flip, "prob",
            note=f"stacked batch 2; tuần tự p50 = {l_flip_seq['p50']:.2f} ms")
        crops5 = [f"corner{i}" for i in range(4)] + ["center"]
        crops10 = crops5 + [f"{c}_flip" for c in crops5]
        l5 = tta_latency(model, 5, lambda x: INF.views_multicrop(x, S), S, input_size=full, dtype="fp32",
                         device=dname, iters=iters, label="I02a 5-crop")
        l10 = tta_latency(model, 10, lambda x: INF.views_multicrop(x, S, flip=True), S, input_size=full,
                          dtype="fp32", device=dname, iters=iters, label="I02b 10-crop")
        lat_rows += [l5, l10]
        add("I02a", "TTA 5 crop", P(crops5), 5, l5, "prob", note=f"4 góc + giữa, crop {S} từ ảnh {full}")
        add("I02b", "TTA 10 crop", P(crops10), 10, l10, "prob", note="5 crop + bản lật")
        ms = [k for k in ("center", "full", f"full{full + 32}") if k in bank]
        if len(ms) == 3:
            l_ms = tta_latency(model, 3, lambda x: [center(x), x, INF.views_multiscale(x, [full + 32])[0]], S,
                               input_size=full, dtype="fp32", device=dname, iters=iters, label="I02c multiscale")
            lat_rows.append(l_ms)
            add("I02c", "TTA nhiều tỉ lệ", P(ms), 3, l_ms, "prob",
                note=f"center-crop {S} + ảnh nguyên {full} + ảnh nguyên phóng {full + 32}")
        # gộp logit (I03)
        for eid, src, ks, k in (("I03a", "I01", ["center", "center_flip"], 2), ("I03b", "I02a", crops5, 5),
                                ("I03c", "I02b", crops10, 10), ("I03d", "I02c", ms, 3)):
            if all(c in bank for c in ks) and (src != "I02c" or len(ms) == 3):
                lat = next(r for r in rows if r["exp_id"] == src)
                add(eid, f"{src} gộp LOGIT", P(ks, "logit"), k,
                    {"p50": lat["lat_p50_ms"], "p95": lat["lat_p95_ms"], "p99": lat["lat_p99_ms"],
                     "images_per_s": lat["throughput_img_s"]}, "logit", note=f"cùng view với {src}, khác cách gộp")
        # dò độ phân giải (I04)
        for r in (S + 32, S + 64, S + 96):
            if f"res{r}" in bank:
                lr_ = latency_report(model, 1, r, "fp32", dname, iters=iters, label=f"I04 res{r}")
                lat_rows.append(lr_)
                add(f"I04_{r}", f"độ phân giải kiểm tra {r}", P([f"res{r}"]), 1, lr_,
                    note=f"Resize {int(round(r / D.EVAL_CROP_PCT))} -> CenterCrop {r} (train {S})")
            else:
                add(f"I04_{r}", f"độ phân giải kiểm tra {r}", P(["center"]), 1, None,
                    note=notes.get(f"res{r}", "không áp dụng"))
                rows[-1].update({k: None for k in ("val_macro_f1", "val_top1", "val_ece", "val_nll")})
        # ensemble (I05): xác suất val đã lưu của từng lần chạy
        def val_probs(e, s=0):
            z = np.load(self.out / "runs" / e / f"seed{s}" / "val_outputs.npz")
            assert list(z["filenames"]) == list(names), f"{e}: thứ tự file val khác"
            return _softmax(z["logits"])
        bt = self.backbone_table()
        bt = bt[bt["exp_id"].isin([e for e, _ in BACKBONES])].sort_values("val_macro_f1", ascending=False)
        top3 = bt["exp_id"].tolist()[:3]
        ens_models = []
        for e in top3:
            m_, _, _ = TR.load_trained(TR.load_config(e, 0, str(self.out / "runs")), dev)
            ens_models.append(m_)
        l_e3 = multi_model_latency(ens_models, S, "fp32", dname, iters=iters, label=f"I05a ensemble {top3}")
        lat_rows.append(l_e3)
        add("I05a", "Ensemble 3 backbone tốt nhất (B)", INF.ensemble_probs([val_probs(e) for e in top3]), 3, l_e3,
            model_desc="+".join(top3), note="trung bình xác suất; val logit lưu lúc train (AMP)")
        del ens_models
        allb = bt["exp_id"].tolist()
        add("I05b", f"Ensemble {len(allb)} backbone", INF.ensemble_probs([val_probs(e) for e in allb]), len(allb),
            model_desc="+".join(allb), note="độ trễ ≈ tổng độ trễ từng model (không đo riêng)")
        t00_models = [TR.load_trained(TR.load_config("T00", s, str(self.out / "runs")), dev)[0] for s in SEEDS_FINAL]
        l_es = multi_model_latency(t00_models, S, "fp32", dname, iters=iters, label="I05c ensemble T00 x3 seed")
        lat_rows.append(l_es)
        add("I05c", "Ensemble 3 seed (T00)", INF.ensemble_probs([val_probs("T00", s) for s in SEEDS_FINAL]), 3, l_es,
            model_desc="T00 seed0+1+2")
        # EMA (I06a) và soup (I06b)
        t13 = self.summary("T13")
        if t13 and t13.get("best_ema_val_macro_f1") is not None:
            rows.append({"exp_id": "I06a", "method": "Trọng số EMA (T13) vs trọng số thường cùng lần chạy",
                         "model": "T13_seed0", "K": 1, "space": "-", "val_macro_f1": t13["best_ema_val_macro_f1"],
                         "val_top1": None, "val_ece": None, "val_nll": None,
                         "lat_p50_ms": lat0["p50"], "lat_p95_ms": lat0["p95"], "lat_p99_ms": lat0["p99"],
                         "throughput_img_s": lat0_b32["images_per_s"], "rel_cost_vs_I00": 1.0,
                         "note": f"best raw val macro-F1 = {t13['best_raw_val_macro_f1']:.4f}; "
                                 f"best EMA = {t13['best_ema_val_macro_f1']:.4f}; không tốn thêm lúc suy luận"})
        soup = copy.deepcopy(t00_models[0])
        soup.load_state_dict(INF.model_soup([m.state_dict() for m in t00_models]))
        std_loader = TR.build_eval_loader(cfg, val_df, mean, std, img_size=S, batch_size=128)
        _, ys, ls = INF.predict_logits(soup, std_loader, dev)
        assert (ys == y).all()
        add("I06b", "Model soup đều 3 seed (T00)", _softmax(ls), 1, lat0, model_desc="soup(T00 seed0,1,2)",
            note="trung bình trọng số; head mỗi seed khởi tạo khác nhau", thr=lat0_b32["images_per_s"])
        del t00_models, soup
        # Temperature scaling (I07): khớp trên val; báo cả ECE in-sample và ECE chéo 2 nửa val
        def ts_eval(logits):
            T = INF.fit_temperature(logits, y)
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(y))
            a, b = idx[: len(y) // 2], idx[len(y) // 2:]
            Ta, Tb = INF.fit_temperature(logits[a], y[a]), INF.fit_temperature(logits[b], y[b])
            cv = np.empty((len(y), logits.shape[1]))
            cv[b] = INF.apply_temperature(logits[b], Ta)
            cv[a] = INF.apply_temperature(logits[a], Tb)
            return T, INF.apply_temperature(logits, T), _metrics(y, cv)["ece"]
        T0, p_ts, ece_cv = ts_eval(bank["center"])
        add("I07a", "Temperature scaling trên I00", p_ts, 1, lat0, note=(
            f"T = {T0:.3f}; ECE trước {_metrics(y, P(['center']))['ece']:.4f}; ECE chéo 2 nửa val {ece_cv:.4f}"),
            thr=lat0_b32["images_per_s"])
        # FP16 / AMP / gộp BN (I08)
        for eid, dt in (("I08a", "amp"), ("I08b", "fp16")):
            mm = copy.deepcopy(model)
            if dt == "fp16":
                mm = mm.half()
            _, _, l_ = INF.predict_logits(mm, std_loader, dev, amp=dt == "amp", half=dt == "fp16")
            lt = latency_report(model, 1, S, dt, dname, iters=iters, label=f"{eid} {dt} b1")
            lt32 = latency_report(model, 32, S, dt, dname, iters=max(50, iters // 2), label=f"{eid} {dt} b32")
            lat_rows += [lt, lt32]
            flips = int((l_.argmax(1) != bank["center"].argmax(1)).sum())
            add(eid, f"1 view {dt.upper()}", _softmax(l_), 1, lt, thr=lt32["images_per_s"],
                note=f"{flips} ảnh val đổi nhãn so với FP32")
            del mm
        bn_model, bn_desc = (model, ckpt) if has_batchnorm(model) else (None, None)
        if bn_model is None:  # ConvNeXt/ViT/Swin không có BN: minh hoạ trên B01 (ResNet-50)
            try:
                bcfg = TR.load_config("B01", 0, str(self.out / "runs"))
                bn_model, bmean, bstd = TR.load_trained(bcfg, dev)
                bn_desc = "B01_seed0 (resnet50; model chính không có BN)"
            except FileNotFoundError:
                bn_model = None
        if bn_model is not None:
            fused = INF.fuse_conv_bn(bn_model, img_size=S)
            if bn_model is model:
                ref_logits = bank["center"]
                _, _, lf = INF.predict_logits(fused, std_loader, dev)
            else:
                bl = TR.build_eval_loader(bcfg, val_df, bmean, bstd, img_size=S, batch_size=128)
                _, _, ref_logits = INF.predict_logits(bn_model, bl, dev)
                _, _, lf = INF.predict_logits(fused, bl, dev)
            lu = latency_report(bn_model, 1, S, "fp32", dname, iters=iters, label="I08c chưa gộp BN fp32 b1")
            lfz = latency_report(fused, 1, S, "fp32", dname, iters=iters, bn_fused=True, label="I08c gộp BN fp32 b1")
            lfz16 = latency_report(fused, 1, S, "fp16", dname, iters=iters, bn_fused=True, label="I08c gộp BN fp16 b1")
            lat_rows += [lu, lfz, lfz16]
            add("I08c", "Gộp BN vào conv (FP32)", _softmax(lf), 1, lfz, model_desc=bn_desc, note=(
                f"{fused.fuse_info['n_fused']} cặp conv-BN; sai số logit lớn nhất {np.abs(lf - ref_logits).max():.2e}; "
                f"chưa gộp p50 {lu['p50']:.2f} ms; gộp + FP16 p50 {lfz16['p50']:.2f} ms"))
            rows[-1]["rel_cost_vs_I00"] = lfz["p50"] / lu["p50"]
            del fused
        # độ trễ đầy đủ cho mọi backbone (sheet Latency)
        for e, _ in BACKBONES + [(b[0], b[1]) for b in BONUS_BACKBONES]:
            if self.summary(e) is None:
                continue
            bc = TR.load_config(e, 0, str(self.out / "runs"))
            bm, _, _ = TR.load_trained(bc, dev)
            for dt in ("fp32", "amp", "fp16"):
                for bs in (1, 32):
                    lat_rows.append(latency_report(bm, bs, bc.img_size, dt, dname,
                                                   iters=iters if bs == 1 else max(50, iters // 2),
                                                   label=f"{e} {bc.backbone}"))
            del bm
            torch.cuda.empty_cache()
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        pd.DataFrame(lat_rows).to_csv(self.out / "inference" / "latency.csv", index=False)
        _json(self.out / "inference" / "notes.json", {"view_notes": notes, "recipe_cfg": dataclasses.asdict(cfg),
                                                      "T_I00": T0})
        # TTA đổi nhãn: đúng -> sai và sai -> đúng (slide trang 66)
        base_pred = bank["center"].argmax(1)
        flips = {}
        for eid, ks in (("I01", ["center", "center_flip"]), ("I02b", crops10)):
            tp = P(ks).argmax(1)
            flips[eid] = {"right_to_wrong": int(((base_pred == y) & (tp != y)).sum()),
                          "wrong_to_right": int(((base_pred != y) & (tp == y)).sum()),
                          "changed": int((base_pred != tp).sum())}
        _json(self.out / "inference" / "tta_flips.json", flips)
        del model
        torch.cuda.empty_cache()
        return df

    def select_inference(self, force: str | None = None) -> dict:
        """Suy luận ngoại tuyến cho F01: trong {I00, I01, I02a-c, I03a-d, I04_*} lấy macro-F1 val cao nhất;
        chỉ dùng nếu hơn I00 >= TTA_MIN_GAIN, nếu không giữ 1 view. Ensemble/soup/EMA không xét vì chung kết
        cần một dự đoán cho mỗi seed. Luôn áp dụng temperature scaling (T khớp trên val của từng seed).
        Cấu hình thời gian thực (F02) = cùng model, 1 view + TS, FP32."""
        if "inference" in self.decisions and not force:
            return self.decisions["inference"]
        df = pd.read_csv(self.out / "inference" / "inference.csv")
        ok = df[df["exp_id"].str.match(r"^I0[0-4]") & df["val_macro_f1"].notna()]
        i00 = float(ok.loc[ok["exp_id"] == "I00", "val_macro_f1"].iloc[0])
        best = ok.sort_values(["val_macro_f1", "K"], ascending=[False, True]).iloc[0]
        pick = best["exp_id"] if best["val_macro_f1"] - i00 >= TTA_MIN_GAIN else "I00"
        if force:
            pick = force
        row = df[df["exp_id"] == pick].iloc[0]
        return self._decide("inference", {
            "exp_id": pick, "method": row["method"], "K": int(row["K"]), "space": row["space"],
            "val_macro_f1": float(row["val_macro_f1"]), "I00_val_macro_f1": i00,
            "best_candidate": best["exp_id"], "best_candidate_f1": float(best["val_macro_f1"]),
            "min_gain": TTA_MIN_GAIN, "forced": bool(force),
            "temperature_scaling": True, "realtime": "1 view + TS, FP32"})

    # ======================================================================= #
    # Bước 4: chung kết
    # ======================================================================= #
    def _views_for(self, eid: str, S: int, full: int):
        """Trả về (views_fn trên ảnh nguyên `full`, space) cho mã suy luận eid."""
        import inference as INF

        def center(x, s=S):
            o = (x.shape[-1] - s) // 2
            return x[..., o:o + s, o:o + s]
        if eid == "I00":
            return (lambda x: [center(x)]), "prob"
        if eid in ("I01", "I03a"):
            return (lambda x: [center(x), INF.view_hflip(center(x))]), ("logit" if eid == "I03a" else "prob")
        if eid in ("I02a", "I03b"):
            return (lambda x: INF.views_multicrop(x, S)), ("logit" if eid == "I03b" else "prob")
        if eid in ("I02b", "I03c"):
            return (lambda x: INF.views_multicrop(x, S, flip=True)), ("logit" if eid == "I03c" else "prob")
        if eid in ("I02c", "I03d"):
            return (lambda x: [center(x), x, INF.views_multiscale(x, [full + 32])[0]]), \
                ("logit" if eid == "I03d" else "prob")
        if eid.startswith("I04_"):
            import dataset as D
            r = int(eid.split("_")[1])
            big = int(round(r / D.EVAL_CROP_PCT))
            return (lambda x: [center(INF.views_multiscale(x, [big])[0], r)]), "prob"
        raise ValueError(f"không hỗ trợ suy luận {eid} cho chung kết")

    def final_jobs(self) -> list[dict]:
        r = self.decisions["recipe"]
        return [dict(exp_id="F01", desc="final", backbone=r["backbone"], seed=s, **r["overrides"])
                for s in SEEDS_FINAL]

    def final_predict(self, exp_id: str, seed: int, inf_id: str, with_ts: bool, out_ids: dict) -> dict:
        """Tính dự đoán val và test của một model đã train (một lượt duy nhất qua test) rồi ghi file.

        out_ids: {"main": "F01", "uncal": "F01_uncal", "realtime": "F02"} (khoá nào None thì không ghi).
        Không tính chỉ số test ở đây; eval.py làm việc đó sau khi mọi file đã ghi xong.
        """
        import torch

        import dataset as D
        import inference as INF
        from eval import save_predictions

        pdir = self.out / "predictions"
        main_test = pdir / f"{out_ids['main']}_seed{seed}_test.csv"
        if main_test.exists():
            print(f"{main_test.name} đã có, KHÔNG chạy test lại")
            return _json(self.out / "runs" / exp_id / f"seed{seed}" / "final_predict.json") or {}
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = TR.load_config(exp_id, seed, str(self.out / "runs"))
        model, mean, std = TR.load_trained(cfg, dev)
        S = cfg.img_size
        full = int(round(S / D.EVAL_CROP_PCT))
        vf, space = self._views_for(inf_id, S, full)
        _, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
        if cfg.eval_limit:
            val_df = val_df.iloc[: cfg.eval_limit].reset_index(drop=True)
        res = {"exp_id": exp_id, "seed": seed, "inference": inf_id, "space": space}
        logits_final = {}
        for split, df in (("val", val_df), ("test", test_df)):
            loader = TR.build_eval_loader(cfg, df, mean, std, img_size=full, crop_pct=1.0, batch_size=64)
            names, y, views = INF.predict_views(model, loader, dev, lambda x: [*vf(x)] + [vf_center(x, S)])
            main_views, center_logits = views[:-1], views[-1]
            probs = INF.aggregate_views(main_views, space)
            z = INF.probs_to_logits(probs) if space == "prob" and len(main_views) > 1 else (
                np.mean(main_views, 0) if len(main_views) > 1 else main_views[0])
            logits_final[split] = (names, y, z, center_logits)
            np.savez_compressed(self.out / "runs" / exp_id / f"seed{seed}" / f"final_{split}_views.npz",
                                filenames=np.array(names), y=y, views=np.stack(main_views), center=center_logits)
        T = INF.fit_temperature(logits_final["val"][2], logits_final["val"][1]) if with_ts else 1.0
        Tc = INF.fit_temperature(logits_final["val"][3], logits_final["val"][1]) if with_ts else 1.0
        res.update({"T": T, "T_realtime": Tc})
        for split, (names, y, z, zc) in logits_final.items():
            save_predictions(pdir / f"{out_ids['main']}_seed{seed}_{split}.csv", names, y, INF.apply_temperature(z, T))
            if out_ids.get("uncal"):
                save_predictions(pdir / f"{out_ids['uncal']}_seed{seed}_{split}.csv", names, y, _softmax(z))
            if out_ids.get("realtime"):
                save_predictions(pdir / f"{out_ids['realtime']}_seed{seed}_{split}.csv", names, y,
                                 INF.apply_temperature(zc, Tc))
        _json(self.out / "runs" / exp_id / f"seed{seed}" / "final_predict.json", res)
        del model
        torch.cuda.empty_cache()
        return res

    def stage4_final(self) -> dict:
        """F01 x 3 seed (train), rồi dự đoán test MỘT lần cho F01 (+F02 thời gian thực) và mốc T00 (+I00)."""
        import torch
        from benchmark import latency_report, tta_latency

        self.run_jobs(self.final_jobs(), "stage4")
        inf = self.decisions["inference"]
        realtime = "F02" if inf["exp_id"] != "I00" else None
        for s in SEEDS_FINAL:
            self.final_predict("F01", s, inf["exp_id"], True,
                               {"main": "F01", "uncal": "F01_uncal", "realtime": realtime})
            self.final_predict("T00", s, "I00", False, {"main": "T00"})
        # độ trễ của chính model chung kết (seed 0)
        lat_path = self.out / "inference" / "latency_final.json"
        if not lat_path.exists():
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dname = "cuda" if dev.type == "cuda" else "cpu"
            cfg = TR.load_config("F01", 0, str(self.out / "runs"))
            model, _, _ = TR.load_trained(cfg, dev)
            import dataset as D
            S = cfg.img_size
            full = int(round(S / D.EVAL_CROP_PCT))
            lat = {"realtime_1view_fp32": latency_report(model, 1, S, "fp32", dname, label="F 1-view fp32 b1"),
                   "realtime_1view_fp16": latency_report(model, 1, S, "fp16", dname, label="F 1-view fp16 b1"),
                   "b32_fp32": latency_report(model, 32, S, "fp32", dname, iters=50, label="F 1-view fp32 b32")}
            if inf["exp_id"] != "I00":
                vf, _ = self._views_for(inf["exp_id"], S, full)
                lat["offline_tta_fp32"] = tta_latency(model, inf["K"], vf, S, input_size=full, dtype="fp32",
                                                      device=dname, label=f"F01 {inf['exp_id']} fp32 b1")
            _json(lat_path, lat)
            del model
        return self.run_eval()

    def run_eval(self) -> dict:
        """Gọi eval.py gốc (score cho từng nhóm, grade cho chung kết vs mốc). In và lưu kết quả."""
        lab = Path(self.labels_dir)
        evalpy = str(Path(TR.EVAL_DIR) / "eval.py")
        pdir, edir = self.out / "predictions", self.out / "eval_out"
        lat = _json(self.out / "inference" / "latency_final.json") or {}
        outs = {}
        groups = ["F01", "F01_uncal", "T00"] + (["F02"] if list(pdir.glob("F02_seed*_test.csv")) else [])
        for tag in groups:
            cmd = [sys.executable, evalpy, "score", "--pred", str(pdir / f"{tag}_seed*_test.csv"),
                   "--test-csv", str(lab / "test_subset0.csv"), "--labels", str(lab / "labels.csv"),
                   "--tag", tag, "--out", str(edir)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            print(r.stdout, r.stderr)
            outs[tag] = r.stdout
        p95 = (lat.get("realtime_1view_fp32") or {}).get("p95")
        rt_tag = "F02" if "F02" in groups else "F01"
        cmd = [sys.executable, evalpy, "grade", "--final", str(pdir / "F01_seed*_test.csv"),
               "--baseline", str(pdir / "T00_seed*_test.csv"), "--uncal", str(pdir / "F01_uncal_seed*_test.csv"),
               "--final-val", str(pdir / "F01_seed*_val.csv"), "--val-csv", str(lab / "val_subset0.csv"),
               "--test-csv", str(lab / "test_subset0.csv"), "--labels", str(lab / "labels.csv"), "--out", str(edir)]
        if p95 is not None:
            cmd += ["--latency-p95-ms", f"{p95:.3f}", "--latency-method", "proper"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout, r.stderr)
        outs["grade"] = r.stdout
        (edir / "eval_stdout.md").write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in outs.items()))
        return {"realtime_tag": rt_tag, "realtime_p95_ms": p95}

    # ======================================================================= #
    # Bước 5: phần thưởng và phân tích lỗi
    # ======================================================================= #
    def stage5_bonus(self) -> None:
        import analysis as AN
        AN.robustness(self)
        AN.error_analysis(self)


def vf_center(x, s):
    o = (x.shape[-1] - s) // 2
    return x[..., o:o + s, o:o + s]


# --------------------------------------------------------------------------- #
# Worker (tiến trình con, một GPU)
# --------------------------------------------------------------------------- #
def worker(queue: Path) -> int:
    jobs = json.loads((queue / "jobs.json").read_text())
    claims = queue / "claims"
    claims.mkdir(exist_ok=True)
    failed = 0
    for j in jobs:
        key = f"{j['exp_id']}_seed{j['seed']}"
        try:
            fd = os.open(claims / key, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue
        try:
            TR.run(Config(**j))
        except Exception:  # noqa: BLE001
            import traceback
            failed += 1
            print(f"LỖI job {key}:\n{traceback.format_exc()}", flush=True)
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--queue", required=True)
    args = ap.parse_args()
    if args.cmd == "worker":
        sys.exit(worker(Path(args.queue)))


if __name__ == "__main__":
    main()
