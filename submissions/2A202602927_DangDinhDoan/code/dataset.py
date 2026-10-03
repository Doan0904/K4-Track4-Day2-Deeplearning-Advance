"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Hoàn thiện từ bộ khung starter/dataset.py. Quy tắc chia dữ liệu (S1-S6) ở README.md, mục 2.1.

Giao diện (giữ nguyên như bộ khung):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Thêm so với bộ khung:
    build_cache(...)  : giải mã toàn bộ JPEG một lần vào file .npy uint8 (N, 3, 256, 256) để
                        DataLoader không phải giải mã lại mỗi epoch (Kaggle chỉ có 4 vCPU).
    eval_transform    : transform đánh giá có tham số crop_pct (dùng cho dò độ phân giải, I04).

Quyết định tiền xử lý (ghi vào báo cáo):
    - Ảnh gốc 256x256. Đánh giá: Resize(round(img_size / 0.875)) rồi CenterCrop(img_size).
      Với img_size = 224, Resize(256) không đổi gì, tức là center-crop 224 từ ảnh 256.
    - Mọi transform chạy trên tensor uint8 (torchvision.transforms.v2), chuẩn hoá ở bước cuối.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)

TOTAL_IMAGES = 17509
EVAL_CROP_PCT = 0.875   # 224 / 256
AUG_CHOICES = ("basic", "color", "trivial", "randaug", "flipv")


# --------------------------------------------------------------------------- #
# Đọc và kiểm tra split
# --------------------------------------------------------------------------- #
def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Trả về ba DataFrame nguyên bản (cột Filename, Label). KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    out = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
        if not {"Filename", "Label"} <= set(df.columns):
            raise ValueError(f"{split}_subset{fold}.csv thiếu cột Filename/Label: {list(df.columns)}")
        df["Label"] = df["Label"].astype(int)
        out.append(df)
    return tuple(out)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, labels_csv: str | Path | None = None, verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. giao của từng cặp tập theo Filename phải RỖNG
    3. hợp ba tập phải bằng đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir`
    5. (thêm) không có Filename trùng trong một tập; nhãn trong subset so với labels.csv (chỉ ghi lại
       chỗ lệch, KHÔNG sửa vì quy tắc S1; chấm điểm theo nhãn của subset)
    Vi phạm 1-4 thì raise AssertionError để dừng ngay.
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    names = {k: set(v["Filename"]) for k, v in splits.items()}

    for k, v in splits.items():
        assert not v["Filename"].duplicated().any(), f"{k}: có Filename bị trùng"
        assert v["Label"].between(0, NUM_CLASSES - 1).all(), f"{k}: nhãn ngoài 0..8"

    n = {k: int(len(v)) for k, v in splits.items()}
    total = sum(n.values())
    frac = {k: n[k] / total for k in n}

    overlap = {
        "train&val": len(names["train"] & names["val"]),
        "train&test": len(names["train"] & names["test"]),
        "val&test": len(names["val"] & names["test"]),
    }
    union = len(names["train"] | names["val"] | names["test"])
    assert all(v == 0 for v in overlap.values()), f"giao giữa các tập khác rỗng: {overlap}"
    assert union == TOTAL_IMAGES, f"hợp ba tập = {union}, kỳ vọng {TOTAL_IMAGES}"
    for k, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        assert abs(frac[k] - target) <= 0.01, f"tỉ lệ {k} = {frac[k]:.4f} lệch > 1 điểm % so với {target}"

    images_dir = Path(images_dir)
    on_disk = {p.name for p in images_dir.glob("*.jpg")}
    missing = sorted((names["train"] | names["val"] | names["test"]) - on_disk)
    assert not missing, f"{len(missing)} file trong CSV không có trong {images_dir}, ví dụ {missing[:3]}"

    # Không assert: S1 cấm sửa CSV, nên chỉ ghi lại để báo cáo (fold 0 có 1 ảnh train như vậy).
    label_mismatch = None
    if labels_csv is not None and Path(labels_csv).exists():
        ref = pd.read_csv(labels_csv)
        ref_map = dict(zip(ref["Filename"], ref["Label"].astype(int)))
        label_mismatch = [{"split": k, "Filename": f, "subset_label": int(l), "labels_csv_label": int(ref_map[f])}
                          for k, df in splits.items() for f, l in zip(df["Filename"], df["Label"])
                          if int(ref_map[f]) != int(l)]

    per_class = {k: {CLASS_NAMES[c]: int((v["Label"] == c).sum()) for c in range(NUM_CLASSES)}
                 for k, v in splits.items()}
    per_class["all"] = {c: sum(per_class[s][c] for s in splits) for c in CLASS_NAMES}

    report = {
        "n": n, "total": total, "fraction": {k: round(v, 4) for k, v in frac.items()},
        "overlap": overlap, "union": union, "missing_files": len(missing),
        "images_on_disk": len(on_disk), "label_mismatch_vs_labels_csv": label_mismatch,
        "per_class": per_class,
    }
    if verbose:
        print(f"Số ảnh: {n} (tổng {total}); tỉ lệ {report['fraction']}")
        print(f"Giao: {overlap}; hợp = {union} (kỳ vọng {TOTAL_IMAGES}); thiếu file: {len(missing)}")
        if label_mismatch:
            print(f"CẢNH BÁO: {len(label_mismatch)} ảnh có nhãn subset khác labels.csv (giữ nguyên subset): "
                  f"{label_mismatch}")
        print(pd.DataFrame(per_class).to_string())
    return report


# --------------------------------------------------------------------------- #
# Cache ảnh đã giải mã
# --------------------------------------------------------------------------- #
def _decode(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (256, 256):  # DeepWeeds đều 256x256; phòng hờ thì resize và báo
            im = im.resize((256, 256), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1).copy()


def build_cache(filenames, images_dir: str | Path, cache_path: str | Path, workers: int = 4) -> Path:
    """Giải mã mọi ảnh vào một file .npy uint8 (N, 3, 256, 256) + file .json chứa thứ tự tên file.

    Chỉ là cache tốc độ (ảnh giống hệt ảnh gốc), không thay đổi dữ liệu. ~3,4 GB cho 17.509 ảnh.
    """
    from multiprocessing import Pool

    cache_path = Path(cache_path)
    names_path = cache_path.with_suffix(".json")
    filenames = sorted(set(filenames))
    if cache_path.exists() and names_path.exists() and json.loads(names_path.read_text()) == filenames:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(str(cache_path) + ".tmp", mode="w+", dtype=np.uint8,
                                    shape=(len(filenames), 3, 256, 256))
    paths = [str(Path(images_dir) / f) for f in filenames]
    with Pool(workers) as pool:
        for i, img in enumerate(pool.imap(_decode, paths, chunksize=64)):
            arr[i] = img
    arr.flush()
    del arr
    os.replace(str(cache_path) + ".tmp", cache_path)
    names_path.write_text(json.dumps(filenames))
    return cache_path


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def eval_transform(img_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                   crop_pct: float = EVAL_CROP_PCT):
    """Resize(round(img_size / crop_pct)) -> CenterCrop(img_size) -> float -> Normalize.

    crop_pct = 1.0 cho ảnh nguyên khung (dùng khi TTA nhiều crop tự cắt trên ảnh 256).
    """
    import torch
    from torchvision.transforms import v2

    resize = int(round(img_size / crop_pct))
    ops = []
    if resize != 256:
        ops.append(v2.Resize(resize, antialias=True))
    if img_size != resize:
        ops.append(v2.CenterCrop(img_size))
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    return v2.Compose(ops)


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Tạo transform (torchvision.transforms.v2, đầu vào là tensor uint8 CxHxW).

    Train, theo `aug` (trục B của GUIDE.md mục 3):
      - "basic"   : RandomResizedCrop(img_size) + lật ngang                     (công thức nền)
      - "color"   : basic + ColorJitter(0.3, 0.3, 0.3, 0.05)
      - "trivial" : basic + TrivialAugmentWide
      - "randaug" : basic + RandAugment(2, 9)
      - "flipv"   : basic + lật dọc (ảnh chụp cây từ trên xuống nên lật dọc có thể hợp lệ;
                    thí nghiệm để kiểm chứng)
    Mixup/CutMix trộn theo batch nên nằm ở losses.py.

    Val/test: center-crop img_size từ ảnh 256 (xem eval_transform). Không augmentation ngẫu nhiên.
    """
    import torch
    from torchvision.transforms import v2

    if not train:
        return eval_transform(img_size, mean, std)
    if aug not in AUG_CHOICES:
        raise ValueError(f"aug={aug!r} không hợp lệ, chọn một trong {AUG_CHOICES}")
    ops = [v2.RandomResizedCrop(img_size, antialias=True), v2.RandomHorizontalFlip()]
    if aug == "color":
        ops.append(v2.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(v2.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(v2.RandAugment(num_ops=2, magnitude=9))
    elif aug == "flipv":
        ops.append(v2.RandomVerticalFlip())
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    return v2.Compose(ops)


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Tensor (N, 3, H, W) đã chuẩn hoá -> ảnh float [0, 1] để vẽ."""
    import torch
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x * s + m).clamp(0, 1)


# --------------------------------------------------------------------------- #
# Dataset và DataLoader
# --------------------------------------------------------------------------- #
try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # cho phép import module khi chưa có torch (chỉ dùng hàm pandas)
    _TorchDataset = object


class DeepWeedsDataset(_TorchDataset):
    """Dataset đọc ảnh theo DataFrame (Filename, Label).

    __getitem__(i) -> (ảnh đã transform, nhãn int, tên file str).
    Nếu có `cache_path` (file .npy của build_cache) thì đọc ảnh uint8 từ memmap, nếu không thì
    giải mã JPEG bằng PIL. Hai đường cho cùng một tensor uint8.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None,
                 cache_path: str | Path | None = None):
        self.filenames = df["Filename"].astype(str).tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.cache_path = str(cache_path) if cache_path else None
        self._arr = None
        self._index = None
        if self.cache_path:
            order = json.loads(Path(self.cache_path).with_suffix(".json").read_text())
            pos = {f: i for i, f in enumerate(order)}
            self._index = [pos[f] for f in self.filenames]

    def __len__(self) -> int:
        return len(self.filenames)

    def load_uint8(self, i: int):
        import torch
        if self.cache_path:
            if self._arr is None:  # mở memmap trong từng worker
                self._arr = np.load(self.cache_path, mmap_mode="r")
            return torch.from_numpy(np.array(self._arr[self._index[i]]))
        return torch.from_numpy(_decode(str(self.images_dir / self.filenames[i])))

    def __getitem__(self, i: int):
        img = self.load_uint8(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], self.filenames[i]


def seed_worker(worker_id: int) -> None:
    """Seed numpy/random trong worker từ seed torch của worker (tái lập augmentation)."""
    import torch
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                cache_path: str | Path | None = None, seed: int = 0):
    """Tạo DataLoader.

    - train=True: shuffle (hoặc sampler="balanced": WeightedRandomSampler, trọng số 1/n_lớp,
      lấy có hoàn lại, mỗi epoch đúng len(df) mẫu); drop_last=True để BatchNorm ổn định.
    - train=False: không shuffle, giữ đúng thứ tự df (để ghép logit với Filename).
    - generator và worker_init_fn được seed để thứ tự batch/augmentation tái lập theo `seed`.
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ds = DeepWeedsDataset(df, images_dir, transform, cache_path=cache_path)
    g = torch.Generator()
    g.manual_seed(seed)
    smp = None
    shuffle = train
    if train and sampler == "balanced":
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES).astype(np.float64)
        w = 1.0 / counts[np.asarray(ds.labels)]
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(ds),
                                    replacement=True, generator=g)
        shuffle = False
    elif sampler not in (None, "none", "balanced"):
        raise ValueError(f"sampler={sampler!r} không hợp lệ")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=smp, drop_last=train,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=seed_worker, generator=g,
    )
