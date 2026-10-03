# Lab Day 2 — DeepWeeds: backbone, công thức huấn luyện, suy luận

**Sinh viên:** Đặng Đình Đoàn — MSSV 2A202602927

> **Trạng thái:** code và notebook đã xong; **chưa có kết quả**. Toàn bộ huấn luyện chạy trên Kaggle.
> Sau khi chạy, `results.xlsx`, `report.md`, `curves/`, `predictions/` sẽ được thêm vào thư mục này
> (lấy từ `submission_export.zip` mà notebook tạo ra).

## Notebook chạy lại được

- Kaggle: _(điền link notebook Kaggle sau khi chạy)_
- File: [`code/lab_day2.ipynb`](code/lab_day2.ipynb), tự `git clone` repo này, tải dữ liệu, chạy Bước 0 → 5.

## Cách chạy

1. Kaggle → *New Notebook* → *File → Import Notebook* → dán link GitHub của `code/lab_day2.ipynb`.
2. *Settings*: **GPU T4 x2** (hoặc P100), **Internet On**. (Tuỳ chọn) *Add Input* một dataset chứa `images.zip`
   của DeepWeeds; nếu không có, notebook tự tải từ Zenodo và kiểm tra MD5 `b7b30f96d466fba86016aa5a26606e0f`.
3. **Chạy thử**: đặt `SMOKE = True` ở ô đầu tiên, *Run All* (khoảng 15–25 phút). Chế độ này chạy mọi bước với
   2 epoch × 3 bước train và 96 ảnh val, chỉ để bắt lỗi; số liệu KHÔNG dùng để báo cáo.
4. **Chạy thật**: `SMOKE = False` → *Save Version* → *Save & Run All (Commit)*. Ước lượng 3–5 giờ trên T4 x2
   (tự đo epoch đầu trong log rồi tính lại).
5. Tải output của version: `submission_export.zip` (không có checkpoint) và notebook đã chạy. Giải nén vào thư mục này.

Nếu phiên bị ngắt: tạo version mới, thêm output version cũ làm Input, đặt `RESUME_FROM = "/kaggle/input/<...>/outputs"`.
Lần chạy đã xong được bỏ qua; lần chạy dở chạy tiếp từ checkpoint epoch cuối; file dự đoán test đã có thì không chạy lại.

Chạy một thí nghiệm lẻ từ dòng lệnh:

```bash
cd code
python train.py --set exp_id=B01 desc=resnet50 backbone=resnet50 seed=0 \
    images_dir=../data/images labels_dir=../data/labels
python -m unittest test_code -v      # test tự viết, chạy trên CPU, không cần dữ liệu
```

## Thứ tự thí nghiệm và seed

| Bước | exp_id | Nội dung | Seed |
|---|---|---|---|
| 0 | — | Kiểm tra split S1–S6, EDA, loss ban đầu ≈ ln 9, overfit 16 ảnh, ảnh sau augmentation | 0 |
| 1 | B01–B07 | ResNet-50, ResNeXt-50, ConvNeXt-T, DeiT-S, Swin-T, EfficientNet-B0, MobileNetV3-L; cùng công thức nền | 0 |
| 1 (thưởng) | B08 | DINOv2 ViT-S/14 đóng băng + linear probe | 0 |
| 2 | T00 | Công thức nền trên backbone đã chọn (đo nhiễu) | 0, 1, 2 |
| 2 | T01–T14 | Mỗi lần khác T00 đúng 1 yếu tố: khởi tạo (scratch, frozen), augmentation (color, TrivialAugment, lật dọc, Mixup, CutMix), loss (label smoothing, focal, CE có trọng số), sampler cân bằng, cùng LR head/backbone, EMA, độ phân giải 256 | 0 |
| 2 | T15 | Kết hợp các yếu tố thắng | 0 |
| 3 | I00–I08 | 1 view, TTA lật, 5/10 crop, đa tỉ lệ, gộp prob vs logit, độ phân giải 256/288/320, ensemble (backbone, seed), EMA, model soup, temperature scaling, AMP/FP16, gộp BN; độ trễ p50/p95/p99 | — |
| 4 | F01 (+F02) | Cấu hình chung kết, test một lần mỗi seed; F02 = cùng model, 1 view (thời gian thực) | 0, 1, 2 |
| 4 | T00 + I00 | Mốc so sánh | 0, 1, 2 |

Seed chỉ đổi khởi tạo head, thứ tự batch và augmentation (S5); split luôn là fold 0 nguyên bản.

**Luật chọn tự động (chỉ dựa trên val), ghi kèm số liệu căn cứ vào `decisions.json`:**

- *Backbone*: trong các backbone có macro-F1 val ≥ (tốt nhất − 0,005), vì 1 seed nên không phân biệt được trong khoảng
  này, chọn cái có thời gian train/epoch nhỏ nhất (rẻ nhất cho ~18 lần chạy ở Bước 2).
- *Kết hợp T15*: mỗi nhóm loại trừ nhau (khởi tạo / aug / mix / loss / sampler / LR / EMA / độ phân giải) lấy giá trị
  có Δ lớn nhất nếu Δ > std(T00); CE có trọng số và sampler cân bằng không đi cùng nhau. Ít hơn 2 yếu tố vượt nhiễu thì
  bổ sung các yếu tố Δ > 0 cho đủ 2.
- *Công thức chung kết*: macro-F1 val cao nhất trong {T00 (mean 3 seed), các ablation vượt nhiễu, T15}.
- *Suy luận chung kết*: phương pháp I00–I04 có macro-F1 val cao nhất, chỉ dùng nếu hơn 1 view ≥ 0,002; luôn thêm
  temperature scaling (T khớp trên val của từng seed).

## Phiên bản thư viện

Ghi tự động trong `logs/<exp_id>/seed<k>/config.json` (Python, torch, torchvision, timm, CUDA, cuDNN, tên GPU) và in ở
ô đầu notebook. Code cần `torch >= 2.3`, `torchvision >= 0.16`, `timm >= 1.0`, `pandas`, `numpy`, `matplotlib`, `openpyxl`.
Trọng số timm (tag ghi trong `results.xlsx`, sheet Backbones): `resnet50.a1_in1k`, `resnext50_32x4d.a1h_in1k`,
`convnext_tiny.fb_in1k`, `deit_small_patch16_224.fb_in1k`, `swin_tiny_patch4_window7_224.ms_in1k`,
`efficientnet_b0.ra_in1k`, `mobilenetv3_large_100.ra_in1k`, `vit_small_patch14_dinov2.lvd142m`. Tất cả, trừ DINOv2,
chỉ được huấn luyện trước trên ImageNet-1k.

## Cấu trúc thư mục (sau khi chạy)

```
code/            dataset.py model.py losses.py train.py inference.py benchmark.py (hoàn thiện từ starter/)
                 experiments.py (danh sách thí nghiệm, luật chọn, chạy 2 GPU) analysis.py (EDA, biểu đồ, xlsx)
                 test_code.py (test tự viết) lab_day2.ipynb
results.xlsx     Summary, Backbones, Training, Inference, Final, PerClass, Latency, Robustness, Decisions
report.md        báo cáo
curves/          <exp_id>_<mota>.png, mỗi thí nghiệm huấn luyện một ảnh (nhiều seed vẽ chồng)
predictions/     F01_seed<k>_{test,val}.csv, F01_uncal_*, F02_*, T00_* (định dạng eval.py)
figures/         EDA, đánh đổi F1–độ trễ, ablation, ma trận nhầm lẫn, Grad-CAM, reliability
eval_results/    đầu ra của eval.py score/grade
logs/            config.json, history.csv, lr_trace.csv, summary.json, logit val/test (.npz) của từng lần chạy
inference/       bảng suy luận, độ trễ, lệch phân phối
decisions.json   các lựa chọn tự động và số liệu căn cứ
```

Checkpoint (`*.pt`) không commit; nằm trong output Kaggle (`outputs/ckpt/`).

## Ghi chú về dữ liệu

`train_subset0.csv` gán ảnh `20170714-110407-3.jpg` nhãn 0 (Chinee apple), còn `labels.csv` gán nhãn 1 (Lantana).
Theo S1 file CSV được giữ nguyên, không sửa; ảnh này chỉ nằm trong train nên không ảnh hưởng chấm test.
Notebook in ra và lưu chỗ lệch này vào `eda.json` (`label_mismatch_vs_labels_csv`).
