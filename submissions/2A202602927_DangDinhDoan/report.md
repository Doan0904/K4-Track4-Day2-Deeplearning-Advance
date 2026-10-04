# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

**Sinh viên:** Đặng Đình Đoàn — 2A202602927 · **Phần cứng:** Kaggle, 2 × Tesla T4 · **Phần mềm:** Python 3.13, torch 2.11.0+cu128, torchvision 0.26.0, timm 1.0.29, CUDA 12.8

Mọi con số dưới đây lấy từ log của lần chạy thật (`logs/<exp_id>/seed<k>/`), từ `results.xlsx`, hoặc tính lại bằng `eval.py` trên `predictions/`. Số trích từ bài báo gốc được ghi rõ là *trích dẫn*.

## 1. Tóm tắt

- **Làm gì:** 8 backbone cùng công thức nền (B01–B08), 15 thí nghiệm công thức huấn luyện mỗi lần khác nền đúng một yếu tố (T01–T15) cùng nền `T00` chạy 3 seed, 21 dòng suy luận (I00–I08), rồi chung kết `F01` 3 seed. Toàn bộ trên fold 0 chia sẵn của DeepWeeds.
- **Cấu hình tốt nhất (offline, `F01`):** ConvNeXt-T (`convnext_tiny.fb_in1k`) tinh chỉnh toàn bộ, ảnh 256, label smoothing 0,1, thêm lật dọc, 12 epoch, suy luận TTA 10 crop (5 crop + bản lật), gộp xác suất, temperature scaling với T khớp trên val.
- **Kết quả test (3 seed, mean ± std, 3.507 ảnh, mỗi seed chạy test đúng một lần):** top-1 **97,55 ± 0,10 %**, macro-F1 **0,9693 ± 0,0011**, ECE **0,0054 ± 0,0016**. Recall Chinee apple **93,5 %** và Snake weed **93,5 %**.
- **So với mốc** (`T00` + `I00`, cùng 3 seed): macro-F1 **0,9507 ± 0,0040**, top-1 96,16 ± 0,37 %. Cải thiện macro-F1 **+0,0186**, lớn hơn std lớn nhất của hai nhóm (0,0040).
- **Cấu hình thời gian thực (`F02`):** cùng model, 1 view, FP32, p95 = **6,7 ms** ở batch 1 trên T4. macro-F1 test 0,9666 ± 0,0015, top-1 97,41 ± 0,12 %.
- **Kết luận chính:** (i) họ backbone (ConvNeXt / Swin / DeiT so với các CNN còn lại) quyết định nhiều nhất, nhưng thứ hạng này gắn với công thức nền cố định, không phải trần của kiến trúc (mục 3); (ii) trong công thức huấn luyện, chỉ **độ phân giải 256** có tác dụng rõ (+0,0080 trên val, ~5 lần std của `T00`); label smoothing và lật dọc chỉ hơn nền khoảng 1 std nên không phân biệt được; (iii) TTA 10 crop thêm khoảng +0,003 macro-F1 test với chi phí độ trễ ×8,4.
- `eval.py grade` cho phần I của rubric 20/20 (đề xuất, theo ngưỡng tạm thời của `eval.py`).

## 2. Dữ liệu và thiết lập

**Dữ liệu.** DeepWeeds, 17.509 ảnh RGB 256×256 (kiểm tra: tất cả đúng 256×256), 9 lớp. Dùng nguyên bản `train/val/test_subset0.csv` của tác giả (S1), không sửa, không gộp val vào train (S3).

| Tập | Số ảnh | Ghi chú |
|---|---|---|
| train | 10.501 | 59,97 % |
| val | 3.501 | 20,00 % |
| test | 3.507 | 20,03 % |

Các kiểm tra bắt buộc (`eda.json`): giao train∩val, train∩test, val∩test đều **bằng 0**; hợp ba tập đúng **17.509**; không thiếu file nào. Số ảnh theo lớp (đủ train/val/test ở `eda_class_counts.csv`): `Negatives` 9.106 (52,0 %), mỗi loài 1.009–1.126 ảnh; tỉ lệ lớp lớn nhất / nhỏ nhất = **9,02**. So với Table 1 của bài báo, đếm thật lệch hai ảnh: Chinee apple 1.126 (bài báo 1.125) và Lantana 1.063 (bài báo 1.064). Điều này khớp với một điểm lệch nhãn khác ở dưới. Hình: `figures/eda_class_distribution.png`, `figures/eda_samples.png`.

**Một điểm bất thường trong dữ liệu gốc.** Ảnh `20170714-110407-3.jpg` mang nhãn 0 (Chinee apple) trong `train_subset0.csv` nhưng nhãn 1 (Lantana) trong `labels.csv`. Theo S1 mình giữ nguyên file subset (ảnh này chỉ nằm trong train nên không ảnh hưởng chấm test) và chỉ ghi lại.

**Thống kê ảnh:** mean kênh (0,379; 0,389; 0,379), std (0,237; 0,235; 0,233), khác ImageNet khá nhiều; mình vẫn dùng mean/std của trọng số timm (ImageNet) vì đó là phân phối mà trọng số tiền huấn luyện đã quen.

**Kiểm tra pipeline trước khi chạy thật** (`sanity/sanity.json`, `sanity/*.png`): loss ban đầu của head mới **2,185** (kỳ vọng ln 9 = 2,197); overfit 16 ảnh xuống loss **0,0013**; focal γ=0 khác CE **0,0** (cũng có unit test); ảnh sau augmentation đã giải chuẩn hoá cho thấy ảnh và nhãn khớp nhau. 29 unit test tự viết (`code/test_code.py`) chạy trên CPU: focal γ=0 ≡ CE, label smoothing ≡ `torch`, CutMix tính λ theo diện tích thật, Mixup, không decay norm/bias, đóng băng giữ BN ở eval, lịch LR, EMA, gộp BN trên ResNet/EfficientNet/MobileNet, TTA, temperature scaling tìm lại đúng T, soup, đo độ trễ.

**Chỉ số** theo đúng README mục 2.2 (macro-F1 là chỉ số chính, ECE 15 bin, std mẫu `ddof=1`). Mọi số test trong báo cáo đã **tính lại bằng `eval.py score/grade` trên file trong `predictions/`** và khớp với bảng của notebook.

**Công thức nền `T00`** (GUIDE 1.4): AdamW, LR backbone 1e-4 / head 1e-3, weight decay 0,05 không áp cho norm/bias/`pos_embed`, warmup 1 epoch rồi cosine, CE, batch 64, 12 epoch, AMP, `RandomResizedCrop(224)` + lật ngang. Val/test: `Resize(256)` → `CenterCrop(224)` (với ảnh gốc 256 thì là center-crop). Chọn checkpoint theo macro-F1 val (hòa thì lấy epoch sớm hơn). Seed chỉ đổi khởi tạo head, thứ tự batch và augmentation (S5). Cùng cấu hình chạy hai lần cùng seed có thể lệch nhẹ vì cuDNN `benchmark=True` (không bật chế độ deterministic để chạy nhanh).

**Cách chọn tự động.** Các lựa chọn (backbone, kết hợp T15, công thức chung kết, suy luận chung kết) do code chọn theo luật cố định **chỉ dựa trên val**, ghi cùng số liệu căn cứ vào `decisions.json`. Test chỉ được tính ở Bước 4, sau khi mọi quyết định đã ghi.

**Thời gian chạy** (2 × T4, hai tiến trình song song): Bước 1 36 phút, Bước 2 117 phút, Bước 3 12 phút, Bước 4 56 phút, tổng khoảng 3,7 giờ.

## 3. So sánh backbone (B01–B08)

Cùng công thức nền, seed 0, ảnh 224. Số tham số và GMAC đếm bằng `torch.utils.flop_counter` (MAC = FLOPs/2). Độ trễ ở đây là đo sơ bộ cuối mỗi lần train (FP32, batch 1, 50 lần).

| exp_id | Backbone (tag trọng số) | Tham số (M) | GMAC | macro-F1 val | top-1 val | s/epoch | p50 b1 (ms) |
|---|---|---|---|---|---|---|---|
| B01 | ResNet-50 (`a1_in1k`) | 23,5 | 4,09 | 0,7989 | 0,8558 | 34 | 6,3 |
| B02 | ResNeXt-50 32x4d (`a1h_in1k`) | 23,0 | 4,23 | 0,7652 | 0,8143 | 42 | 11,5 |
| **B03** | **ConvNeXt-T (`fb_in1k`)** | 27,8 | 4,45 | **0,9557** | 0,9666 | 51 | 5,6 |
| B04 | DeiT-S (`fb_in1k`) | 21,7 | 4,24 | 0,9476 | 0,9632 | 33 | 5,3 |
| B05 | Swin-T (`ms_in1k`) | 27,5 | 4,49 | **0,9580** | 0,9700 | 62 | 9,6 |
| B06 | EfficientNet-B0 (`ra_in1k`) | 4,0 | 0,38 | 0,7514 | 0,8181 | 32 | 14,2 |
| B07 | MobileNetV3-L (`ra_in1k`) | 4,2 | 0,22 | 0,6364 | 0,7418 | 23 | 6,5 |
| B08 | DINOv2 ViT-S/14, đóng băng + linear probe (thưởng) | 21,6 | 5,51 | 0,8734 | 0,8972 | 17 | 6,6 |

Đủ ràng buộc: ResNet (B01), ResNeXt (B02) và ConvNeXt (B03), transformer (B04, B05), mạng nhẹ (B06, B07). Đường cong từng thí nghiệm: `curves/B01_resnet50.png` … `curves/B08_dinov2_small_linear_probe.png`. Biểu đồ F1 theo độ trễ và GMAC: `figures/backbones_tradeoff.png`.

**Chọn backbone.** Luật: lấy các backbone có macro-F1 val trong khoảng 0,005 của tốt nhất (1 seed nên không phân biệt được trong khoảng này), rồi chọn cái train nhanh nhất cho khoảng 18 lần chạy ở Bước 2. Hai ứng viên là ConvNeXt-T (0,9557) và Swin-T (0,9580); chọn **ConvNeXt-T** vì 51 s/epoch so với 62 s/epoch của Swin-T, và độ trễ batch 1 thấp hơn (5,6 so với 9,6 ms). Chênh 0,0023 giữa hai model nằm trong khoảng nhiễu của `T00` (std 0,0016), nên mình không khẳng định ConvNeXt-T tốt hơn Swin-T.

**Nhận xét.**

- Ba model transformer/ConvNeXt (B03–B05) đạt 0,948–0,958, cách xa nhóm còn lại (0,64–0,80). Khoảng cách này **không nên đọc là "kiến trúc ResNet/EfficientNet kém"**: ở B06 và B07, F1 val vẫn còn tăng ở epoch cuối (epoch tốt nhất 12/12, mô hình chưa hội tụ); ở B01 và B02, F1 val đi ngang từ khoảng epoch 9–11 ở mức thấp (0,80 và 0,76) trong khi loss val không giảm thêm. Cả bốn dùng chung một công thức (AdamW, LR 1e-4, 12 epoch) có thể không hợp với chúng. Mình chưa chạy thí nghiệm nào để kiểm chứng nguyên nhân, nên đây chỉ là giả thuyết: trọng số `a1_in1k` của ResNet-50 được huấn luyện bằng công thức "ResNet strikes back" (loss BCE) có thể khó tinh chỉnh bằng CE với LR thấp; mạng nhẹ thường cần LR và số epoch khác. Kết luận hợp lệ chỉ là: *với công thức nền này, trong 12 epoch*, ConvNeXt/Swin/DeiT đứng đầu.
- FLOPs không dự đoán được thời gian: ResNeXt-50 (4,2 GMAC) p50 11,5 ms, còn ConvNeXt-T (4,5 GMAC) 5,6 ms; EfficientNet-B0 chỉ 0,38 GMAC lại chậm nhất (14,2 ms). Swin-T chậm hơn DeiT-S gần gấp đôi dù cùng cỡ GMAC.
- DINOv2 đóng băng (chỉ train head) đạt 0,8734, thấp hơn ConvNeXt-T tinh chỉnh toàn bộ (0,9557) nhưng train nhanh nhất (17 s/epoch). Ảnh cỏ dại ngoài đồng khác xa dữ liệu tiền huấn luyện nên tinh chỉnh vẫn cần thiết (khớp với `T02` dưới đây).

## 4. Công thức huấn luyện (T00–T15)

Backbone: ConvNeXt-T. `T00` chạy 3 seed để đo nhiễu: macro-F1 val 0,9557 / 0,9533 / 0,9526, **mean 0,9539, std 0,0016**. Mỗi `T01–T14` khác `T00` đúng một yếu tố, 1 seed (seed 0). Tiêu chí "vượt nhiễu": |Δ| lớn hơn 0,0016 (std của `T00`). Vì mỗi dòng chỉ 1 seed, chênh lệch cỡ 1–2 lần std **không đủ để kết luận**.

| exp_id | Trục | Khác `T00` ở điểm nào | macro-F1 val | Δ so với mean `T00` | Nhận xét |
|---|---|---|---|---|---|
| T01 | A | khởi tạo từ đầu | 0,3107 | −0,6432 | sụp đổ |
| T02 | A | đóng băng backbone, chỉ train head | 0,6891 | −0,2648 | kém hẳn |
| T03 | B | + ColorJitter | 0,9479 | −0,0060 | hại |
| T04 | B | TrivialAugment | 0,9516 | −0,0023 | không phân biệt được |
| T05 | B | + lật dọc | 0,9556 | +0,0017 | không phân biệt được |
| T06 | B | Mixup (α=0,2) | 0,9551 | +0,0012 | không phân biệt được |
| T07 | B | CutMix (α=1) | 0,9475 | −0,0064 | hại |
| T08 | C | label smoothing 0,1 | 0,9556 | +0,0017 | không phân biệt được (nhưng ECE val 0,097) |
| T09 | C | focal γ=2 | 0,9499 | −0,0040 | hại |
| T10 | C | CE có trọng số theo lớp | 0,9451 | −0,0087 | hại |
| T11 | D | sampler cân bằng lớp | 0,9493 | −0,0046 | hại |
| T12 | E | LR head = LR backbone (1e-4) | 0,9456 | −0,0082 | hại |
| T13 | F | EMA 0,998 | 0,9542 | +0,0003 | không phân biệt được |
| **T14** | G | **ảnh 256** | **0,9618** | **+0,0080** | **tốt hơn rõ** (~5 lần std) |
| T15 | kết hợp | 256 + label smoothing + lật dọc | 0,9637 | +0,0098 | xem dưới |

Đường cong từng thí nghiệm: `curves/T00_baseline.png` … `curves/T15_combo.png` (T00 vẽ chồng 3 seed). Biểu đồ Δ: `figures/training_ablation.png`. Bảng đầy đủ kèm F1 hai lớp khó, ECE và thời gian: sheet `Training` của `results.xlsx`.

**Cách chọn T15 (tham lam theo trục, đã nêu rõ):** mỗi nhóm loại trừ nhau lấy yếu tố có Δ lớn nhất nếu Δ > std(`T00`): ở đây độ phân giải (T14), label smoothing (T08) và lật dọc (T05) vượt ngưỡng; T05 và T08 chỉ vượt 0,0016 một chút (+0,0017), tức luật đã nhận hai yếu tố **sát ngưỡng**. Thứ tự các trục có thể ảnh hưởng kết quả.

**Cộng dồn hay triệt tiêu?** Tổng Δ riêng lẻ của ba yếu tố là +0,0114, T15 đạt +0,0098: gần cộng dồn. Nhưng T15 hơn T14 (chỉ độ phân giải) chỉ +0,0019, gần bằng nhiễu (1 seed mỗi bên), nên mình **không chứng minh được** label smoothing và lật dọc đóng góp riêng.

**Trả lời các câu hỏi của trục:**

- *A, khởi tạo:* với ~10 nghìn ảnh, train từ đầu 12 epoch không kịp (0,31) và đóng băng không đủ (0,69). Tinh chỉnh toàn bộ là bắt buộc.
- *B, augmentation:* ColorJitter và CutMix làm giảm (−0,006); Mixup, TrivialAugment, lật dọc không phân biệt được với nền. Lật dọc là hợp lệ về nghĩa (ảnh chụp cây từ trên xuống, không có "hướng"). Với CutMix, giả thuyết là hộp cắt dán che mất vật thể nhỏ trong khi nhãn vẫn là loài cỏ (chưa kiểm chứng).
- *C, loss và D, sampler:* với `Negative` chiếm 52 %, các cách cân bằng lớp (CE có trọng số T10, sampler T11) **đều làm giảm** macro-F1 val (−0,009 và −0,005), và F1 Snake weed không tăng. Có thể vì nền đã đủ cân bằng ở các lớp hiếm (mỗi loài ~600 ảnh train), nên cân bằng thêm chỉ làm tăng nhiễu; chưa kiểm chứng.
- *E:* đặt LR head bằng LR backbone làm giảm 0,008, tức LR head gấp 10 lần có ích.
- *F, EMA:* không giúp "miễn phí" ở đây (best EMA 0,9542, best thường 0,9557 trong cùng lần chạy).
- *G:* tăng từ 224 lên 256 là yếu tố có Δ lớn nhất (+0,0080), đổi lại 66 so với 51 s/epoch. Ảnh gốc đã là 256×256, nên 224 là cắt/thu nhỏ ảnh; giả thuyết (chưa kiểm chứng) là chi tiết nhỏ trên cỏ dại bị mất khi thu nhỏ. Mình chưa chạy thí nghiệm 10 so với 20 epoch nên không trả lời được câu về độ dài huấn luyện.

## 5. Suy luận (I00–I08)

Model: `T15` seed 0, trên **val**, không huấn luyện lại. Mốc `I00`: `Resize(293)` → `CenterCrop(256)`, FP32. Độ trễ: Tesla T4, torch 2.11.0+cu128, `cudnn.benchmark=True`, warmup 10 lần, `torch.cuda.synchronize()` trước và sau, 100 lần đo (50 lần cho batch 32), báo p50/p95/p99, **không tính tiền xử lý CPU** (đầu vào là tensor đã ở trên GPU; việc tạo view TTA trên GPU được tính). Toàn bộ trong sheet `Inference` và `Latency`.

| Mã | Phương pháp | K | macro-F1 val | top-1 val | ECE val | p50 b1 (ms) | p95 / p99 (ms) | Chi phí so với I00 |
|---|---|---|---|---|---|---|---|---|
| I00 | 1 view (mốc) | 1 | 0,9637 | 0,9720 | 0,0874 | 6,48 | 6,72 / 6,75 | ×1,00 |
| I01 | TTA lật ngang | 2 | 0,9632 | 0,9717 | 0,0878 | 11,87 | 12,03 / 12,24 | ×1,83 |
| I02a | TTA 5 crop | 5 | 0,9675 | 0,9751 | 0,0923 | 27,43 | 28,38 / 28,54 | ×4,23 |
| **I02b** | **TTA 10 crop** | 10 | **0,9676** | 0,9751 | 0,0922 | 54,27 | 55,57 / 56,33 | ×8,37 |
| I02c | TTA 3 tỉ lệ | 3 | 0,9673 | 0,9751 | 0,0971 | 24,15 | 24,79 / 24,82 | ×3,73 |
| I03b,c,d | gộp **logit** (I02a, I02b, I02c) | 5, 10, 3 | 0,9670, 0,9676, 0,9667 | – | 0,090 / 0,089 / 0,093 | cùng chi phí | | |
| I04 | độ phân giải kiểm tra 288 / 320 / 352 | 1 | 0,9511 / 0,9451 / 0,9423 | 0,961 / 0,956 / 0,953 | 0,083–0,087 | 7,9 / 9,8 / 11,3 | | ×1,2 / 1,5 / 1,7 |
| I05a | Ensemble 3 backbone tốt nhất (B05, B03, B04, ở 224) | 3 | 0,9667 | **0,9769** | 0,0158 | 20,5 | 21,5 / 22,1 | ×3,17 |
| I05b | Ensemble cả 7 backbone | 7 | 0,9578 | 0,9697 | 0,1283 | không đo | | |
| I05c | Ensemble 3 seed T00 (ở 224) | 3 | 0,9609 | 0,9714 | 0,0110 | 16,9 | 18,8 / 19,8 | ×2,61 |
| I06a | Trọng số EMA (từ T13) | 1 | 0,9542 | – | – | như I00 | | ×1 |
| I06b | Model soup 3 seed T00 | 1 | 0,9200 | 0,9389 | 0,0435 | như I00 | | ×1 |
| **I07a** | **Temperature scaling trên I00** (T = 0,627) | 1 | 0,9637 | 0,9720 | **0,0052** | 6,48 | | ×1 |
| I08a | AMP | 1 | 0,9637 | 0,9720 | 0,0874 | 7,82 | 8,18 / 9,40 | ×1,21 |
| I08b | FP16 | 1 | 0,9637 | 0,9720 | 0,0877 | 5,67 | 6,22 / 6,38 | ×0,88 |
| I08c | Gộp BN vào conv (trên ResNet-50 B01, vì ConvNeXt không có BN) | 1 | 0,7993 (bằng B01 gốc) | 0,8558 | 0,0143 | 4,88 so với 5,85 chưa gộp | | ×0,83 |

Biểu đồ đánh đổi: `figures/inference_tradeoff.png`.

**Nhận xét.**

- **TTA.** Lật ngang đơn lẻ không giúp (0,9632 so với 0,9637; trong nhiễu). Cắt nhiều vị trí giúp khoảng +0,0038 (5 crop) và +0,0039 (10 crop); 10 crop không hơn 5 crop dù tốn gấp đôi. Gộp logit hay gộp xác suất không phân biệt được (chênh ≤ 0,0006). Trên val, TTA 10 crop đổi nhãn 33 ảnh: 21 từ sai thành đúng, 10 từ đúng thành sai (`inference/tta_flips.json`), tức TTA cũng có thể phá những ca đúng.
- **Độ phân giải kiểm tra** cao hơn lúc train **làm giảm** F1 (0,9511 ở 288, giảm tiếp ở 320 và 352), không như kỳ vọng của FixRes. Ảnh gốc chỉ 256×256 nên phóng to chỉ nội suy chứ không thêm thông tin (giả thuyết; chưa kiểm chứng).
- **Ensemble.** 3 model khác họ (I05a) cho top-1 val cao nhất (0,9769) và ECE thấp (0,0158) nhưng macro-F1 chỉ ngang TTA 5 crop, với chi phí ×3,2; ensemble 3 seed cùng cấu hình T00 (I05c) hơn từng seed 0,005 trở lên. Ensemble cả 7 backbone **tệ hơn** (0,9578) vì lẫn các model yếu.
- **Soup thất bại** (0,9200, thấp hơn từng seed 0,953–0,956). Giả thuyết hợp lý: head mỗi seed khởi tạo ngẫu nhiên khác nhau nên trung bình trọng số các head không có nghĩa; chưa kiểm chứng (chưa thử soup chỉ phần backbone hoặc chung khởi tạo head).
- **Hiệu chuẩn.** Label smoothing làm model kém tự tin: T = 0,627 (< 1, tức làm nhọn lại), ECE val 0,0874 → 0,0052 (ECE khi khớp T chéo trên hai nửa val: 0,0050, nên không phải quá khớp). Accuracy không đổi.
- **Độ chính xác số học.** FP16 và AMP cho đúng dự đoán như FP32 (0 ảnh val đổi nhãn). Ở batch 1, **AMP chậm hơn FP32** (7,8 so với 6,5 ms) còn FP16 nhanh hơn 12 % (5,7 ms); ở batch 32 cả hai nhanh hơn FP32 khoảng 2,5–3,4 lần trên ConvNeXt-T (236 ảnh/s FP32 so với 633 AMP và 791 FP16, sheet `Latency`). Gộp BN trên ResNet-50: 53 cặp conv-BN, sai số logit lớn nhất 9,3e-5, dự đoán không đổi, p50 giảm 17 % (5,85 → 4,88 ms), kèm FP16 còn 4,46 ms.
- **Offline và thời gian thực.** Dữ liệu ủng hộ kết luận của slide: TTA/ensemble cho thêm chút độ chính xác với chi phí lớn, còn FP16, gộp BN và temperature scaling không tốn thêm hoặc giảm độ trễ. Điều khác với slide: trên T4 với model này, cả TTA 10 crop (p95 55,6 ms) vẫn nằm trong ngân sách 100 ms.

**Suy luận chốt cho chung kết** (luật: I00–I04 có macro-F1 val cao nhất, chỉ dùng nếu hơn I00 ít nhất 0,002): **I02b (TTA 10 crop)** (0,9676 so với 0,9637, hơn 0,0039). Luôn thêm temperature scaling, T khớp trên val của từng seed.

## 6. Cấu hình tốt nhất và chạy test

**`F01`** = ConvNeXt-T (`convnext_tiny.fb_in1k`) · tinh chỉnh toàn bộ · ảnh 256 · `RandomResizedCrop(256)` + lật ngang + lật dọc · label smoothing 0,1 · AdamW (LR backbone 1e-4, head 1e-3, wd 0,05, không decay norm/bias), warmup 1 epoch + cosine, batch 64, 12 epoch, AMP · chọn checkpoint theo macro-F1 val · suy luận: `Resize(293)`, 5 crop 256 (4 góc + giữa) và bản lật, trung bình xác suất, temperature scaling. Seed 0, 1, 2. Cấu hình đầy đủ ở `decisions.json` và `logs/F01/seed*/config.json`. Cấu hình `F02` giống hệt `F01` nhưng 1 view (center-crop 256), T khớp riêng trên logit 1 view.

Test được chạy **đúng một lần cho mỗi seed** trong Bước 4, cho `F01`, `F02` và mốc `T00`, sau khi cấu hình đã chốt trên val. Kết quả (3.507 ảnh, mean ± std qua 3 seed, tính bằng `eval.py`):

| | macro-F1 test | top-1 test | balanced acc | ECE | NLL |
|---|---|---|---|---|---|
| **F01** (TTA 10 crop + TS) | **0,9693 ± 0,0011** | **0,9755 ± 0,0010** | 0,9656 ± 0,0023 | **0,0054 ± 0,0016** | 0,0843 ± 0,0018 |
| F01 chưa temperature scaling | 0,9693 ± 0,0011 | 0,9755 ± 0,0010 | 0,9656 ± 0,0023 | 0,0927 ± 0,0005 | 0,1662 ± 0,0016 |
| F02 (1 view + TS, thời gian thực) | 0,9666 ± 0,0015 | 0,9741 ± 0,0012 | 0,9647 ± 0,0016 | 0,0055 ± 0,0006 | 0,0923 ± 0,0035 |
| Mốc `T00` + `I00` | 0,9507 ± 0,0040 | 0,9616 ± 0,0037 | 0,9522 ± 0,0030 | 0,0082 ± 0,0022 | 0,1201 ± 0,0053 |

- **Cải thiện so với mốc:** macro-F1 +0,0186, lớn hơn std lớn nhất của hai nhóm (0,0040) và lớn hơn 0,01; top-1 +1,39 điểm phần trăm. Chênh giữa val và test của `F01`: macro-F1 val 0,9689 so với test 0,9693 (chênh 0,0004).
- **Đóng góp của từng phần:** mốc → `F02` (công thức huấn luyện, 1 view): +0,0159 macro-F1 test; `F02` → `F01` (TTA 10 crop): +0,0027, cả ba seed đều dương (+0,0034, +0,0029, +0,0019) nhưng chênh chỉ khoảng 2 lần std, nên TTA có giúp nhẹ, còn mức giúp thì chưa chắc chắn.
- **So với bài báo gốc (trích dẫn, chỉ mang tính tham khảo):** ResNet-50 95,7 % và Inception-v3 95,1 % là weighted average accuracy trung bình 5 fold với ~100 epoch và augmentation mạnh; con số của mình là top-1 không trọng số, **một fold**, 12 epoch. Hai định nghĩa không trùng, nên không kết luận "hơn" hay "kém".

**Hai lớp khó** (mean 3 seed, `F01`; mốc `T00` trong ngoặc):

| Lớp | Precision | Recall | F1 |
|---|---|---|---|
| Chinee apple | 0,962 (0,947) | **0,935** (0,888) | 0,948 (0,916) |
| Snake weed | 0,958 (0,927) | **0,935** (0,926) | 0,946 (0,927) |

Recall đều vượt mốc 88,5 % và 88,8 % của bài báo (trích dẫn; cùng lưu ý về định nghĩa). Bảng 9 lớp: sheet `PerClass`.

**Ma trận nhầm lẫn** (cộng 3 seed, `figures/confusion_F01_test.png`, `confusion_T00_test.png`, `confusion_F02_test.png`):

- `F01` sai 258 lượt (3 seed × 3.507 ảnh), `T00` sai 404 lượt.
- Nhầm lẫn lớn nhất của `F01` **không phải** Chinee ↔ Snake mà là loài → `Negatives`: Chinee apple → Negatives 32 (4,7 %), Snake weed → Negatives 31 (5,1 %), Rubber vine → Negatives 23 (3,8 %), cộng chiều ngược lại Negatives → Prickly acacia 18, → Rubber vine 17.
- Cặp Chinee ↔ Snake: `F01` 12 (Chinee → Snake, 1,8 %) và 7 (Snake → Chinee, 1,1 %); `T00` 26 (3,8 %) và 12. Bài báo (trích dẫn) báo 3,4 % và 4,1 %. Parkinsonia → Prickly acacia: `F01` 0 lượt, `T00` 3 (bài báo 1,3 %).
- Công thức mới giảm rõ nhầm Chinee → Snake (26 → 12, −54 %), Negatives → Lantana (41 → 7) và Negatives → Prickly acacia (26 → 18).

**Phân tích ảnh sai** (`figures/errors_gradcam_F01_seed0.png`, `figures/chinee_snake_confusions.png`): `F01` seed 0 sai 84 ảnh, chỉ 5 ảnh là Chinee ↔ Snake. Trong các ảnh xem được, phần lớn nền là thảm cỏ khô, lá và cành lẫn lộn; Grad-CAM thường tập trung vào một vài mảng lá hoặc cành nhỏ, và nhiều ca có độ tin cậy rất cao (0,98–0,99) dù sai. Ở một số ảnh bị nhầm giữa `Negatives` và một loài, có vẻ vật thể mục tiêu rất nhỏ hoặc bị khuất, nên nhãn "loài" hay "không phải loài" dường như phụ thuộc vào một mảng nhỏ. Đó là quan sát trên 12 ảnh, không phải thống kê; giả thuyết (chưa kiểm chứng) là phần sai còn lại do vật thể quá nhỏ/ngụy trang và một số nhãn có thể nhiễu (đã thấy một nhãn mâu thuẫn trong CSV).

## 7. Kết luận và khuyến nghị

**Cấu hình nào tốt nhất, hơn mốc bao nhiêu?** `F01`: macro-F1 test 0,9693 ± 0,0011 so với mốc 0,9507 ± 0,0040, hơn +0,0186; vượt nhiễu.

**Yếu tố nào đóng góp nhiều nhất?** Theo thứ tự độ lớn: (1) *chọn họ backbone và khởi tạo tiền huấn luyện* (ConvNeXt/Swin/DeiT tinh chỉnh ≈ 0,95 so với 0,64–0,80 của các model còn lại, từ đầu 0,31, đóng băng 0,69), nhưng độ lớn này phụ thuộc công thức nền cố định; (2) *công thức huấn luyện*: chủ yếu ảnh 256 (+0,008 val), phần còn lại gần nhiễu; (3) *suy luận*: TTA 10 crop thêm khoảng +0,003 test; temperature scaling không đổi độ chính xác nhưng giảm ECE từ 0,093 xuống 0,005.

**Triển khai trên robot, ngân sách 30–100 ms/khung:** chọn **`F02`** (ConvNeXt-T, 1 view, FP32 hoặc FP16, temperature scaling): p95 **6,7 ms** (FP32) và 6,5 ms (FP16) ở batch 1 trên T4, macro-F1 test 0,9666 ± 0,0015, recall Chinee apple 93,5 % và Snake weed 94,3 %. Nếu còn dư ngân sách thì `F01` (TTA 10 crop, p95 55,6 ms) cho thêm khoảng +0,003 macro-F1. **Lưu ý:** T4 không phải phần cứng robot (bài báo trích dẫn 53,4 ms cho ResNet-50 TensorRT trên Jetson TX2); độ trễ trên thiết bị đích phải đo lại, và có thể gần với ngân sách hơn nhiều.

**Lệch phân phối** (thưởng, trên **val**, không dùng test; `inference/robustness.csv`, model `F01` seed 0, 1 view): macro-F1 val 0,9637 giảm còn 0,938 khi tối bằng nửa, 0,924 khi thêm nhiễu σ = 0,05, 0,588 khi giảm tương phản một nửa và 0,400 khi làm mờ σ = 2. Temperature scaling với T khớp trên val sạch **không còn đáng tin** khi miền đổi: ECE sau TS vẫn tốt hơn trước TS với tối và nhiễu (ví dụ nhiễu 0,10: 0,026 so với 0,131), nhưng **tệ hơn** với giảm tương phản (0,214 so với 0,093) và làm mờ (0,119 so với 0,055 và 0,289 so với 0,169). Nếu thiết bị thật có ảnh mờ hoặc phẳng tương phản, cần tăng cường augmentation tương ứng và hiệu chuẩn lại.

## 8. Hạn chế và việc tiếp theo

- **Một fold, chia ngẫu nhiên không theo địa điểm** nên điểm test có thể lạc quan so với khi gặp địa điểm hoặc mùa mới. Test chỉ 3.507 ảnh; mỗi loài khoảng 200 ảnh test, nên một ảnh sai đổi recall khoảng 0,5 điểm phần trăm.
- **Ablation 1 seed:** `T01–T15` chỉ chạy seed 0, std 0,0016 lấy từ `T00`. Các chênh lệch dưới ~0,003 (T04, T05, T06, T08, T13) được coi là không phân biệt được. T15 hơn T14 chỉ 0,0019 nên đóng góp riêng của label smoothing và lật dọc chưa được chứng minh, và `F01` chưa được so với một cấu hình "chỉ ảnh 256". Chọn tham lam theo trục nên thứ tự trục có thể ảnh hưởng.
- **Công thức nền chung** có thể bất lợi cho ResNet/ResNeXt/EfficientNet/MobileNet (mục 3); chưa thử LR/epoch riêng cho từng backbone, nên bảng B chỉ đúng với công thức này, 12 epoch.
- **Các suy luận phụ chạy trên một seed/một model:** I05, I06 dùng logit lưu lúc huấn luyện (AMP) hoặc model ở 224 chứ không phải model `T15`; soup thất bại và nguyên nhân chưa kiểm chứng; I08c (gộp BN) minh hoạ trên ResNet-50 vì model chính không có BN.
- **Độ trễ chỉ trên Tesla T4**, FP32/AMP/FP16, không tính tiền xử lý CPU và giải mã JPEG; chưa thử TensorRT/ONNX.
- **Chưa làm:** đa fold, chưng cất, test-time adaptation, ONNX (các mục thưởng này không làm). Đã làm thêm: linear probe DINOv2 (B08), thử lệch phân phối, Grad-CAM.
- **Chưa có thí nghiệm tách riêng** vai trò của độ phân giải so với `RandomResizedCrop`, và kiểm tra các giả thuyết ở mục 3, 4, 5 (đều ghi rõ là "chưa kiểm chứng").
- **Nếu có thêm một ngày:** chạy lại bảng backbone với LR/epoch tinh chỉnh riêng cho từng model, chạy ablation vài seed cho các yếu tố sát ngưỡng, thêm fold 1–4, và đo độ trễ trên phần cứng đích.

**Sự cố khi chạy (ghi để minh bạch).** (1) Lần chạy Kaggle đầu tiên dừng ở Bước 3 vì lỗi code của mình (ensemble đo độ trễ Swin ở 256 trong khi Swin cố định 224); đã sửa và **chạy lại toàn bộ từ đầu**. Lần đầu mới có kết quả val, chưa chạm test (Bước 4 chưa chạy); mọi số trong báo cáo này là từ lần chạy thứ hai đầy đủ. (2) Trong lần chạy thứ hai, Bước 5 trên Kaggle gặp `KeyError: 'Label'` ở phân tích ảnh sai (lỗi code của mình). Phần lệch phân phối của Bước 5 đã chạy xong trên Kaggle; phần còn lại (ma trận nhầm lẫn, ảnh sai + Grad-CAM, biểu đồ tổng hợp, `results.xlsx`, gom file) được chạy lại cục bộ trên **CPU** từ chính các file `outputs/` tải từ Kaggle, **không huấn luyện lại**. Các số test trong `results.xlsx` và báo cáo này được tính lại bằng `eval.py` từ `predictions/` và khớp với notebook (`code/lab_day2_kaggle_run.ipynb` là notebook đã chạy trên Kaggle, giữ cả đoạn lỗi).

## 9. Phụ lục: danh sách thí nghiệm

| Nhóm | exp_id | Seed | Có đường cong | Có dự đoán |
|---|---|---|---|---|
| Backbone | B01–B08 | 0 | `curves/B0x_*.png` | val (`logs/<exp_id>/seed0/val_outputs.npz`, logit val) |
| Công thức nền | T00 | 0, 1, 2 | `curves/T00_baseline.png` | `predictions/T00_seed{0,1,2}_{val,test}.csv` |
| Ablation | T01–T14 | 0 | `curves/T01…T14_*.png` | val |
| Kết hợp | T15 | 0 | `curves/T15_combo.png` | val |
| Suy luận | I00–I08 | (model T15 seed 0) | – | `inference/inference.csv`, `val_view_bank.npz` |
| Chung kết | F01 | 0, 1, 2 | `curves/F01_final.png` | `predictions/F01_seed*_{val,test}.csv`, `F01_uncal_*`, `F02_*` |

Cấu hình đầy đủ từng lần chạy: `logs/<exp_id>/seed<k>/config.json` (có tag trọng số timm, version thư viện, GPU); log theo epoch: `history.csv`; LR theo bước: `lr_trace.csv`. Quyết định tự động và số liệu căn cứ: `decisions.json`. Notebook chạy lại: `code/lab_day2.ipynb` (xem `README.md`). Link notebook Kaggle: https://www.kaggle.com/code/ngan1234/computer-vision
