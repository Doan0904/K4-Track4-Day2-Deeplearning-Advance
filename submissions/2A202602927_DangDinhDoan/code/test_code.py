"""test_code.py - kiểm tra tự viết cho các phần dễ sai (RUBRIC mục H). Chạy được không cần GPU, không cần dữ liệu.

    cd submissions/2A202602927_DangDinhDoan/code && python -m unittest test_code -v

Kiểm tra: focal gamma=0 == CE; label smoothing tự cài == torch và eps=0 == CE; trọng số lớp;
CutMix lam = diện tích thật, Mixup trộn đúng, mixed_loss; nhóm tham số (không decay norm/bias);
đóng băng giữ BN ở eval; lịch LR warmup + cosine; EMA; gộp BN (ResNet, EfficientNet với BatchNormAct);
TTA (lật, 5/10 crop, gộp prob/logit); temperature scaling tìm lại đúng T; soup; đo độ trễ;
parse_overrides; định dạng dự đoán qua eval.read_pred.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import benchmark as BM
import dataset as D
import inference as INF
import losses as L
import model as M
import train as TR
from eval import read_pred, save_predictions


def tiny_timm(name="resnet18", **kw):
    import timm
    return timm.create_model(name, pretrained=False, num_classes=9, **kw)


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.z = torch.randn(64, 9) * 3
        self.y = torch.randint(0, 9, (64,))

    def test_focal_gamma0_equals_ce(self):
        self.assertLess(abs(L.FocalLoss(0.0)(self.z, self.y) - F.cross_entropy(self.z, self.y)).item(), 1e-6)

    def test_focal_downweights_easy(self):
        self.assertLess(L.FocalLoss(2.0)(self.z, self.y).item(), F.cross_entropy(self.z, self.y).item())

    def test_focal_alpha(self):
        a = torch.rand(9)
        ref = (F.cross_entropy(self.z, self.y, reduction="none") * a[self.y]).mean()
        self.assertLess(abs(L.FocalLoss(0.0, a)(self.z, self.y) - ref).item(), 1e-6)

    def test_label_smoothing(self):
        self.assertLess(abs(L.LabelSmoothingCE(0.0)(self.z, self.y) - F.cross_entropy(self.z, self.y)).item(), 1e-6)
        ref = nn.CrossEntropyLoss(label_smoothing=0.1)(self.z, self.y)
        self.assertLess(abs(L.LabelSmoothingCE(0.1)(self.z, self.y) - ref).item(), 1e-6)

    def test_build_criterion(self):
        for k in ("ce", "ls", "focal"):
            self.assertTrue(torch.isfinite(L.build_criterion(k)(self.z, self.y)))
        w = L.class_weights([10] * 8 + [90])
        self.assertTrue(torch.isfinite(L.build_criterion("ce_weighted", weight=w)(self.z, self.y)))
        with self.assertRaises(ValueError):
            L.build_criterion("ce_weighted")

    def test_class_weights(self):
        n = [600, 640, 620, 610, 640, 600, 640, 610, 5460]
        w = L.class_weights(n).numpy()
        self.assertAlmostEqual(w.mean(), 1.0, places=5)
        self.assertAlmostEqual(w[0] / w[8], 5460 / 600, places=4)
        wb = L.class_weights(n, beta=0.999).numpy()
        self.assertAlmostEqual(wb.sum(), 9.0, places=4)
        self.assertGreater(wb[0], wb[8])
        self.assertLess(wb[0] / wb[8], w[0] / w[8])  # class-balanced nhẹ hơn 1/n

    def test_cutmix_lambda_is_true_area(self):
        x = torch.zeros(8, 3, 32, 32)
        x[:4] = 1.0  # nửa batch toàn 1, nửa toàn 0 -> đo được diện tích bị dán
        y = torch.arange(8)
        for s in range(30):
            rng = np.random.default_rng(s)
            torch.manual_seed(s)
            xm, (ya, yb, lam) = L.mix_batch(x, y, 1.0, "cutmix", rng)
            self.assertTrue(torch.equal(ya, y))
            for i in range(8):
                src = int(yb[i])
                if (x[i, 0, 0, 0] != x[src, 0, 0, 0]):
                    pasted = (xm[i, 0] != x[i, 0, 0, 0]).float().mean().item()
                    self.assertAlmostEqual(1 - pasted, lam, places=6)

    def test_mixup(self):
        x = torch.rand(6, 3, 8, 8)
        y = torch.arange(6)
        torch.manual_seed(1)
        xm, (ya, yb, lam) = L.mix_batch(x, y, 0.4, "mixup", np.random.default_rng(1))
        self.assertTrue(torch.allclose(xm, lam * x + (1 - lam) * x[yb]))

    def test_mixed_loss(self):
        z = torch.randn(4, 9)
        ya, yb = torch.tensor([0, 1, 2, 3]), torch.tensor([3, 2, 1, 0])
        ce = nn.CrossEntropyLoss()
        ref = 0.3 * ce(z, ya) + 0.7 * ce(z, yb)
        self.assertLess(abs(L.mixed_loss(ce, z, (ya, yb, 0.3)) - ref).item(), 1e-7)
        self.assertLess(abs(L.mixed_loss(ce, z, ya) - ce(z, ya)).item(), 1e-7)


class TestModel(unittest.TestCase):
    def test_param_groups_no_decay_on_norm_bias(self):
        m = tiny_timm()
        groups = M.param_groups(m, 1e-4, 1e-3, 0.05)
        names = {id(p): n for n, p in m.named_parameters()}
        for g in groups:
            for p in g["params"]:
                n = names[id(p)]
                is_head = n.startswith("fc.")
                self.assertEqual(g["lr"], 1e-3 if is_head else 1e-4, n)
                if p.ndim <= 1:
                    self.assertEqual(g["weight_decay"], 0.0, n)
                else:
                    self.assertEqual(g["weight_decay"], 0.05, n)
        self.assertEqual(sum(len(g["params"]) for g in groups), len(list(m.parameters())))

    def test_vit_pos_embed_no_decay(self):
        m = tiny_timm("vit_tiny_patch16_224")
        groups = M.param_groups(m, 1e-4, 1e-3, 0.05)
        nd = [p for g in groups if g["weight_decay"] == 0 for p in g["params"]]
        self.assertTrue(any(p is m.pos_embed for p in nd))

    def test_freeze_keeps_bn_eval(self):
        m = tiny_timm()
        M.freeze_backbone(m)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        self.assertTrue(all(n.startswith("fc.") for n in trainable) and trainable)
        M.set_train_mode(m)
        bn = [x for x in m.modules() if isinstance(x, nn.BatchNorm2d)]
        self.assertTrue(all(not b.training for b in bn))
        rm = bn[0].running_mean.clone()
        m(torch.randn(4, 3, 64, 64))
        self.assertTrue(torch.equal(rm, bn[0].running_mean))
        self.assertTrue(m.fc.training)

    def test_count(self):
        m = tiny_timm()
        self.assertAlmostEqual(M.count_params(m), 11.18, delta=0.05)
        self.assertAlmostEqual(M.count_gmacs(m, 224), 1.82, delta=0.1)  # ResNet-18 ~1,8 GMAC


class TestTrainParts(unittest.TestCase):
    def test_lr_schedule(self):
        total, warm = 1000, 100
        f = [TR.lr_factor(s, total, warm) for s in range(total)]
        self.assertTrue(all(a < b for a, b in zip(f[:warm - 1], f[1:warm])))
        self.assertAlmostEqual(f[warm - 1], 1.0)
        self.assertAlmostEqual(f[warm], 1.0)
        self.assertTrue(all(a >= b for a, b in zip(f[warm:], f[warm + 1:])))
        self.assertLess(f[-1], 1e-4)

    def test_scheduler_with_param_groups(self):
        m = tiny_timm()
        cfg = TR.Config(epochs=2, warmup_epochs=0.5)
        opt = TR.build_optimizer(m, cfg)
        sch = TR.build_scheduler(opt, cfg, steps_per_epoch=10)
        lrs = []
        for _ in range(20):
            opt.step()
            sch.step()
            lrs.append([g["lr"] for g in opt.param_groups])
        lrs = np.array(lrs)
        self.assertAlmostEqual(lrs[3, 0], 1e-4)          # hết warmup sau 5 bước
        self.assertAlmostEqual(lrs[3, -1] / lrs[3, 0], 10.0)  # head gấp 10
        self.assertLess(lrs[-1, 0], 1e-6)

    def test_ema(self):
        lin = nn.Linear(3, 2)
        ema = TR.EMA(lin, 0.5)
        with torch.no_grad():
            w0 = lin.weight.clone()
            lin.weight.add_(1.0)
        ema.update(lin)  # d = min(0.5, 2/11)
        d = 2 / 11
        self.assertTrue(torch.allclose(ema.module.weight, d * w0 + (1 - d) * lin.weight))

    def test_parse_overrides(self):
        o = TR.parse_overrides(["seed=1", "loss=focal", "ema_decay=none", "amp=false", "lr_head=3e-4",
                                "sampler=balanced", "img_size=256"])
        self.assertEqual(o, {"seed": 1, "loss": "focal", "ema_decay": None, "amp": False, "lr_head": 3e-4,
                             "sampler": "balanced", "img_size": 256})
        with self.assertRaises(KeyError):
            TR.parse_overrides(["nope=1"])
        self.assertEqual(TR.pred_path(TR.Config(exp_id="F01", seed=2, pred_dir="p"), "test"),
                         Path("p/F01_seed2_test.csv"))

    def test_evaluate_and_save_predictions(self):
        m = tiny_timm().eval()
        ds = torch.utils.data.TensorDataset(torch.randn(10, 3, 32, 32), torch.randint(0, 9, (10,)))
        loader = [(x, y, [f"img{i}_{j}.jpg" for j in range(len(y))]) for i, (x, y) in
                  enumerate(torch.utils.data.DataLoader(ds, batch_size=4))]
        names, y, logits, loss = TR.evaluate(m, loader, nn.CrossEntropyLoss(), torch.device("cpu"), amp=False)
        self.assertEqual(logits.shape, (10, 9))
        with tempfile.TemporaryDirectory() as d:
            p = save_predictions(Path(d) / "X_seed0_val.csv", names, y, TR._softmax(logits.astype(np.float64)))
            self.assertEqual(read_pred(str(p)).seed, 0)


class TestInference(unittest.TestCase):
    def test_fuse_bn_resnet_and_efficientnet(self):
        for name in ("resnet18", "efficientnet_b0", "mobilenetv3_large_100"):
            m = tiny_timm(name)
            # BN có thống kê không tầm thường
            m.train()
            with torch.no_grad():
                for _ in range(3):
                    m(torch.randn(8, 3, 64, 64))
            m.eval()
            fused = INF.fuse_conv_bn(m, img_size=64)
            self.assertGreater(fused.fuse_info["n_fused"], 10, name)
            x = torch.randn(2, 3, 64, 64)
            with torch.no_grad():
                d = (m(x) - fused(x)).abs().max().item()
            self.assertLess(d, 1e-4, name)
            self.assertFalse(any(type(b) is nn.BatchNorm2d for b in fused.modules()), name)

    def test_fuse_bn_noop_without_bn(self):
        m = tiny_timm("convnext_atto")
        self.assertEqual(INF.fuse_conv_bn(m, img_size=64).fuse_info["n_fused"], 0)

    def test_views(self):
        x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).view(2, 3, 8, 8)
        self.assertTrue(torch.equal(INF.view_hflip(x)[..., 0], x[..., -1]))
        v5 = INF.views_multicrop(x, 6)
        self.assertEqual(len(v5), 5)
        self.assertTrue(all(v.shape == (2, 3, 6, 6) for v in v5))
        self.assertTrue(torch.equal(v5[4], x[..., 1:7, 1:7]))
        self.assertEqual(len(INF.views_multicrop(x, 6, flip=True)), 10)
        self.assertEqual([tuple(v.shape[-2:]) for v in INF.views_multiscale(x, [4, 8, 12])], [(4, 4), (8, 8), (12, 12)])

    def test_aggregate(self):
        rng = np.random.default_rng(0)
        ls = [rng.normal(size=(5, 9)) for _ in range(3)]
        for sp in ("prob", "logit"):
            p = INF.aggregate_views(ls, sp)
            self.assertTrue(np.allclose(p.sum(1), 1))
        self.assertTrue(np.allclose(INF.aggregate_views(ls[:1], "prob"), INF.aggregate_views(ls[:1], "logit")))
        p = INF.aggregate_views(ls, "prob")
        self.assertTrue(np.allclose(INF.apply_temperature(INF.probs_to_logits(p), 1.0), p))
        self.assertTrue(np.allclose(INF.ensemble_probs([p, p]), p))

    def test_temperature_recovers_T(self):
        rng = np.random.default_rng(0)
        n, true_T = 20000, 2.5
        z = rng.normal(size=(n, 9)) * 4
        p = INF.apply_temperature(z, true_T)
        y = np.array([rng.choice(9, p=row) for row in p])
        T = INF.fit_temperature(z, y)
        self.assertAlmostEqual(T, true_T, delta=0.1)
        self.assertTrue((INF.apply_temperature(z, T).argmax(1) == z.argmax(1)).all())

    def test_soup(self):
        a, b = nn.BatchNorm2d(3), nn.BatchNorm2d(3)
        with torch.no_grad():
            a.weight.fill_(1)
            b.weight.fill_(3)
        s = INF.model_soup([a.state_dict(), b.state_dict()])
        self.assertTrue(torch.allclose(s["weight"], torch.full((3,), 2.0)))


class TestBenchmarkAndData(unittest.TestCase):
    def test_bench(self):
        r = BM.bench(lambda: sum(range(1000)), warmup=10, iters=50)
        self.assertTrue(r["p50"] <= r["p95"] <= r["p99"])
        self.assertEqual(r["n"], 50)
        with self.assertRaises(ValueError):
            BM.bench(lambda: None, iters=10)

    def test_latency_report_cpu(self):
        r = BM.latency_report(tiny_timm(), 1, 64, "fp32", "cpu", warmup=10, iters=50)
        self.assertEqual(r["batch"], 1)
        self.assertGreater(r["images_per_s"], 0)

    def test_multi_model_latency_per_model_size(self):
        # Swin/ViT cố định kích thước: mỗi model phải nhận đúng kích thước nó được train (lỗi thật ở Bước 3)
        a = tiny_timm("resnet18")
        b = tiny_timm("vit_tiny_patch16_224", img_size=32)
        r = BM.multi_model_latency([a, b], [64, 32], "fp32", "cpu", warmup=10, iters=50)
        self.assertEqual(r["k_models"], 2)
        self.assertEqual(r["img_size"], "64/32")
        with self.assertRaises(AssertionError):
            BM.multi_model_latency([b], 64, "fp32", "cpu", warmup=10, iters=50)

    def test_transforms(self):
        img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        for aug in D.AUG_CHOICES:
            out = D.build_transforms(True, 224, aug)(img)
            self.assertEqual(out.shape, (3, 224, 224))
            self.assertEqual(out.dtype, torch.float32)
        ev = D.build_transforms(False, 224)(img)
        ref = (img[:, 16:240, 16:240].float() / 255 - torch.tensor(D.IMAGENET_MEAN).view(3, 1, 1)) \
            / torch.tensor(D.IMAGENET_STD).view(3, 1, 1)
        self.assertTrue(torch.allclose(ev, ref, atol=1e-5))  # center-crop 224 từ ảnh 256, không resize
        self.assertEqual(D.eval_transform(288)(img).shape, (3, 288, 288))
        self.assertTrue(torch.allclose(D.denormalize(ev[None])[0], img[:, 16:240, 16:240].float() / 255, atol=1e-5))

    def test_check_split_detects_overlap(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            tr = pd.DataFrame({"Filename": ["a.jpg", "b.jpg"], "Label": [0, 1]})
            va = pd.DataFrame({"Filename": ["b.jpg"], "Label": [1]})
            with self.assertRaises(AssertionError):
                D.check_split(tr, va, va, d, verbose=False)


if __name__ == "__main__":
    unittest.main()
