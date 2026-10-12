"""Bước 2: train mô hình nhận dạng biển từ bo_anh_goc.

Chạy:   python 2_train.py                       (15 epoch, ~100 ảnh biến thể mỗi lớp mỗi epoch)
        python 2_train.py --epoch 3 --moi-lop 10   (chạy thử nhanh, kiểm tra code)
Kết quả: ket_qua/model.pt (lưu sau mỗi epoch nên dừng giữa chừng vẫn dùng được), ket_qua/lich_su.json

Cách học (mỗi epoch):
  1. Từ 295 ảnh gốc làm ra ~30.000 ảnh biến thể MỚI (khác epoch trước) + ~3.000 ảnh 'unknown'. Việc này chạy ở các tiến trình
     con, song song với lúc mô hình đang học epoch hiện tại.
  2. Trước khi học trên lô mới, đo độ chính xác trên chính lô đó: mô hình CHƯA từng thấy các biến thể này nên đây là thước đo
     "gặp biến thể lạ thì nhận ra không" (vẫn là ảnh tổng hợp, chưa phải ảnh thật).
  3. Học: mô hình đoán -> so với nhãn đúng, tính độ sai (cross-entropy) -> chỉnh trọng số một chút (AdamW) -> lặp theo từng
     lô 128 ảnh. Tốc độ học tăng nhanh rồi giảm dần (OneCycle).
  4. 'Độ mạnh nhiễu' m tăng dần từ 0,3 lên 1 để lúc đầu dễ học hơn.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
from torch import nn

import bb_chung as bb

BATCH = 128


def main():
    cpu = os.cpu_count() or 4
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=15)
    ap.add_argument("--moi-lop", type=int, default=100, help="số ảnh biến thể mỗi lớp mỗi epoch")
    ap.add_argument("--unknown", type=int, default=3000, help="số ảnh 'unknown' mỗi epoch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=max(1, min(6, cpu // 3)), help="số tiến trình làm biến thể ảnh")
    ap.add_argument("--ten", default="model", help="tên file kết quả: ket_qua/<ten>.pt")
    a = ap.parse_args()

    keys, info, ngoai = bb.doc_danh_sach()
    bb.KET_QUA.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(2, cpu - a.workers))
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    n_per_epoch = (len(keys) - 1) * a.moi_lop + a.unknown
    steps = n_per_epoch // BATCH
    model = bb.CNN(len(keys)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr * 2, total_steps=a.epoch * steps, pct_start=0.15)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    print(f"{device} | {len(keys)} lớp (gồm unknown) | {n_per_epoch} ảnh/epoch | {a.epoch} epoch | "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} triệu tham số | {a.workers} tiến trình làm ảnh", flush=True)

    def muc_nhieu(e):
        return min(1.0, 0.3 + 0.7 * (e - 1) / max(1.0, 0.4 * a.epoch))

    def giao_viec(pool, e):
        jobs = bb.tao_viec(keys, ngoai, a.moi_lop, a.unknown, rng)
        n = a.workers * 4
        seed = 1000 * e
        return pool.map_async(bb.sinh_lo, [(jobs[k::n], seed + k, muc_nhieu(e)) for k in range(n)])

    history, t_all = [], time.time()
    with Pool(a.workers) as pool:
        pending = giao_viec(pool, 1)
        for e in range(1, a.epoch + 1):
            t0 = time.time()
            parts = pending.get()
            x = np.concatenate([p[0] for p in parts])
            y = np.concatenate([p[1] for p in parts])
            t_gen = time.time() - t0
            if e < a.epoch:
                pending = giao_viec(pool, e + 1)                      # làm sẵn lô cho epoch sau, chạy song song với việc học dưới đây
            xt, yt = torch.from_numpy(x), torch.from_numpy(y)

            model.eval()                                              # đo trên lô CHƯA học
            sel = torch.randperm(len(yt))[:4000]
            with torch.no_grad():
                pred = torch.cat([model(bb.sang_tensor(xt[sel[i:i + 500]]).to(device)).argmax(1).cpu() for i in range(0, len(sel), 500)])
            ok = pred == yt[sel]
            val, val_unk = ok.float().mean().item(), ok[yt[sel] == 0].float().mean().item()

            model.train()
            t1, losses = time.time(), []
            perm = torch.randperm(len(yt))
            for i in range(steps):
                idx = perm[i * BATCH:(i + 1) * BATCH]
                loss = crit(model(bb.sang_tensor(xt[idx]).to(device)), yt[idx].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                losses.append(loss.item())
            history.append({"epoch": e, "nhieu": muc_nhieu(e), "loss": float(np.mean(losses)), "val_chua_hoc": val, "val_unknown": val_unk})
            print(f"Epoch {e:2d}/{a.epoch} | nhiễu {muc_nhieu(e):.2f} | loss {np.mean(losses):.3f} | đúng trên lô chưa học {val:.3f} "
                  f"(unknown {val_unk:.3f}) | chờ làm ảnh {t_gen:.0f}s, học {time.time() - t1:.0f}s", flush=True)
            torch.save({"model_state": model.state_dict(), "keys": keys, "img_size": bb.IMG, "epochs": e}, bb.KET_QUA / f"{a.ten}.pt")
            (bb.KET_QUA / "lich_su.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"Xong sau {(time.time() - t_all) / 60:.1f} phút. Đã lưu {bb.KET_QUA / (a.ten + '.pt')}")


if __name__ == "__main__":
    main()
