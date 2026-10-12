"""Bước 1: xem "ảnh gốc" được làm thành các ảnh biến thể như thế nào (đây chính là dữ liệu để train).

Chạy:   python 1_xem_bien_the.py                      (mặc định vài biển mẫu, mỗi biển 10 biến thể)
        python 1_xem_bien_the.py P102 W207a R301a      (chọn mã khác: tên file trong bo_anh_goc, ví dụ P102 = Cấm đi ngược chiều)
Kết quả: ket_qua/xem_bien_the.png   (hàng 1: ảnh gốc 48x48; các hàng sau: biến thể; dưới cùng: các ví dụ 'unknown')
Mỗi ô là ảnh 48x48 phóng to 3 lần, đúng cái mô hình nhìn thấy khi train.
"""
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bb_chung as bb

N, ZOOM = 10, 3


def font(size):
    for n in ("arial.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


def phong(im):
    return cv2.resize(im, (bb.IMG * ZOOM,) * 2, interpolation=cv2.INTER_NEAREST)


def main():
    keys, info, ngoai = bb.doc_danh_sach()
    chon = [k for k in (sys.argv[1:] or ["P102", "P127_50", "W207a", "R301a", "I407a"]) if k in info]
    if not chon:
        sys.exit(f"Không có mã nào hợp lệ. Ví dụ mã đúng: {', '.join(keys[1:6])}")
    rng = np.random.default_rng(1)
    cell, head = bb.IMG * ZOOM, 26
    rows = []                                                  # (tiêu đề, [ảnh 48x48])
    for k in chon:
        rgba = bb.doc_bien(info[k]["ban_ve"])
        sach = bb.render(rgba, rng, m=0.0)                     # m=0: chỉ đặt biển lên nền, gần như không biến đổi thêm
        rows.append((f"{info[k]['ma']}  {info[k]['ten']}   (ô đầu: ít biến đổi; các ô sau: biến thể)",
                     [sach] + [bb.render(rgba, rng, m=1.0) for _ in range(N - 1)]))
    rows.append(("'unknown' 1: chỉ có nền (cây, nhà, đường...)", [bb.render(None, rng, 1.0) for _ in range(N)]))
    rows.append(("'unknown' 2: biển TRỐNG (đúng khung, không ký hiệu), giống ảnh tam giác bạn gửi", [bb.render(bb.hinh_trong(rng), rng, 1.0) for _ in range(N)]))
    rows.append(("'unknown' 3: biển có thật nhưng không thuộc 295 lớp", [bb.render(bb.doc_bien(info[ngoai[i]]["ban_ve"]), rng, 1.0) for i in range(N)]))

    sheet = Image.new("RGB", (N * cell, len(rows) * (cell + head)), "white")
    d = ImageDraw.Draw(sheet)
    for r, (title, ims) in enumerate(rows):
        y = r * (cell + head)
        d.text((4, y + 4), title, fill=(160, 0, 0), font=font(15))
        for c, im in enumerate(ims):
            sheet.paste(Image.fromarray(phong(im)), (c * cell, y + head))
    bb.KET_QUA.mkdir(exist_ok=True)
    out = bb.KET_QUA / "xem_bien_the.png"
    sheet.save(out)
    print("Đã lưu", out)
    print(f"{len(keys) - 1} lớp, {len(ngoai)} biển ngoài danh sách (chỉ làm ví dụ 'unknown').")


if __name__ == "__main__":
    main()
