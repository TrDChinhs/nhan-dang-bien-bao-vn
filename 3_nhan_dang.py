"""Bước 3: đưa ẢNH CỦA BẠN vào, mô hình trả lời đó là biển gì, kèm ảnh của từng bước.

Cách dùng:
  1. Bỏ ảnh (png/jpg) vào thư mục vn_images/ . Ảnh nên là MỘT biển, biển chiếm gần hết khung ảnh.
  2. Chạy:   python 3_nhan_dang.py                      (xử lý mọi ảnh trong vn_images/)
             python 3_nhan_dang.py duong/dan/anh.jpg      (một ảnh, hoặc một thư mục khác)
Muốn đo độ chính xác: tạo thư mục con trùng mã lớp, ví dụ vn_images/W242a/anh1.jpg, vn_images/P102/anh2.jpg
  (tên thư mục = tên file trong bo_anh_goc, bỏ .png); chương trình sẽ so với đáp án và tổng kết đúng/sai.
Kết quả: in ra màn hình + ket_qua/nhan_dang/<tên ảnh>/ gồm 6 ảnh, mỗi ảnh là một bước:
  1_anh_dau_vao  2_vuong_48x48  3_khoi1  4_khoi2  5_khoi3  6_ket_qua
Lưu ý: bản này CHỈ phân loại ảnh đã cắt sẵn biển; chưa có bước tìm biển trong ảnh phố.
"""
import argparse
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import bb_chung as bb

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def chu(size):
    for n in ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


def tieu_de(img, text, size=16, pad=34):
    """Thêm dải chữ phía trên một ảnh PIL (xuống dòng theo bề ngang)."""
    lines = textwrap.wrap(text, max(20, img.width // (size // 2 + 2)))
    h = 8 + len(lines) * (size + 4)
    out = Image.new("RGB", (max(img.width, 360), img.height + h), "white")
    d = ImageDraw.Draw(out)
    for i, ln in enumerate(lines):
        d.text((6, 4 + i * (size + 4)), ln, fill=(160, 0, 0), font=chu(size))
    out.paste(img, (0, h))
    return out


def luoi_dac_trung(a, n=16, cell=96):
    """a: [C,H,W] sau ReLU -> lưới 4x4 gồm n kênh mạnh nhất (mỗi kênh chuẩn hoá riêng 0..255)."""
    order = np.argsort(-a.reshape(a.shape[0], -1).mean(1))[:n]
    side = int(np.ceil(np.sqrt(n)))
    g = 3
    canvas = np.full((side * (cell + g) + g, side * (cell + g) + g), 255, np.uint8)
    for j, c in enumerate(order):
        m = a[c] / max(float(a[c].max()), 1e-6)
        m = cv2.resize((m * 255).astype(np.uint8), (cell, cell), interpolation=cv2.INTER_NEAREST)
        y, x = (j // side) * (cell + g) + g, (j % side) * (cell + g) + g
        canvas[y:y + cell, x:x + cell] = m
    return Image.fromarray(canvas).convert("RGB")


def ten_lop(k, info):
    return ("unknown", "KHÔNG phải biển trong 295 lớp (hoặc mô hình không nhận ra)") if k == bb.UNKNOWN else (info[k]["ma"], info[k]["ten"])


def anh_ket_qua(sq, top, info, expected):
    W, H = 920, 470
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    d.text((10, 8), "Kết quả: xác suất của các lớp mô hình cho là giống nhất", fill=(160, 0, 0), font=chu(19))
    th = Image.fromarray(sq).resize((180, 180), Image.LANCZOS)
    out.paste(th, (10, 44))
    d.text((10, 228), "ảnh của bạn (hình vuông)", fill="black", font=chu(13))
    y = 46
    for i, (k, p) in enumerate(top):
        ma, ten = ten_lop(k, info)
        mau = (0, 150, 70) if i == 0 else (120, 120, 120)
        d.rectangle([215, y + 20, 215 + int(560 * p), y + 38], fill=mau)
        d.text((215, y), f"{ma}  {textwrap.shorten(ten, 58, placeholder='...')}", fill="black", font=chu(15))
        d.text((220 + int(560 * p), y + 19), f"{100 * p:.1f}%", fill="black", font=chu(15))
        y += 52
    d.text((10, 312), "Ảnh gốc của 3 lớp đứng đầu (để bạn đối chiếu bằng mắt):", fill=(160, 0, 0), font=chu(15))
    for i, (k, p) in enumerate(top[:3]):
        x = 10 + i * 300
        if k == bb.UNKNOWN:
            d.text((x, 345), "(unknown: không có ảnh gốc)", fill="black", font=chu(14))
            continue
        ref = Image.open(info[k]["anh"]).convert("RGB")
        ref.thumbnail((110, 110))
        out.paste(ref, (x, 340))
        d.text((x + 120, 350), f"{ma_ngan(k, info)}", fill="black", font=chu(15))
        d.text((x + 120, 372), f"{100 * p:.1f}%", fill="black", font=chu(15))
    if expected:
        d.text((10, 448), f"Đáp án theo tên thư mục: {expected}", fill=(0, 0, 160), font=chu(15))
    return out


def ma_ngan(k, info):
    return info[k]["ma"]


def xu_ly(path, model, keys, info, a, expected):
    img = bb.doc_anh(path)
    sq, small = bb.tien_xu_ly(img)
    x = bb.sang_tensor(small[None])
    with torch.no_grad():
        logits, maps = model(x, giua=True)
    pr = torch.softmax(logits, 1)[0].numpy()
    order = np.argsort(-pr)[:max(a.top, 3)]
    top = [(keys[i], float(pr[i])) for i in order]
    if not a.khong_anh:
        out = bb.KET_QUA / "nhan_dang" / path.stem
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.png"):
            f.unlink()
        im = Image.fromarray(img)
        s = 420 / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        tieu_de(im, f"Bước 1. Ảnh đầu vào: {img.shape[1]}x{img.shape[0]} px. Mô hình KHÔNG đọc ảnh này trực tiếp.").save(out / "1_anh_dau_vao.png")
        pair = Image.new("RGB", (200 + 20 + 288, 288), "white")
        pair.paste(Image.fromarray(sq).resize((200, 200), Image.LANCZOS), (0, 0))
        pair.paste(Image.fromarray(small).resize((288, 288), Image.NEAREST), (220, 0))
        tieu_de(pair, f"Bước 2. Thêm viền cho thành hình vuông ({sq.shape[1]}x{sq.shape[0]}) rồi thu về {bb.IMG}x{bb.IMG} = {bb.IMG * bb.IMG} điểm ảnh "
                      f"(bên phải: đúng {bb.IMG}x{bb.IMG} điểm đó, phóng 6 lần). Đây là toàn bộ thứ mô hình nhìn thấy.", 15).save(out / "2_vuong_48x48.png")
        mo_ta = [("3_khoi1", "Bước 3. Khối 1: 32 'bản đồ' 48x48. Mỗi ô là kết quả một bộ lọc 3x3 trượt khắp ảnh; sáng = bộ lọc thấy đúng mẫu nó tìm "
                              "(cạnh, góc, mảng màu). Hiện 16 bản mạnh nhất."),
                 ("4_khoi2", "Bước 4. Khối 2: 64 bản đồ 24x24. Ghép các cạnh/góc ở khối 1 thành hình đơn giản (đường cong, góc nhọn, vòng)."),
                 ("5_khoi3", "Bước 5. Khối 3: 128 bản đồ 12x12. Ghép tiếp thành bộ phận của biển (viền tam giác, vạch ngang, chữ số...). "
                             "Sau đó 128x6x6 = 4608 số được cộng-nhân thành 296 điểm.")]
        for (name, text), m in zip(mo_ta, maps):
            tieu_de(luoi_dac_trung(m[0].numpy()), text, 15).save(out / f"{name}.png")
        tieu_de(anh_ket_qua(sq, top[:a.top], info, expected), "Bước 6. 296 điểm -> softmax -> xác suất. Lớp cao nhất là câu trả lời.", 16).save(out / "6_ket_qua.png")
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("duong_dan", nargs="*", default=[str(bb.ROOT / "vn_images")])
    ap.add_argument("--model", default=str(bb.KET_QUA / "model.pt"))
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--khong-anh", action="store_true", help="chỉ in kết quả, không lưu ảnh từng bước")
    a = ap.parse_args()
    if not Path(a.model).exists():
        sys.exit(f"Chưa có mô hình {a.model}. Chạy trước:  python 2_train.py")
    model, keys, ck = bb.nap_model(a.model)
    _, info, _ = bb.doc_danh_sach()
    files = []
    for p in map(Path, a.duong_dan):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXT) if p.is_dir() else [p]
    if not files:
        sys.exit("Không thấy ảnh nào. Bỏ ảnh png/jpg vào thư mục vn_images/ rồi chạy lại.")
    print(f"Mô hình: {a.model} (đã train {ck.get('epochs', '?')} epoch, {len(keys) - 1} lớp + unknown)\n")
    n_exp = ok1 = ok3 = 0
    for f in files:
        exp = f.parent.name if f.parent.name in keys else None
        top = xu_ly(f, model, keys, info, a, exp)
        k, p = top[0]
        ma, ten = ten_lop(k, info)
        chac = "chắc" if p >= 0.8 else "KHÔNG chắc lắm" if p >= 0.5 else "RẤT không chắc (đừng tin)"
        print(f"{f.name}")
        print(f"   -> {ma}  {ten}   ({100 * p:.1f}%, {chac})")
        print("      " + " | ".join(f"{ten_lop(kk, info)[0]} {100 * pp:.1f}%" for kk, pp in top[:a.top]))
        if exp:
            n_exp += 1
            ok1 += k == exp
            ok3 += exp in [kk for kk, _ in top[:3]]
            print(f"      đáp án {exp}: {'ĐÚNG' if k == exp else 'SAI'}")
    if n_exp:
        print(f"\nTổng kết ({n_exp} ảnh có đáp án theo tên thư mục): đúng top-1 {ok1}/{n_exp}, có trong top-3 {ok3}/{n_exp}")
    if not a.khong_anh:
        print(f"\nẢnh từng bước: {bb.KET_QUA / 'nhan_dang'}")


if __name__ == "__main__":
    main()
