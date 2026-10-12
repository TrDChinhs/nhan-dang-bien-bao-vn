"""Phần dùng chung của 3 file: 1_xem_bien_the.py, 2_train.py, 3_nhan_dang.py.

Luồng cả dự án (mỗi bước một file):
    bo_anh_goc/ (mỗi lớp 1 ảnh sạch)  --[làm biến thể: nghiêng, nền, sáng-tối, mờ, nhiễu...]-->  ảnh 48x48 có nhãn
    --[2_train.py: CNN học "ảnh -> lớp"]-->  ket_qua/model.pt  --[3_nhan_dang.py: ảnh của bạn -> lớp]-->  kết quả

Ở đây có: đọc danh sách lớp, làm biến thể ảnh (render), mô hình CNN, tiền xử lý ảnh khi nhận dạng.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parent
GOC = ROOT / "bo_anh_goc"                              # ảnh gốc (nền trắng) + danh_sach.csv
MAU = ROOT / "data" / "vn_signs" / "templates"         # chính các bản vẽ đó nhưng nền trong suốt (để ghép lên nền khác)
NEN_DUC = ROOT / "data" / "vn_signs" / "bg_pool.npy"   # 8000 mảnh nền 96x96 cắt từ ảnh phố
CANH_VN = ROOT / "vn_web" / "scenes"                   # ảnh phố Việt Nam
KET_QUA = ROOT / "ket_qua"
UNKNOWN = "unknown"
IMG = 48           # cỡ ảnh mô hình nhìn thấy
S = 96             # vẽ ở 96 rồi thu về 48 (mượt viền hơn)


# ----------------------------------------------------------------------------- danh sách lớp
def doc_danh_sach():
    """-> (keys, info, ngoai). keys[0] = 'unknown', keys[1:] = 295 lớp; ngoai = khoá các biển không làm lớp riêng
    (chỉ dùng làm ví dụ 'unknown'); info[khoá] = {ma, ten, anh (ảnh gốc nền trắng), ban_ve (PNG nền trong suốt)}."""
    rows = list(csv.reader(open(GOC / "danh_sach.csv", encoding="utf-8-sig")))[1:]
    keys, info, ngoai = [UNKNOWN], {}, []
    for nhom, ma, ten, anh, _, ban_ve, *_ in rows:
        key = Path(anh).stem
        (ngoai if "không làm lớp riêng" in nhom else keys).append(key)
        info[key] = {"ma": ma, "ten": ten, "anh": GOC / anh, "ban_ve": MAU / ban_ve}
    return keys, info, ngoai


_cache = {}


def doc_bien(path):
    """PNG nền trong suốt -> mảng RGBA đã cắt sát biển."""
    p = str(path)
    if p not in _cache:
        a = np.asarray(Image.open(path).convert("RGBA"))
        ys, xs = np.where(a[..., 3] > 10)
        _cache[p] = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return _cache[p]


# ----------------------------------------------------------------------------- nền
def nguon_nen():
    """Nạp (một lần mỗi tiến trình) các nguồn nền: mảnh nền phố đã cắt sẵn + ảnh phố Việt Nam (thu nhỏ)."""
    if "nen" not in _cache:
        pool = np.load(NEN_DUC, mmap_mode="r") if NEN_DUC.exists() else None
        canh = []
        for p in sorted(CANH_VN.glob("*.jpg")):
            im = Image.open(p)
            im.draft("RGB", (1200, 1200))
            im = im.convert("RGB")
            im.thumbnail((1200, 1200))
            canh.append(np.asarray(im))
        _cache["nen"] = (pool, canh)
    return _cache["nen"]


def nen_ngau_nhien(rng):
    pool, canh = nguon_nen()
    r = rng.random()
    if r < 0.12 or (pool is None and not canh):           # đồng màu / chuyển sắc (trời, tường, mặt đường)
        c1, c2 = rng.integers(20, 235, 3), rng.integers(20, 235, 3)
        t = np.linspace(0, 1, S)[:, None, None] if rng.random() < 0.5 else np.linspace(0, 1, S)[None, :, None]
        return ((c1 * (1 - t) + c2 * t) * np.ones((S, S, 1))).astype(np.float32)
    if canh and (pool is None or r < 0.5):                # mảnh cắt từ ảnh phố Việt Nam
        im = canh[rng.integers(len(canh))]
        h, w = im.shape[:2]
        c = int(np.exp(rng.uniform(np.log(48), np.log(min(h, w) * 0.6))))
        y, x = rng.integers(0, h - c), rng.integers(0, w - c)
        bg = cv2.resize(im[y:y + c, x:x + c], (S, S), interpolation=cv2.INTER_AREA)
    else:
        bg = np.array(pool[rng.integers(len(pool))])
    if rng.random() < 0.5:
        bg = bg[:, ::-1]                                  # lật NỀN thì được; lật BIỂN thì không (rẽ trái <-> rẽ phải)
    return np.ascontiguousarray(bg).astype(np.float32)


# ----------------------------------------------------------------------------- các bước làm biến thể
def dat_bien(rgba, rng, rot=10.0, persp=0.12):
    """Thu/phóng, xoay, méo phối cảnh (như chụp lệch góc, xa gần) rồi đặt vào khung SxS. Trả (rgb nhân alpha, alpha)."""
    h, w = rgba.shape[:2]
    sc = S * rng.uniform(0.55, 0.95) / max(h, w)
    nh, nw = max(4, round(h * sc)), max(4, round(w * sc))
    t = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    a = t[..., 3:4] / 255.0
    pm = np.concatenate([t[..., :3] * a, a * 255.0], 2)
    src = np.float32([[0, 0], [nw, 0], [nw, nh], [0, nh]])
    corners = src - np.float32([nw / 2, nh / 2])
    corners += rng.uniform(-1, 1, (4, 2)) * rng.uniform(0, persp) * np.float32([nw, nh])
    th = np.deg2rad(rng.uniform(-rot, rot))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    dst = corners @ R.T + np.float32([S / 2, S / 2]) + rng.uniform(-0.07, 0.07, 2) * S
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    out = cv2.warpPerspective(pm, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out[..., :3], out[..., 3:4] / 255.0


def mo_chuyen_dong(img, rng):
    k = int(rng.integers(3, 10))
    ker = np.zeros((k, k), np.float32)
    cv2.line(ker, (0, k // 2), (k - 1, k // 2), 1.0, 1)
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
    ker = cv2.warpAffine(ker, M, (k, k))
    return cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))


def render(rgba, rng, m=1.0):
    """Một ảnh biến thể 48x48 từ hình mẫu sạch rgba (None = chỉ có nền, không biển). m (0..1) = độ mạnh của 'nhiễu'.
    Thứ tự: nền -> cột -> đặt biển (nghiêng/xa gần) -> sáng-tối, phai màu -> bóng đổ -> che khuất -> mờ -> giảm độ phân giải -> nhiễu -> nén JPEG."""
    bg = nen_ngau_nhien(rng) * rng.uniform(0.7, 1.2)
    if rng.random() < 0.3:                                              # cột đỡ biển
        x, w, g = S // 2 + int(rng.integers(-6, 7)), int(rng.integers(3, 8)), int(rng.integers(90, 200))
        cv2.rectangle(bg, (x - w // 2, S // 2), (x + w // 2, S), (g, g, g), -1)
    if rgba is None:
        img = bg
    else:
        rgb, alpha = dat_bien(rgba, rng, rot=10 * (0.3 + 0.7 * m), persp=0.12 * (0.3 + 0.7 * m))
        rgb = rgb * rng.uniform(1 - 0.4 * m, 1 + 0.2 * m) * (1 + rng.uniform(-0.08, 0.08, 3) * m)        # sáng/tối, lệch màu
        lum = rgb.mean(2, keepdims=True)
        rgb = lum + (rgb - lum) * rng.uniform(1 - 0.45 * m, 1.1)                                          # phai màu
        if rng.random() < 0.3 * m:
            f = rng.uniform(0.1, 0.45)
            rgb = rgb * (1 - f) + alpha * rng.uniform(100, 190) * f                                       # bám bụi, nhạt
        img = rgb + (1 - alpha) * bg
    if rng.random() < 0.5 * m:                                          # ánh sáng một bên
        ramp = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(1.0, 1.3), S)
        img = img * (ramp[None, :, None] if rng.random() < 0.5 else ramp[:, None, None])
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.25 * m:                                         # bị che một phần (cành cây, nhãn dán, xe)
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x0, y0 = int(rng.integers(0, S - 10)), int(rng.integers(0, S - 10))
        x1, y1 = min(S, x0 + int(rng.integers(8, S // 3))), min(S, y0 + int(rng.integers(8, S // 3)))
        cv2.rectangle(img, (x0, y0), (x1, y1), c, -1) if rng.random() < 0.5 else cv2.ellipse(img, ((x0 + x1) // 2, (y0 + y1) // 2), ((x1 - x0) // 2 + 1, (y1 - y0) // 2 + 1), 0, 0, 360, c, -1)
    if rng.random() < 0.7 * m:                                          # mờ (lấy nét kém / rung)
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 0.5 + 1.8 * m)) if rng.random() < 0.6 else mo_chuyen_dong(img, rng)
    if rng.random() < 0.5 * m:                                          # camera độ phân giải thấp / biển ở xa
        f = rng.uniform(0.3, 0.8)
        img = cv2.resize(cv2.resize(img, (max(8, int(S * f)),) * 2, interpolation=cv2.INTER_AREA), (S, S), interpolation=cv2.INTER_LINEAR)
    if m > 0:
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(0, 10 * m), img.shape), 0, 255).astype(np.uint8)   # nhiễu hạt
    if rng.random() < 0.5 * m:                                          # nén JPEG
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img
    return cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)


def hinh_trong(rng):
    """Biển 'trống' (đúng khung, đúng màu, KHÔNG có ký hiệu) -> RGBA. Dùng làm ví dụ 'unknown':
    mô hình phải học rằng khung giống biển chưa đủ, phải có ký hiệu đúng."""
    rgb = np.zeros((S, S, 3), np.uint8)
    al = np.zeros((S, S), np.uint8)

    def poly(pts, col):
        p = np.array(pts, np.int32)
        cv2.fillPoly(rgb, [p], col)
        cv2.fillPoly(al, [p], 255)

    def disc(r, col):
        cv2.circle(rgb, (S // 2, S // 2), r, col, -1)
        cv2.circle(al, (S // 2, S // 2), r, 255, -1)

    red, blue, yellow, white = (205, 25, 35), (0, 75, 165), (250, 200, 0), (255, 255, 255)
    inner = white if rng.random() < 0.7 else yellow
    k = rng.integers(0, 6)
    if k == 0:                                                        # tam giác viền đỏ
        tri = np.float32([[48, 5], [91, 84], [5, 84]])
        poly(tri, red)
        poly((tri - tri.mean(0)) * 0.68 + tri.mean(0) + [0, 2], inner)
    elif k == 1:                                                      # tròn viền đỏ
        disc(46, red)
        disc(int(rng.integers(30, 36)), inner)
    elif k == 2:                                                      # tròn xanh đặc
        disc(46, blue)
        if rng.random() < 0.5:
            cv2.circle(rgb, (S // 2, S // 2), 40, white, 2)
    elif k == 3:                                                      # bát giác đỏ (kiểu STOP)
        c = np.array([48, 48]) + 46 * np.stack([np.cos(np.deg2rad(22.5 + 45 * np.arange(8))), np.sin(np.deg2rad(22.5 + 45 * np.arange(8)))], 1)
        poly(c, red)
    elif k == 4:                                                      # hình thoi
        col = yellow if rng.random() < 0.6 else blue
        poly([[48, 3], [93, 48], [48, 93], [3, 48]], col)
        poly([[48, 14], [82, 48], [48, 82], [14, 48]], inner if col == yellow else blue)
    else:                                                             # chữ nhật xanh/lục có viền trắng
        col = blue if rng.random() < 0.6 else (0, 120, 70)
        poly([[6, 14], [90, 14], [90, 82], [6, 82]], col)
        cv2.rectangle(rgb, (11, 19), (85, 77), white, 2)
    return np.dstack([rgb, al])


def sinh_lo(args):
    """Làm một lô ảnh. jobs: [(nhãn, khoá hoặc '#nen' | '#trong' | khoá biển ngoài danh sách)]. Chạy trong tiến trình con."""
    jobs, seed, m = args
    rng = np.random.default_rng(seed)
    keys, info, ngoai = doc_danh_sach()
    xs, ys = [], []
    for label, key in jobs:
        if key == "#nen":
            xs.append(render(None, rng, m))
        elif key == "#trong":
            xs.append(render(hinh_trong(rng), rng, m))
        else:
            xs.append(render(doc_bien(info[key]["ban_ve"]), rng, m))
        ys.append(label)
    return np.stack(xs), np.array(ys, np.int64)


def tao_viec(keys, ngoai, moi_lop, n_unknown, rng):
    """Danh sách việc cho 1 epoch: mỗi lớp `moi_lop` ảnh + `n_unknown` ảnh 'unknown' (40% chỉ nền, 40% biển trống, 20% biển ngoài danh sách)."""
    jobs = [(i, k) for i, k in enumerate(keys) if i > 0 for _ in range(moi_lop)]
    for j in range(n_unknown):
        r = j / max(n_unknown, 1)
        jobs.append((0, "#nen" if r < 0.4 else "#trong" if r < 0.8 else ngoai[j % len(ngoai)]))
    return [jobs[i] for i in rng.permutation(len(jobs))]


# ----------------------------------------------------------------------------- mô hình
class CNN(nn.Module):
    """3 khối [Conv3x3 -> BN -> ReLU] x2 -> MaxPool: 48 -> 24 -> 12 -> 6, kênh 32/64/128; rồi 4608 -> 256 -> n lớp."""

    def __init__(self, n_cls):
        super().__init__()

        def khoi(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.MaxPool2d(2), nn.Dropout(0.25))
        self.khoi = nn.ModuleList([khoi(3, 32), khoi(32, 64), khoi(64, 128)])
        self.dau = nn.Sequential(nn.Flatten(), nn.Linear(128 * 6 * 6, 256), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(256, n_cls))

    def forward(self, x, giua=False):
        maps = []
        for k in self.khoi:
            a = k[:6](x)                      # đến ReLU thứ hai (trước MaxPool): "bản đồ đặc trưng" của khối
            maps.append(a)
            x = k[6:](a)
        out = self.dau(x)
        return (out, maps) if giua else out


def sang_tensor(x):
    """uint8 [N,H,W,3] -> float [N,3,H,W] trong [-1,1]."""
    x = x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x))
    return (x.permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5


# ----------------------------------------------------------------------------- ảnh của người dùng
def doc_anh(path):
    """Đọc ảnh bất kỳ (png/jpg, có thể có nền trong suốt) -> RGB uint8."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        im = im.convert("RGBA")
        nen = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(nen, im)
    return np.asarray(im.convert("RGB"))


def vuong(img):
    """Thêm viền (lặp lại màu mép ảnh) để thành hình vuông, KHÔNG làm méo tỉ lệ biển."""
    h, w = img.shape[:2]
    d = abs(h - w)
    t, l = (d // 2, 0) if h < w else (0, d // 2)
    b, r = (d - d // 2, 0) if h < w else (0, d - d // 2)
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_REPLICATE) if d else img


def tien_xu_ly(img):
    """Ảnh người dùng -> (ảnh vuông, ảnh 48x48 mô hình nhìn thấy)."""
    sq = vuong(img)
    return sq, cv2.resize(sq, (IMG, IMG), interpolation=cv2.INTER_AREA)


def nap_model(path=None, device="cpu"):
    ck = torch.load(path or KET_QUA / "model.pt", map_location=device)
    model = CNN(len(ck["keys"])).to(device)
    model.load_state_dict(ck["model_state"])
    return model.eval(), ck["keys"], ck
