# Nhận dạng biển báo Việt Nam: xây lại từ `bo_anh_goc`

## Luồng

```
bo_anh_goc/ (295 lớp, mỗi lớp 1 ảnh sạch)
   │  làm biến thể: nghiêng, xa/gần, nền phố, sáng-tối, phai màu, mờ, nhiễu hạt, JPEG, bị che, cột biển
   ▼
~32.000 ảnh 48x48 có nhãn mỗi epoch (mỗi epoch ảnh mới)  +  lớp 'unknown' (nền, biển trống, biển ngoài danh sách)
   │  train CNN nhỏ (1,5 triệu tham số)
   ▼
ket_qua/model.pt  ──►  ảnh của bạn (vn_images/)  ──►  tên biển + xác suất + ảnh từng bước
```

## Cài đặt (người mới tải về)

Cần Python 3.10 trở lên, rồi cài thư viện:

```
pip install -r requirements.txt
```

Chỉ để **nhận dạng** thì không cần train: mô hình đã train nằm sẵn trong `ket_qua/model.pt`. Bỏ ảnh vào `vn_images/` rồi chạy
`python 3_nhan_dang.py`. (Phần train, xem biến thể cần thêm `data/` và `vn_web/`; nền phố `bg_pool.npy` quá lớn nên không có trong kho.)

## Ba lệnh

Chạy trong thư mục `D:\TGMT`, môi trường `bienbao` (hoặc mở file trong Spyder và bấm F5):

| Bước | Lệnh | Làm gì | Thời gian (CPU) |
|---|---|---|---|
| 1 | `python 1_xem_bien_the.py` | xem ảnh gốc được làm biến thể thế nào → `ket_qua/xem_bien_the.png` | vài giây |
| 2 | `python 2_train.py` | train; lưu `ket_qua/model.pt` sau mỗi epoch | khoảng 30 phút |
| 3 | `python 3_nhan_dang.py` | nhận dạng mọi ảnh trong `vn_images/` → `ket_qua/nhan_dang/<tên ảnh>/` (6 ảnh, mỗi ảnh một bước) | vài giây |

Nhận dạng một ảnh riêng: `python 3_nhan_dang.py duong\dan\anh.jpg`.

## Đưa ảnh của bạn vào

- Bỏ ảnh `.png`/`.jpg` vào `vn_images/`. Ảnh nên là **một biển**, biển chiếm gần hết khung ảnh (bản này chưa có bước tìm biển trong ảnh phố).
- Muốn đo độ chính xác: bỏ ảnh vào thư mục con mang **mã lớp**, ví dụ `vn_images/W242a/anh1.jpg`. Mã lớp = tên file trong `bo_anh_goc/` bỏ `.png`
  (xem `bo_anh_goc/danh_sach.csv`). Chương trình so với đáp án và tổng kết đúng/sai.

## Các file

| File | Vai trò |
|---|---|
| `bb_chung.py` | đọc danh sách lớp, **làm biến thể ảnh**, mô hình CNN, tiền xử lý ảnh |
| `1_xem_bien_the.py`, `2_train.py`, `3_nhan_dang.py` | ba bước ở trên |
| `bo_anh_goc/` | 295 ảnh sạch + `danh_sach.csv` (mã, tên, **tác giả, giấy phép**) |
| `data/vn_signs/templates/` | cùng các bản vẽ đó nhưng nền trong suốt (để ghép lên nền khác) |
| `data/vn_signs/bg_pool.npy`, `vn_web/scenes/` | nền phố (Đức, và 10 ảnh Việt Nam) |
| `data/vn_signs/real/` | 103 ảnh phố thật có nhãn, **chỉ để kiểm tra**, không đưa vào train |

## Kết quả đo hiện tại (model 15 epoch, 12-10-2026)

| Phép đo | Kết quả |
|---|---|
| Ảnh tổng hợp **chưa học** (ngay trong lúc train) | 97,5% |
| Ảnh phố **thật** (78 ảnh có biển thuộc danh sách), trượt cửa sổ không detector, xác suất ≥ 50% | 21/78 ≈ 27% tìm thấy đúng lớp |
| Như trên, nhưng tính cả cửa sổ chắc 30–50% | tối đa khoảng 30/78 ≈ 38% |
| Ảnh tam giác trống của bạn | `unknown` 95,6% (đúng: không có ký hiệu thì không phải biển nào) |

Khoảng cách 97,5% → 27% là khoảng cách giữa "ảnh tổng hợp từ 1 bản vẽ" và "ảnh chụp thật". Cách thu hẹp (chưa làm): dùng nhiều bản vẽ mỗi lớp,
train dài hơn trên GPU (Kaggle), ResNet-18 tiền huấn luyện, và tinh chỉnh bằng ảnh thật (tách riêng tập kiểm tra).

## Giới hạn cần nhớ

- Mô hình chỉ thấy ảnh **tổng hợp** từ 1 bản vẽ mỗi lớp; độ chính xác trên ảnh thật **phải đo bằng ảnh thật** (cách đo ở trên), không suy ra từ số trong lúc train.
- 'Đúng trên lô chưa học' khi train là ảnh tổng hợp, không phải ảnh thật.
- Biển ngoài 295 lớp, hoặc biển hỏng/che nhiều, có thể bị gán nhầm; luôn nhìn thêm xác suất (dưới ~50% thì đừng tin).
- Ảnh trong `bo_anh_goc` theo giấy phép CC BY / CC BY-SA / Public domain: báo cáo phải ghi nguồn (có trong `danh_sach.csv`).
