# Kaggle notebooks — chạy PRISM từng step

Mỗi `.ipynb` chạy **đúng 1 step** qua `scripts/kaggle_pipeline.py`. Input được tìm tự
động theo tên file khắp `/kaggle/input` (kể cả output notebook step trước đã attach), nên
bạn **chỉ cần attach** dataset/output phù hợp — không phải sửa path trong code.

## Quy trình dùng
1. Import `.ipynb` step cần chạy lên Kaggle (New Notebook → File → Upload).
2. Bật **GPU / Internet** theo ghi chú đầu notebook.
3. **Add Data**: attach dataset gốc + **output notebook của step ngay trước**.
4. Run All → **Save Version (Save & Run All)** để giữ output.
5. Notebook step sau: attach chính output vừa lưu.

## Thứ tự & phụ thuộc

| Notebook | Step | Máy | Attach (ngoài input gốc) |
|---|---|---|---|
| `01_store` | store | CPU | raw pool + splits + gold ABSA |
| `02_data` | data | CPU | hamos-mabsa + splits |
| `03_train` | train | GPU | output 02 (t2t) |
| `04_selftrain` | selftrain | GPU | output 03 (ckpt) + t2t + dev.jsonl + store |
| `05_infer` | infer | GPU | output 04 (ckpt student) + store |
| `06_photos` | photos | CPU+Net | store |
| `07_c_verifier` | c_verifier | GPU | hamos-mabsa |
| `08_c_apply_verifier` | apply_verifier | GPU | output 05 + 06 + 07 |
| `09_c_bridge` | bridge | CPU | output 08 (vimg) |
| `10_c_apply` | apply | CPU | output 05 + 09 (bridge.pkl) |
| `11_drift` | drift | CPU | output 10 (quads_weighted) |
| `00_full_pipeline` | all | GPU+Net | tất cả — tự bỏ qua step đã có output |

**Bất biến:** `store → data → train → selftrain → infer → photos → c_verifier →
c_apply_verifier → c_bridge → c_apply → drift`. Không đảo `selftrain → infer`.

Chi tiết input/output từng step: [../docs/pipeline_io.md](../docs/pipeline_io.md).

## Ghi chú
- `04_selftrain` dùng `--limit 4000` (giới hạn pseudo-label) để vừa thời lượng Kaggle;
  mặc định 200k sẽ mất ~58h/vòng. Tăng dần nếu còn thời gian.
- Bước 6→10 nên chạy **cùng 1 notebook** vì `pool_image_index.json` chứa đường dẫn ảnh
  tuyệt đối (`/kaggle/working/...`) — attach lại ở notebook khác sẽ lệch path.
- Chưa có `hamos-mabsa` trên Kaggle: bỏ 07→10, chạy thẳng `11_drift` (tự fallback
  `pool_quads` với `w=conf_seq`).
