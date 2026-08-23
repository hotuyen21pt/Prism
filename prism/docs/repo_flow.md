# Flow và nhiệm vụ từng file trong PRISM

## 1. Mục tiêu

PRISM phân tích sự thay đổi theo thời gian của sentiment ở mức aspect trong review khách sạn Booking.com. Pipeline sử dụng dữ liệu gold để huấn luyện extractor, chạy extractor trên pool review chưa gán nhãn, tính độ tin cậy của quad, rồi phát hiện sentiment drift theo thời gian.

Flow chính:

```text
data/raw/*.jsonl
    -> module_a_store.py
    -> module_b_data.py
    -> module_b_train.py
    -> module_b_eval.py / module_b_infer.py
    -> module_c_reliability.py
    -> module_d_drift.py
    -> outputs/drift/
```

`eval_injection.py` chạy sau cùng để kiểm tra robustness và negative control.

## 2. Dữ liệu đầu vào

- `data/raw/hotel_booking_unlabeled.jsonl`: pool review Booking.com chưa gán nhãn.
- `data/raw/hotel_absa_labeled.jsonl`: metadata review, chủ yếu dùng để bổ sung hoặc đối chiếu `review_date`; file này không chứa quad.
- `data/raw/train.jsonl`: dữ liệu gold dùng để train.
- `data/raw/dev.jsonl`: dữ liệu validation.
- `data/raw/test.jsonl`: dữ liệu đánh giá cuối.
- `data/README.md`: mô tả dữ liệu, schema và quy tắc không để review gold lọt vào pool.

Mỗi dòng gold thường có `instance_id`, `source_review_id`, `segment_id`, `text`, `review_date` và `quads`. Một quad gồm aspect, category, taxonomy code, opinion, sentiment và các span tương ứng.

## 3. Cấu hình và tiện ích

- `src/prism/__init__.py`: khai báo Python package.
- `src/prism/config.py`: định nghĩa đường dẫn, thư mục output, mốc thời gian, cohort, strata và biến môi trường như `PRISM_HAMOS_ROOT`.
- `src/prism/utils.py`: các hàm dùng chung để đọc JSONL, parse ngày, chuẩn hóa dữ liệu và xử lý quad.
- `pyproject.toml`: metadata và cấu hình project/test.
- `requirements.txt`: các thư viện Python cần cài.

## 4. Module A: tạo temporal store

File: `src/prism/module_a_store.py`

Module A đọc pool review, parse ngày, lọc khoảng thời gian, chuẩn hóa dữ liệu, loại duplicate và dựng blocklist chống leakage từ gold. Sau đó module gắn hotel, thời gian, cohort, strata, độ dài review, trạng thái có ảnh và trạng thái gold.

Output chính:

- `outputs/store/reviews.jsonl.gz`
- `outputs/store/store_report.json`
- `outputs/store/hotel_cohorts.json`

## 5. Module B-data: chuẩn bị dữ liệu extractor

File: `src/prism/module_b_data.py`

Module B-data chuyển review và các quad thành format text-to-text để mT5 học ánh xạ:

```text
review text -> aspect, category, opinion, sentiment
```

Module này cũng tạo chronological split và parser để chuyển output text của model trở lại quad có cấu trúc.

Output chính:

- `outputs/extract/train.t2t.jsonl`
- `outputs/extract/dev.t2t.jsonl`
- `outputs/extract/test.t2t.jsonl`
- `outputs/extract/chrono_train.t2t.jsonl`
- `outputs/extract/chrono_test.t2t.jsonl`

## 6. Train, evaluate và inference

### `src/prism/module_b_train.py`

Fine-tune model mT5 trên dữ liệu gold. Model được lưu trong `models/seed_extractor/`.

### `src/prism/module_b_eval.py`

Đánh giá prediction trên dev/test bằng precision, recall và F1. Có thể phân tích theo ngôn ngữ, độ dài, polarity và category. Kết quả nằm trong `outputs/extract/eval_report.json`.

### `src/prism/module_b_infer.py`

Chạy extractor trên pool chưa gán nhãn, sinh các quad dự đoán và thông tin provenance/reliability ban đầu. Kết quả nằm trong `outputs/extract/pool_quads.<cohort>.jsonl.gz`.

### `src/prism/module_b_selftrain.py`

Bước tùy chọn teacher-student: lọc pseudo-label có độ tin cậy cao, train student, đánh giá trên dev và dừng nếu chất lượng giảm hoặc polarity bị lệch. Lịch sử nằm trong `outputs/extract/selftrain_history.json`.

## 7. Module C: reliability

File: `src/prism/module_c_reliability.py`

Module C ước lượng độ tin cậy của quad bằng audit con người, ảnh và đặc trưng text.

Các stage chính:

1. `train_verifier`: train image verifier.
2. `apply_verifier`: tính điểm hỗ trợ của ảnh cho quad.
3. `bridge`: học bridge từ đặc trưng text/image sang reliability.
4. `apply`: gắn trọng số `w_q` vào quad.

Output chính:

- `outputs/reliability/verifier.pkl`
- `outputs/reliability/verifier_report.json`
- `outputs/reliability/bridge.pkl`
- `outputs/reliability/bridge_report.json`
- `outputs/reliability/quads_weighted.jsonl.gz`

Các file hỗ trợ:

- `src/prism/make_audit_samples.py`: tạo mẫu audit D0 và human audit.
- `scripts/download_pool_photos.py`: tải ảnh pool, hỗ trợ resume và giới hạn tốc độ.
- `outputs/reliability/d0_sample_300.jsonl`: mẫu D0.
- `outputs/reliability/audit_sample_300.jsonl`: mẫu audit.

## 8. Module D: phát hiện drift

File: `src/prism/module_d_drift.py`

Module D đọc quad đã weighted, gom theo thời gian/hotel/aspect, rồi:

1. Tính complaint share và negativity rate.
2. Chuẩn hóa compositional data bằng CLR.
3. Loại ảnh hưởng mùa vụ.
4. Ước lượng trend bằng OLS.
5. Tìm changepoint.
6. Chạy permutation test và bootstrap.
7. Điều chỉnh nhiều kiểm định bằng BH-FDR.

Kết quả nằm trong `outputs/drift/drift_results.<cohort>.<level>.json`.

## 9. Kiểm tra robustness

### `src/prism/eval_injection.py`

Kiểm tra pipeline bằng các tình huống injection:

- composition injection
- valence injection
- shuffle negative control

Kết quả nằm trong `outputs/drift/injection_*.json`.

### `scripts/probe_complaint_composition.py`

Probe độc lập bằng keyword để kiểm tra complaint composition. Đây là công cụ kiểm chứng, không phải bước bắt buộc của pipeline chính.

## 10. Smoke test và unit test

- `src/prism/smoke_test.py`: smoke test các phần chính không cần Torch.
- `tests/test_utils.py`: test utility và parse ngày.
- `tests/test_module_b_data.py`: test linearization/parser quad.
- `tests/test_module_d.py`: test aggregation và drift.
- `tests/test_injection.py`: test injection và negative control.

Chạy test:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover tests
python -m prism.smoke_test
```

## 11. Thứ tự chạy đề xuất

```powershell
$env:PYTHONPATH="src"

python -m prism.module_a_store
python -m prism.module_b_data
python -m prism.make_audit_samples

python -m prism.module_b_train
python -m prism.module_b_eval
python -m prism.module_b_infer

python -m prism.module_c_reliability --stage train_verifier
python scripts/download_pool_photos.py
python -m prism.module_c_reliability --stage apply_verifier
python -m prism.module_c_reliability --stage bridge
python -m prism.module_c_reliability --stage apply

python -m prism.module_d_drift
python -m prism.eval_injection --test composition
python -m prism.eval_injection --test valence
python -m prism.eval_injection --test shuffle
```

Tóm tắt: **Raw data -> Store -> Text-to-text dataset -> Extractor -> Reliability weighting -> Temporal drift -> Robustness checks**.