# PRISM — Input / Output từng bước

Thứ tự chạy hợp lý end-to-end và **input → output** của mỗi step.
Đường dẫn mặc định theo `src/prism/config.py`:

- `STORE_DIR   = outputs/store`
- `EXTRACT_DIR = outputs/extract`
- `RELIAB_DIR  = outputs/reliability`
- `DRIFT_DIR   = outputs/drift`
- `MODEL_DIR   = models`
- Override bằng env: `PRISM_WORK_DIR`, `PRISM_MODEL_DIR`, `PRISM_TABSA_ROOT`, `PRISM_HAMOS_ROOT`.

Máy: **[GPU]** = cần GPU (chạy Kaggle), **[CPU]** = chạy local, **[NET]** = cần mạng.

---

## 0. Dữ liệu thô (tiên quyết — có sẵn)

| Item | Đường dẫn |
|---|---|
| Pool review chưa gán nhãn | `data/raw/hotel_booking_unlabeled.jsonl` |
| Gold split | `data/raw/{train,dev,test}.jsonl` |
| Gold ABSA (metadata/ngày) | `data/raw/hotel_absa_labeled.jsonl` |
| Gold quads/segments/images | `hamos-mabsa/` (trỏ bởi `PRISM_HAMOS_ROOT`) |

---

## 1. Module A — Store  `[CPU]`

`python -m prism.module_a_store`

| | |
|---|---|
| **In** | `data/raw/hotel_booking_unlabeled.jsonl`, `data/raw/{train,dev,test}.jsonl`, gold ABSA |
| **Out** | `outputs/store/reviews.jsonl.gz`, `outputs/store/hotel_cohorts.json`, `outputs/store/store_report.json` |

Lọc thời gian, khử trùng lặp, dựng blocklist chống leakage gold, gắn hotel/period/cohort/stratum/has_photo/in_gold.

---

## 2. Module B-data — Text2Text  `[CPU]`

`python -m prism.module_b_data`

| | |
|---|---|
| **In** | gold quads (`hamos-mabsa/`), `data/raw/{train,dev,test}.jsonl` |
| **Out** | `outputs/extract/{train,dev,test}.t2t.jsonl`, `outputs/extract/chrono_{train,test}.t2t.jsonl`, `outputs/extract/data_report.json` |

Chuyển `review text → linearize(quads)` cho mT5; tạo chronological split; kèm parser đảo ngược.

---

## 3. Module B-train — Seed teacher  `[GPU]`

`python -m prism.module_b_train`

| | |
|---|---|
| **In** | `outputs/extract/train.t2t.jsonl`, `outputs/extract/dev.t2t.jsonl`, base `google/mt5-small` |
| **Out** | `models/seed_extractor/` (`config.json` + `model.safetensors` + tokenizer + `training_log.json`) |

Kaggle: `kaggle_pipeline.py --step train`.

---

## 4. Module B-selftrain — Teacher→Student  `[GPU]`

`python -m prism.module_b_selftrain --rounds 2 --cohort B-anchor`

| | |
|---|---|
| **In** | `models/seed_extractor/`, `outputs/extract/{train,dev}.t2t.jsonl`, `data/raw/dev.jsonl` (gold eval), **store** (`outputs/store/reviews.jsonl.gz` + `hotel_cohorts.json`) |
| **Out** | `models/selftrain_round<r>/`, `outputs/extract/selftrain_history.json` (chứa `final_ckpt`) |

Mỗi vòng **tự** gọi: infer (pseudo-label, `--limit 200000`) → lọc theo `tau/tau_post` → train student → eval dev F1 → dừng sớm nếu F1 giảm hoặc phân phối lệch `> max_skew`.
Kaggle: `kaggle_pipeline.py --step selftrain --infer-batch 2 --infer-score-batch 4 --limit 4000`.

> Store là input của bước **infer nội bộ**, KHÔNG phải sản phẩm của B. Nó là output Module A.

---

## 5. Module B-infer — Pool chính thức  `[GPU]`

`python -m prism.module_b_infer --ckpt models/selftrain_round<N cuối> --cohort <cohort>`

| | |
|---|---|
| **In** | checkpoint **student cuối** (`final_ckpt`), **store** (`reviews.jsonl.gz` + `hotel_cohorts.json`) |
| **Out** | `outputs/extract/pool_quads.<cohort>.jsonl.gz` |

Sinh quad + `conf_seq` + `p_model` + posterior provenance + cờ `provenance_flip`.
Lặp cho từng cohort: `T-unbiased`, `A-dense`, `B-anchor`, `corpus`.
Kaggle: `kaggle_pipeline.py --step infer --ckpt <...> --cohort <...> --batch 2 --score-batch 4`.

> Phải chạy SAU selftrain, dùng student cuối. Bản `pool_quads` sinh bằng seed teacher chỉ là sơ bộ, cần đè lại.

---

## 6. Tải ảnh pool (cho Module C)  `[NET]`

`python scripts/download_pool_photos.py --cohort <cohort> --limit 0`

| | |
|---|---|
| **In** | `outputs/store/reviews.jsonl.gz` (URL/ảnh theo cohort) |
| **Out** | ảnh tải về + `outputs/reliability/pool_image_index.json` (`{review_uid: path}`) |

Resume an toàn: ảnh đã có được bỏ qua.

### (kèm) Mẫu audit — `python -m prism.make_audit_samples`
| **Out** | `outputs/reliability/audit_sample_300.jsonl`, `outputs/reliability/d0_sample_300.jsonl` (dùng ở stage `bridge`) |

---

## 7. Module C — Reliability (4 stage, đúng thứ tự)  `[CPU/GPU]`

```
python -m prism.module_c_reliability --stage train_verifier
python -m prism.module_c_reliability --stage apply_verifier --quads outputs/extract/pool_quads.<cohort>.jsonl.gz
python -m prism.module_c_reliability --stage bridge        --quads outputs/extract/pool_quads.<cohort>.jsonl.gz
python -m prism.module_c_reliability --stage apply         --quads outputs/extract/pool_quads.<cohort>.jsonl.gz
```

| Stage | In | Out |
|---|---|---|
| `train_verifier` | gold ảnh (`hamos-mabsa/`) | `outputs/reliability/verifier.pkl`, `verifier_report.json` |
| `apply_verifier` | `pool_quads.<cohort>`, `pool_image_index.json` | điểm ảnh→category cho quad có ảnh |
| `bridge` | `pool_quads.<cohort>`, `audit_sample_300.jsonl` | `outputs/reliability/bridge.pkl`, `bridge_report.json` |
| `apply` | `pool_quads.<cohort>` | `outputs/reliability/quads_weighted.jsonl.gz` (gắn trọng số `w`) |

---

## 8. Module D — Drift  `[CPU]`  **[kết quả cuối]**

`python -m prism.module_d_drift --quads outputs/reliability/quads_weighted.jsonl.gz --cohort corpus`

| | |
|---|---|
| **In** | `outputs/reliability/quads_weighted.jsonl.gz` (hoặc `pool_quads.*` với `w=conf_seq` nếu bỏ qua C) |
| **Out** | `outputs/drift/drift_results.<cohort>.<level>.json` |

Hai kênh π (prevalence/share-of-complaints) + ν (valence), mỗi kênh **raw & adjusted** (CLR, khử mùa vụ, OLS trend, changepoint, permutation + bootstrap, BH-FDR).

---

## 9. Robustness (tùy chọn)  `[CPU]`

```
python -m prism.eval_injection --test composition
python -m prism.eval_injection --test valence
python -m prism.eval_injection --test shuffle
```
| **In** | `quads_weighted.jsonl.gz` | **Out** | `outputs/drift/injection_*.json` |

---

## Sơ đồ phụ thuộc

```
data/raw + hamos-mabsa
   │
   ├─(1)─ module_a_store ─────────────► outputs/store/{reviews.jsonl.gz, hotel_cohorts.json}
   │                                          │ (store dùng ở 4 & 5)
   └─(2)─ module_b_data ──► *.t2t.jsonl       │
                               │              │
              (3) train ──► seed_extractor    │
                               │              │
              (4) selftrain ◄──┴──── store ◄──┤   ──► selftrain_round* (final_ckpt)
                               │              │
              (5) infer ◄── final_ckpt + store┘   ──► pool_quads.<cohort>.jsonl.gz
                               │
              (6) download_photos ──► pool_image_index.json
                               │
              (7) module_c_reliability (train_verifier→apply_verifier→bridge→apply)
                               │              ──► quads_weighted.jsonl.gz
                               │
              (8) module_d_drift ───────────► outputs/drift/drift_results.*.json
```

**Bất biến thứ tự:** `A → B-data → train → selftrain → infer → (photos) → C → D`.
Không đảo `selftrain → infer` (infer chính thức phải dùng checkpoint tốt nhất từ selftrain).

---

# Orchestrator 1-file: `scripts/kaggle_pipeline.py`  (mỗi cell = 1 step)

Gộp toàn bộ `src/prism` thành một entrypoint. **Mỗi cell chạy đúng 1 step** để lỗi
step nào chỉ hỏng step đó. Tự tìm input theo tên file khắp `/kaggle/input`, stage vào
`WORK_DIR/MODEL_DIR`, output lưu ở `/kaggle/working` (thành output notebook → attach cho
step sau). Thiếu file nào báo lỗi rõ "chạy step X trước".

Tên step: `store · data · train · selftrain · infer · photos · c_verifier ·
c_apply_verifier · c_bridge · c_apply · drift · injection · all`

```python
# Cell setup (1 lần)
!rm -rf /kaggle/working/Prism
!git clone --depth 1 https://github.com/hotuyen21pt/Prism.git /kaggle/working/Prism
P = "/kaggle/working/Prism/prism/scripts/kaggle_pipeline.py"
```

```python
# mỗi cell 1 step — ví dụ chuỗi Module B (GPU)
!python {P} --step train                                          # → seed_extractor
!python {P} --step selftrain --cohort B-anchor --infer-batch 2 --infer-score-batch 4
!python {P} --step infer --cohort T-unbiased --batch 2 --score-batch 4 \
        --ckpt /kaggle/input/notebooks/tuyennguyen21pt/module-b-selftrain/models/selftrain_round2
```

```python
# Module C + D (mỗi cell 1 stage; photos→c_* nên CÙNG notebook để giữ path ảnh)
!python {P} --step photos           --cohort T-unbiased
!python {P} --step c_verifier
!python {P} --step c_apply_verifier --cohort T-unbiased          # → pool_quads_vimg (có v_image)
!python {P} --step c_bridge         --cohort T-unbiased          # đọc v_image → bridge.pkl
!python {P} --step c_apply          --cohort T-unbiased          # → quads_weighted.jsonl.gz
!python {P} --step drift            --cohort corpus              # kết quả cuối
```

```python
# HOẶC chạy hết 1 cell — tự BỎ QUA step đã có output dưới /kaggle/working
!python {P} --step all --cohort T-unbiased        # thêm --force để chạy lại tất cả
```

Ghi chú nối Module C (đã xử lý sẵn trong orchestrator):
`apply_verifier` sinh `pool_quads_vimg.<cohort>.jsonl.gz` (thêm trường `v_image`);
`bridge` **đọc file vimg này** (không phải pool_quads gốc) vì cần `v_image`; `apply`
đọc pool_quads gốc + `bridge.pkl` → gắn `w`.
