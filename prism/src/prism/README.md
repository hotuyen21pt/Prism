# PRISM — Pipeline huấn luyện & phân tích drift trên dữ liệu thật

Hiện thực hoá method trong `docs/method_table_q1.md`:
**P**rovenance-anchored, **R**eliability-calibrated, **I**mage-verified **S**entiment **M**onitoring.

```text
src/prism/
├── config.py               # MỌI hằng số/ngưỡng (số [đo] là số đã audit, không đặt tay)
├── utils.py                # parse ngày VN, strata, CLR, BH-FDR, Welch-t, quad_uid, I/O
├── module_a_store.py       # A — Temporal Quad Store (blocklist, dedup, cửa sổ, cohort)
├── module_b_data.py        # B — gold -> định dạng text-to-text + chronological probe
├── module_b_train.py       # B — fine-tune mT5 seed extractor
├── module_b_eval.py        # B — E1/E1b: F1 tách ngôn ngữ/độ dài/cực tính (instance & quad)/category
├── module_b_infer.py       # B — suy luận pool + P_model(s|x) rescore + provenance posterior
├── module_b_selftrain.py   # B — vòng lặp teacher-student có guardrail
├── module_c_reliability.py # C — verifier + apply_verifier + cầu nối + w_q  [go/no-go]
├── module_d_drift.py       # D — drift 2 kênh (π complaint-share, ν negativity) × (raw, adj)
├── eval_injection.py       # E3/E4 — injection & negative control (thí nghiệm lõi)
├── make_audit_samples.py   # tạo 2 mẫu annotation người (D0 + AUDIT, reservoir chuẩn)
└── smoke_test.py           # kiểm tra end-to-end trên mẫu nhỏ (✅ PASS trên dữ liệu thật)

scripts/download_pool_photos.py     # tải ảnh pool cho C2 (resumable, rate-limited)
scripts/probe_complaint_composition.py  # CACT keyword-probe (kiểm chứng khả thi ban đầu)
tests/                              # unit test (stdlib) — không cần dữ liệu/torch
```

Mọi output ghi vào `outputs/` (đổi bằng biến môi trường `PRISM_WORK_DIR`):
`outputs/store/` · `outputs/extract/` · `outputs/reliability/` · `outputs/drift/` · `models/`.

> **Changelog 2026-08-22 (audit code):** thêm kênh ν hiệu chỉnh thành phần + FDR
> (estimand trung tâm — trước đây chỉ có chuỗi thô); thay loại-ô-nhỏ bằng shrinkage
> (loại ô đột ngột sinh trend giả trong chính kênh adj — đã tái hiện bằng injection
> nền null); P_model(s|x) thật bằng rescore 3 biến thể (bỏ one-hot cứng); E3a chạy
> trên nền null + tiêu chí PASS chặt; E4 lặp nhiều seed; verifier thêm baseline
> không-ảnh (ΔAUC) chống rò rỉ category-prior; thêm stage `apply_verifier` (C2) +
> script tải ảnh pool; sửa bug khoá `quad_uid` không khớp giữa mẫu audit và bridge;
> bootstrap đưa ra ngoài vòng per-aspect (nhanh |aspects|×); reservoir sampling chuẩn.

---

## 0. Cài đặt môi trường

Đã có sẵn: Python 3.9.6, numpy/pandas/scipy/statsmodels/PIL/tqdm.
**Thiếu (chỉ cần cho Module B-train/infer và C):**

```bash
cd /Users/macbook/RnD/tabsa
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
# Module B (huấn luyện + suy luận)
pip install torch torchvision transformers sentencepiece accelerate scikit-learn
# Module C (image verifier)
pip install open_clip_torch
# tuỳ chọn: kênh semantic drift + langid thật
pip install sentence-transformers fasttext-wheel
```

Máy Mac (Apple Silicon) tự dùng MPS; không cần cấu hình gì thêm.
**Module A, D, smoke_test, eval_injection và make_audit_samples chạy bằng thư viện chuẩn — không cần torch.**

Mọi lệnh dưới đây chạy từ `/Users/macbook/RnD/tabsa` với `PYTHONPATH=src`:

```bash
export PYTHONPATH=src
```

---

## 1. Unit test + smoke test (LÀM ĐẦU TIÊN — ~2 phút, không cần torch)

```bash
python3 -m unittest discover tests    # 37 test thuần logic, ~0,2s, không cần dữ liệu
python3 -m prism.smoke_test           # end-to-end trên dữ liệu thật
```

Xác nhận 5 điều trên **dữ liệu thật**: đường dẫn đúng · blocklist bắt được leak
(148 dòng/20k mẫu) · 23.995 quad round-trip sạch · Module D chạy đủ 36 kỳ ·
negative control shuffle (5 lần lặp) cho FPR = 0. Nếu bước này fail thì **dừng**,
không chạy tiếp.

---

## 2. Module A — Temporal Quad Store (~10 phút, không cần torch)

```bash
python3 -m prism.module_a_store
```

| Việc | Chi tiết |
|---|---|
| Parse ngày | regex `ngày D tháng M năm YYYY` ([đo] 99,99% khớp; 184 null) |
| Cắt cửa sổ | **2022-03 → 2025-02** (loại 2025-03 crawl dở + 2022-02 thưa) |
| Dedup | khoá `(hotel,name,date,pos,neg)` — [đo] 15.137 dòng trùng |
| **Gold blocklist** 🔴 | khoá `(hotel,date)` + hash text; gắn cờ `in_gold` — **các dòng này không bao giờ được train/self-train/infer** |
| Strata | `country_bloc`(4) × `traveller`(5) — dùng cho hiệu chỉnh thành phần |
| Cohort | `A-dense`(≥1000 review + gold) · `B-anchor`(≥300 + gold) · `T-unbiased`(514 hotel test — extractor chưa thấy) · `corpus` |

Ra: `outputs/store/reviews.jsonl.gz` (179 MB) + `store_report.json` + `hotel_cohorts.json`.

**✅ Kết quả run thật (2026-08-21):** đọc 1.949.604 → ghi **1.921.661** review
(loại: 12.602 ngoài cửa sổ · 15.157 trùng · 184 không ngày). Có text: **1.141.532**.
**`flag_gold_leak` = 18.475** — nhiều hơn 8.796 gold vì khoá `(hotel, date)` cố tình thô;
over-flag là hướng an toàn (thà loại nhầm vài nghìn dòng khỏi training còn hơn lọt test set).
Cohort: A-dense **270** · B-anchor **1.207** · T-unbiased **514** · corpus 10.629.
Nếu `flag_gold_leak` = 0 → blocklist hỏng, dừng lại.

---

## 3. Module B — Seed extractor

### 3.1 Chuẩn bị dữ liệu (không cần torch)

```bash
python3 -m prism.module_b_data
```

Chuyển `train/dev/test.jsonl` (8.816/1.895/1.890 segment · 23.995 quad, split
**hotel-disjoint**) sang text-to-text:

```
input : "extract quads: Nhân viên nhiệt tình"
target: "<quad> Nhân viên | SER_ATTITUDE | nhiệt tình | positive </quad>"
```

Đồng thời tạo **chronological probe** (E1c): `chrono_train` ≤2024-06 (7.046) /
`chrono_test` ≥2024-07 (3.665) — chỉ cắt từ train+dev, test gốc giữ nguyên.

### 3.2 Tạo mẫu D0 cho annotation (không cần torch — làm SỚM, chặn D-a)

```bash
python3 -m prism.make_audit_samples          # -> outputs/reliability/d0_sample_300.jsonl
```

300 review phân tầng theo length_bin × có/không text_neg (reservoir sampling chuẩn).
Người annotate điền trường `gold_quads` (TOÀN BỘ quad của review, văn bản đầy đủ
không cắt). Từ kết quả, dựng bảng recall
`{"positive|L0": 0.xx, ...}` → lưu `outputs/drift/recall_table.json` cho bước D-a.

### 3.3 Huấn luyện (cần torch; ~2-4h MPS, ~40ph GPU)

```bash
python3 -m prism.module_b_train --model google/mt5-base --epochs 10
# máy yếu: --model google/mt5-small --batch 4 --grad-accum 8
```

Dùng **mT5** vì gold đa ngữ (80,5% en · 16,2% vi). Giữ checkpoint tốt nhất theo dev loss
→ `models/seed_extractor/`.

### 3.4 Đánh giá — E1/E1b/E1c (con số tham chiếu bắt buộc)

```bash
# E1 + E1b: test chuẩn, tách ngôn ngữ / độ dài / cực tính / category
python3 -m prism.module_b_eval

# E1c: chronological probe — TRAIN LẠI trên chrono_train rồi đo chrono_test
python3 -m prism.module_b_train --train-file outputs/extract/chrono_train.t2t.jsonl \
    --out models/chrono_probe --epochs 10
python3 -m prism.module_b_eval --ckpt models/chrono_probe \
    --test-file outputs/extract/chrono_test.t2t.jsonl   # cần gold-file tương ứng, xem --help
```

Đọc `outputs/extract/eval_report.json`:
- `by_language.vi` thấp hơn hẳn `en` → đúng dự đoán (gold 80,5% en); cân nhắc oversample vi.
- `by_length.L2` thấp → **xác nhận D-a recall lệch theo độ dài** — số này đi thẳng vào paper.
- **`by_polarity_quad`** (P/R/F1 mức quad theo cực tính) — số đo trực tiếp cho recall
  lệch cực tính của M3a; `by_polarity` (mức instance) chỉ để tham khảo.
- E1c tụt mạnh so với E1 → temporal generalization kém → mọi kết luận drift phải hạ mức tự tin.

### 3.5 Suy luận pool + provenance posterior

```bash
# LUÔN chạy cohort nhỏ trước (T-unbiased 514 hotel), corpus sau cùng
python3 -m prism.module_b_infer --cohort T-unbiased
python3 -m prism.module_b_infer --cohort B-anchor
python3 -m prism.module_b_infer --cohort corpus        # 1,15M review — chạy qua đêm
```

Mỗi review sinh tối đa 2 đơn vị: `(text_pos, φ=POS)` và `(text_neg, φ=NEG)`.
Sentiment cuối = posterior kết hợp **P_model(s|x) thật** (chấm điểm lại 3 biến thể
sentiment bằng teacher forcing — không phải one-hot xấp xỉ) với prior **bất đối
xứng đo trên gold** (POS→93,1% · NEG→64,1% — `config.PROVENANCE_PRIOR`),
**không lọc cứng**; quad bị lật (`provenance_flip=true`) được giữ để audit.
`--no-rescore` tắt bước chấm điểm (chỉ để debug — kết quả paper phải có rescore).
λ mặc định 0,7 (`config.PROVENANCE_LAMBDA`) — **cần chọn lại trên dev** trước run
chính thức (quét λ ∈ {0.5…0.9}, tối đa dev F1 sentiment).
Dòng `in_gold=true` **tự động bị bỏ qua** (blocklist).

### 3.6 Tạo mẫu AUDIT 300 quad (sau khi có pool quads)

```bash
python3 -m prism.make_audit_samples --quads outputs/extract/pool_quads.T-unbiased.jsonl.gz
# -> outputs/reliability/audit_sample_300.jsonl ; người annotate điền "correct": 0|1
```

Phân tầng conf_seq(3) × φ(2) × provenance_flip(2). Khoá `quad_uid` sinh bằng
`utils.quad_uid` — **cùng hàm** mà Module C bridge dùng để đối chiếu (không tự dựng khoá).

### 3.7 Self-training (tuỳ chọn, sau khi 3.4 đạt)

```bash
python3 -m prism.module_b_selftrain --rounds 2 --cohort B-anchor
```

Guardrail tự dừng khi: dev F1 giảm, hoặc %negative của pseudo lệch >10 điểm so với
gold (15,4%) — dấu hiệu error propagation. Lịch sử: `outputs/extract/selftrain_history.json`.

---

## 4. Module C — Cross-modal reliability [go/no-go]

**Ngưỡng chốt trước (sửa trong `module_c_reliability.py::GO_NOGO` nếu nhóm quyết khác):**
AUC verifier ≥ 0,70 · **ΔAUC so với baseline không-ảnh ≥ 0,05** (chống rò rỉ
category-prior: one-hot category tự đoán được y vì FACILITY có mặt ở hầu hết review —
AUC cao ≠ ảnh có ích) · Spearman(r̂, human audit) ≥ 0,30.

```bash
# C1: học verifier ảnh↔category trên 9.219 ảnh gold (split hotel-disjoint)
python3 -m prism.module_c_reliability --stage train_verifier
# -> đọc outputs/reliability/verifier_report.json: "go_nogo": "GO" | "NO-GO"
#    (kèm auc_no_image_baseline, delta_auc, auc_by_category)

# C2: tải ảnh pool rồi gắn v_image cho quad thuộc review có ảnh
python3 scripts/download_pool_photos.py --cohort T-unbiased
python3 -m prism.module_c_reliability --stage apply_verifier \
    --quads outputs/extract/pool_quads.T-unbiased.jsonl.gz \
    --out   outputs/extract/pool_quads.T-unbiased.vimg.jsonl.gz

# C3: cầu nối text->P[V=1] (kèm IPW has_photo) + đối chiếu human audit (mẫu §3.6)
python3 -m prism.module_c_reliability --stage bridge \
    --quads outputs/extract/pool_quads.T-unbiased.vimg.jsonl.gz \
    --audit outputs/reliability/audit_sample_300.jsonl

# gắn w_q cho toàn bộ quad (tự fallback conf_seq nếu NO-GO)
python3 -m prism.module_c_reliability --stage apply \
    --quads outputs/extract/pool_quads.corpus.jsonl.gz \
    --out   outputs/reliability/quads_weighted.jsonl.gz
```

Định dạng file audit người (300 quad, lấy mẫu phân tầng): mỗi dòng đã có sẵn
`quad_uid`, người annotate chỉ điền `"correct": 0|1`.

**NO-GO không chặn pipeline** — `apply` fallback về `conf_seq`; bài lùi về A+B+D
(vẫn đủ IP&M/KBS, mất mũi nhọn Information Fusion). Vì vậy **chạy C sớm** để biết scope.

---

## 5. Module D — Drift (khung đã kiểm chứng, không cần torch)

```bash
# kết quả chính: corpus, mức taxonomy_code (14 code đủ mẫu; 17 code hiếm tự gộp lên category)
python3 -m prism.module_d_drift --quads outputs/reliability/quads_weighted.jsonl.gz \
    --level taxonomy_code --cohort corpus --n-boot 1000

# đối chiếu cohort không thiên vị (E5): extractor CHƯA TỪNG thấy 514 hotel này
python3 -m prism.module_d_drift --quads outputs/reliability/quads_weighted.jsonl.gz \
    --level taxonomy_code --cohort T-unbiased --n-boot 1000
```

Mỗi aspect cho ra **hai kênh × hai bản**:

| Kênh | Ý nghĩa | Thang | raw | adj |
|---|---|---|---|---|
| **π** | share-of-complaints (quad φ=NEG hoặc sentiment=negative) | CLR | ✓ | direct standardization |
| **ν** | negativity rate P(neg \| nhắc aspect) — **estimand trung tâm** | tỷ lệ | ✓ | direct standardization |

Ô strata nhỏ được **shrink về share/tỷ lệ gộp của kỳ** (prior cường độ
`--min-stratum` / `--min-valence-w`) thay vì loại đột ngột — loại ô làm mix strata
đóng góp tự trôi theo thời gian và sinh trend giả trong chính kênh adj (đã tái hiện
bằng injection nền null). Mỗi chuỗi: slope/năm + t-stat + changepoint + `p_perm`
(null riêng từng chuỗi) + `p_fdr` — **lưới FDR chính = π-adj ∪ ν-adj**; lưới raw
FDR riêng chỉ để so like-for-like trong E3a. Bootstrap CI (kênh π, resample review
trong strata, MỘT bản resample dùng chung cho mọi aspect) + verdict/verdict_val:

```
XU HƯỚNG THẬT | GIẢ (do thành phần) | BỊ CHE, lộ ra sau hiệu chỉnh | ĐẢO DẤU | phẳng
```

**Số chính thức của paper là các cột `adj`/`val_adj` sau FDR.** Cột raw chỉ để đối chiếu.
Trong JSON: `periods` là trục của `nu_*_series`, `pi_periods` là trục của `pi_*_series`.
`--out` đổi đường dẫn output (mặc định `drift_results.<cohort>.<level>.json`).

D-a (hiệu chỉnh recall): tạo `outputs/drift/recall_table.json` từ **tập audit D0** (§3.2).
⚠️ Bước áp bảng recall vào ước lượng hiện là **stub** (chỉ log) — phải hoàn thiện
khi có bảng thật, trước run kết quả cuối.

---

## 6. Đánh giá lõi — injection & negative control (không cần torch)

```bash
Q=outputs/reliability/quads_weighted.jsonl.gz

# E3a composition: XÁO timestamp làm nền null trước (dữ liệu thật có drift thật,
#   không thể đòi adj im lặng trên đó), rồi đổi tỷ trọng strata (mô phỏng VN
#   89,7%→16,1%), KHÔNG đổi nội dung.
#   PASS = adj im lặng (0 aspect sau FDR) VÀ raw báo (>=1, cũng sau FDR)
#   INCONCLUSIVE = cả hai im lặng (injection quá yếu) — không kết luận được
python3 -m prism.eval_injection --quads $Q --test composition

# E3c valence: lật δ% pos->neg của AM_FOOD sau 2023-09, δ ∈ {2,5,10,20}%
#   (chỉ đổi sentiment, KHÔNG đổi φ — injection đổi đúng một biến)
#   -> detection trên kênh ν-adj + changepoint đúng vị trí (±3 kỳ); π-adj báo phụ
python3 -m prism.eval_injection --quads $Q --test valence --aspect AM_FOOD --t0 2023-09

# E4 negative control: xáo trộn timestamp, LẶP --repeats 5 seed khác nhau
#   -> FPR thực nghiệm trung bình trên lưới (π-adj ∪ ν-adj) phải ≈ α=0,05
python3 -m prism.eval_injection --quads $Q --test shuffle --repeats 5
```

Kết quả: `outputs/drift/injection_*.json`. Ba file này là **bảng evaluation trung tâm**
của paper. Injection ghi kết quả drift vào file riêng (`drift_results.inj_*.json`)
— **không ghi đè** kết quả corpus chính.

---

## 7. Thứ tự chạy chuẩn (đường găng)

```text
1.  smoke_test                     ✅ PASS (blocklist 148 leak/20k · roundtrip 23.995 quad
                                           sạch · FPR shuffle 0,0/5 lặp · unit-test PASS)
2.  module_a_store                 ✅ ĐÃ CHẠY (1.921.661 review, 18.475 gold-leak flagged)
3.  module_b_data                  ✅ ĐÃ CHẠY (chrono probe 7.046/3.665)
3b. make_audit_samples (D0)        ✅ ĐÃ CHẠY (300 review, 6 ô strata)
                                   → GIAO ANNOTATE NGAY (chặn D-a; ~1,5 tuần người)
4.  module_b_train                 (~2-4h MPS)  ← cần cài torch trước (§0)
5.  module_b_eval  (+ chrono probe E1c) → chốt bảng F1 tham chiếu
6.  module_c train_verifier        ← GO/NO-GO theo ΔAUC — chạy sớm để biết scope
7.  module_b_infer --cohort T-unbiased → B-anchor → corpus  (giữ rescore bật)
7b. make_audit_samples --quads ... (AUDIT 300 quad) → GIAO ANNOTATE
8.  scripts/download_pool_photos + module_c apply_verifier → bridge → apply
9.  module_d_drift  (corpus + T-unbiased, --n-boot 1000)
10. eval_injection composition + valence + shuffle --repeats 5
11. (tuỳ chọn) module_b_selftrain rồi lặp 7-10 với checkpoint mới
```

## 8. Quy tắc bất biến (vi phạm = kết quả vô hiệu)

1. **Không bao giờ** train/self-train/infer trên dòng `in_gold=true`.
2. `review_score` **chỉ** dùng validate (E8), **không bao giờ** làm feature lọc pseudo-label.
3. **Không** dùng temporal consistency làm bộ lọc — nó xoá chính tín hiệu drift cần đo.
4. Số chính thức = cột `adj`/`val_adj` **sau FDR**; luôn báo cáo kèm cột raw để minh bạch.
5. Ngưỡng go/no-go (AUC, ΔAUC, Spearman) và ngưỡng E8 (Spearman 0,3–0,7)
   **chốt trước khi chạy** — chống HARKing.
6. Kết quả chính thức của Module B infer **phải** chạy với rescore (không `--no-rescore`).
7. Mọi run ghi kèm `config`/`args` vào output JSON — tái lập được.

## 9. Hai mẫu annotation người (TÁCH BIỆT — không được trộn)

| Mẫu | Tạo ở bước | File | Người annotate điền | Dùng cho |
|---|---|---|---|---|
| D0 | §3.2 (sau Module A) | `outputs/reliability/d0_sample_300.jsonl` | `gold_quads`: toàn bộ quad của review | D-a recall table |
| AUDIT | §3.6 (sau infer) | `outputs/reliability/audit_sample_300.jsonl` | `correct`: 0/1 cho từng quad | Module C bridge |

D0 đo **recall của extractor** (annotate đầy đủ văn bản không cắt); AUDIT đo
**precision của pseudo-quad**. Trộn hai mẫu là circular. Sau khi annotate D0,
build bảng recall `{"positive|L0": 0.xx, ...}` → `outputs/drift/recall_table.json`.
