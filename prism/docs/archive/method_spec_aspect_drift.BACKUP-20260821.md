# Đặc tả Method chi tiết — Aspect-Level Sentiment Drift in Hotel Reviews

Tài liệu này cụ thể hoá kiến trúc đã sửa ở lần review trước thành một method spec có thể lập trình
được: input/output chính xác theo schema dữ liệu thật, công thức cho từng bước, và checklist công
việc theo đúng những gì đã đo được trên `reviews_with_dates.jsonl` (HAMoS) và
`hotel_booking_data2.jsonl` (Booking pool). Không có phần nào dưới đây dựa trên giả định chưa kiểm
chứng — mọi con số tham chiếu đều lấy từ các lần audit đã chạy trong phiên làm việc này.

---

## 1. Input đầu vào — chính xác theo schema đã audit

### 1.1 Tập gold (HAMoS) — dùng để huấn luyện extractor và làm test set cuối cùng

Nguồn: `reviews_with_dates.jsonl` (8.796 dòng) + quad annotation gốc trong `train/dev/test.jsonl`.

| Trường | Kiểu | Ghi chú |
|---|---|---|
| `source_review_id` | str | khoá nối với quad annotation, dạng `H<hotel_id>_R<index>` |
| `hotel_id` | int | **3.399 hotel duy nhất**, 3.387 khớp được với Booking pool |
| `review_text` / `text` | str | văn bản gốc, đa ngôn ngữ |
| `review_date` | str `YYYY-MM-DD` | đã phục hồi từ raw crawl; **8.795/8.796 có ngày** (1 bản ghi `H11975221_R00001` không map được) |
| `languages` | list | en 7.548 · vi 1.093 · other 242 |
| `split` | str | train 6.156 / dev 1.320 / test 1.320 — **giữ nguyên split gốc, không random lại** |
| `quads[]` | list of dict | `aspect_term, aspect_category, taxonomy_code, opinion_term, sentiment, aspect_implicit, opinion_implicit, primary_image_id, aspect_span, opinion_span` |

⚠️ **Việc phải làm trước khi dùng:** kiểm tra 3.517 `direct_conflicting_review_ids` trong
`review_date_mapping_report.json` có giao với 8.795 review đã map hay không. Nếu có, ngày của các
review đó không đáng tin và cần loại khỏi tập train temporal model.

### 1.2 Tập pool chưa gán nhãn (Booking) — dùng cho self-training + suy luận trend

Nguồn: `hotel_booking_data2.jsonl` (678.219 dòng, 10.555 hotel).

| Trường | Kiểu | Ghi chú |
|---|---|---|
| `hotel_id` | **str** (khác kiểu với HAMoS — cần ép kiểu khi join) | 10.555 hotel |
| `review_date` | str `DD/MM/YYYY` | 678.147/678.219 có ngày (99.99%) |
| `review_score` | float 1–10 | lệch dương mạnh (9–10 chiếm 57%) |
| `review_positive` / `review_negative` | str | tách sẵn theo cực tính — dùng làm **weak aspect signal** giá rẻ và làm nguồn đối chiếu label-free |
| `review_photo` | dict `{url: caption}` | 78.132 review (11,5%) có ảnh, 173.911 ảnh, có caption |
| `room`, `state` (loại khách: cặp đôi/gia đình/khách lẻ/nhóm), `country`, `nights`, `stars_rating`, `time_crawl` | metadata | dùng để stratify, kiểm soát nhiễu reviewer-population |

**Tập con quan trọng nhất — "anchor hotels":** 388 hotel vừa có gold quad (HAMoS) vừa có ≥300 review
trong pool. Đây là nơi model vừa có thể **evaluate bằng F1 thật (từ HAMoS)** vừa có đủ khối lượng để
**ước lượng trend đáng tin (từ pool)** — nên bắt đầu toàn bộ pipeline ở tập con này trước khi mở rộng
ra 452 hotel nói chung hay 10.555 hotel toàn pool.

**Tập con phụ — "photo-rich hotels":** 345 hotel có ≥30 ảnh/năm trong ≥2 năm riêng biệt (118 hotel
trong ≥3 năm) — dùng cho nhánh visual corroboration tuỳ chọn.

### 1.3 Input hình thức hoá cho model

```
Đơn vị quan sát cơ bản: một review r = (h, t, T, S, M)
  h  = hotel_id (int, đã ép kiểu thống nhất)
  t  = review_date (đã parse về (year, month))
  T  = văn bản (review_text hoặc review_positive/review_negative)
  S  = review_score (có thể null với HAMoS)
  M  = metadata phụ (language, traveller_type, country, has_photo)
```

---

## 2. Method cụ thể — theo từng giai đoạn

### Giai đoạn 0 — Data unification (bắt buộc, chưa có trong sơ đồ trước)

1. Ép kiểu `hotel_id` về cùng dạng ở cả hai nguồn; xác nhận lại con số overlap = 3.387/3.399.
2. Parse `review_date` hai định dạng khác nhau (`YYYY-MM-DD` vs `DD/MM/YYYY`) về cùng một biểu diễn
   `(year, month, day)`.
3. Chạy language ID thật (fastText/langid) trên toàn bộ 678k pool — hiện tại con số en 47,6% / vi
   31,4% chỉ là **heuristic dựa trên dấu tiếng Việt**, chưa đủ tin cậy để dùng làm biến kiểm soát.
4. Đóng băng **một checkpoint encoder duy nhất** (cho cả text lẫn ảnh nếu dùng nhánh visual) và ghi
   version — chống nhiễu "model version drift" đã nêu ở lần review trước.
5. Xác định 3 tầng cohort: **anchor hotels (388)** → **large hotels (452, ≥300 review nhưng có thể
   không có gold)** → **long-tail hotels (phần còn lại của 10.555)**.

### Giai đoạn 1 — Quad Extraction (Teacher/Student bán giám sát)

- **Backbone:** tái sử dụng extractor ASQP hiện có (SVU-ASQP/AVP-ASQP), fine-tune trên 24k quad gold
  của HAMoS, giữ nguyên `train/dev/test` split gốc.
- **Baseline F1 bắt buộc trước khi mở rộng:** đo P/R/F1 (exact-match quad) trên `test` (1.320 review)
  — đây là con số tham chiếu, mọi bước sau không được làm giảm F1 này mà không giải thích được.
- **Self-training trên pool:** áp dụng cho 456.656 review pool thuộc 3.387 hotel đã khớp. Dùng bộ
  lọc pseudo-label theo hướng đã được literature xác nhận hoạt động — **pseudo-label scorer học
  được** kiểu Zhang et al. (ACL 2024, self-training cho ASQP) — **không dùng temporal-instance
  consistency làm bộ lọc** (đã đo thực nghiệm: disagreement 11,97% so với 12,11% khi xáo trộn thời
  gian → chênh lệch không đáng kể, coi như nhiễu).
- **Vòng lặp có kiểm soát:** log phân phối lớp pseudo-label mỗi vòng; điều kiện dừng = plateau
  macro-F1 trên dev, hoặc F1 lớp thiểu số (implicit aspect, taxonomy hiếm) giảm, hoặc ECE tăng.

### Giai đoạn 2 — Quad Reliability

Trọng số $w_i \in [0,1]$ cho mỗi quad dự đoán, tổng hợp từ:

$$w_i = \sigma\big(\alpha \cdot \text{conf}_i + \beta \cdot \text{agree}_i + \gamma \cdot \text{tax}_i\big)$$

- $\text{conf}_i$: log-probability sinh chuỗi của model, hiệu chỉnh bằng temperature scaling trên
  HAMoS dev.
- $\text{agree}_i$: đồng thuận giữa extractor chính và một teacher thứ hai (seed khác, hoặc một LLM
  được prompt độc lập cho cùng nhiệm vụ ASQP).
- $\text{tax}_i$: ràng buộc cứng — `taxonomy_code` dự đoán có nằm trong tập taxonomy cố định của
  HAMoS hay không (loại thẳng nếu vi phạm, không cần học).
- **Bắt buộc:** hiệu chỉnh $\alpha,\beta,\gamma$ trên một **mẫu audit nhỏ có người gán nhãn** (200–300
  quad lấy ngẫu nhiên có trọng số theo độ khó, tương tự thiết kế idea-04 trước đây) — để xác nhận
  $\text{agree}_i$ cao thật sự tương ứng với đúng, tránh circularity giữa hai model cùng lỗi hệ thống.

### Giai đoạn 3 — Temporal Encoder: tổng hợp theo (hotel, aspect, kỳ)

$$x_{h,a,t} = \Big[\ \underbrace{\textstyle\sum_i w_i \mathbb{1}[a_i=a]}_{\text{tần suất có trọng số}},\ \ \underbrace{\%\text{pos}, \%\text{neg}, \%\text{neu}}_{\text{phân phối sentiment}},\ \ \bar{e}_{a,t},\ \ \bar{S}_t,\ \ n_t,\ \ \bar{S}^{\text{global}}_t\ \Big]$$

- $\bar{e}_{a,t}$: trung bình embedding opinion term trong kỳ (encoder đã đóng băng).
- $\bar{S}_t$: điểm rating trung bình của hotel trong kỳ (nếu có).
- $\bar{S}^{\text{global}}_t$: **điểm rating trung bình toàn nền tảng** trong kỳ — dùng để trừ đi
  (detrend). Đã đo: dao động 7,97–8,41 suốt 2022–2025, gần như phẳng nhưng vẫn phải trừ tường minh
  (đã kiểm chứng: detrend làm số hotel significant giảm nhẹ 183→177, tức là ảnh hưởng thật dù nhỏ).
- **Chọn độ dài kỳ theo mật độ review:** median chỉ 20 review/hotel toàn pool → với hotel dưới
  ngưỡng, dùng **kỳ quý (quarterly)** thay vì tháng để tránh bin rỗng; chỉ hotel trong nhóm ≥300
  review mới đủ dày để dùng kỳ tháng.
- Thêm biến kiểm soát nhiễu: tỷ trọng `traveller_type`/`country` trong kỳ (giảm nhẹ rủi ro nhiễu
  "đổi thành phần khách" thay vì "đổi chất lượng hotel" — dataset không có `reviewer_id` nên đây là
  cách kiểm soát khả dụng duy nhất).

### Giai đoạn 4 — Dynamic State Modeling: có phân cấp theo hotel

**Vấn đề cần giải:** 452 hotel có ≥300 review nhưng phần lớn 10.555 hotel không đủ dữ liệu để ước
lượng riêng lẻ (median 20). Đề xuất mô hình 2 tầng, chọn 1 trong 2 phương án theo ngân sách thời
gian:

- **Phương án tối thiểu (khuyến nghị làm trước):** local-level state space đơn giản (kiểu
  exponential smoothing / Kalman filter một chiều) cho từng $(h,a)$, nhưng **tham số được co về
  (shrink) trung bình tổng hợp theo aspect toàn corpus** bằng empirical Bayes — hotel ít dữ liệu tự
  động "mượn sức mạnh" từ hotel nhiều dữ liệu cùng aspect.
- **Phương án đầy đủ:** Bayesian Structural Time Series phân cấp, prior cấp hotel lồng trong prior
  cấp aspect toàn corpus — chuẩn hơn, nhưng tốn thời gian implement hơn, nên để ở bản mở rộng sau khi
  phương án tối thiểu đã chạy được trên tập anchor (388 hotel).

$$z_{h,a,t} = f(z_{h,a,t-1}, x_{h,a,t};\ \theta_a^{\text{global}}, \theta_{h,a}^{\text{local}})$$

**Walk-forward split (bắt buộc, thay cho fit toàn chuỗi):** với chuỗi span trung bình ~38 tháng, dùng
rolling-origin: train trên $[t_0, t-h]$, dự đoán $t$, trượt cửa sổ tới hết chuỗi. Không đánh giá
forecast trên dữ liệu đã nằm trong tập fit.

### Giai đoạn 4′ — Ba đầu ra: hợp nhất Drift magnitude + Changepoint, tách riêng Forecast

Thay vì ba head cạnh tranh gradient như bản gốc, dùng lại **chính phương pháp đã kiểm chứng thành
công trên dữ liệu thật** (best-split hai mẫu + permutation null, đã cho max real *t* = 11,94 so với
max null *t* = 4,24) — nhưng áp dụng lên **residual dự báo** thay vì giá trị thô, để giải quyết mâu
thuẫn mục tiêu huấn luyện:

$$e_{h,a,t} = x_{h,a,t} - \hat{x}_{h,a,t\,|\,t-1} \quad(\text{residual dự báo 1 bước})$$

- **Changepoint:** vị trí $c^*$ tối đa hoá thống kê Welch-t giữa hai đoạn residual trước/sau $c$
  (cùng công thức đã dùng ở audit).
- **Drift magnitude:** chính là $|\bar{e}_{\text{before}} - \bar{e}_{\text{after}}|$ tại $c^*$ — hiệu
  ứng lượng hoá đi kèm khoảng tin cậy bootstrap, **không cần model riêng**.
- **Forecast $t+h$:** đánh giá độc lập bằng walk-forward, so với **naive persistence** (giá trị kỳ
  trước) và **seasonal-naive** — cả hai bắt buộc phải có trong bảng kết quả, vì với rating gần như
  phẳng (biên độ 7,97–8,41), naive persistence có khả năng đã mạnh.

### Giai đoạn 5 — Drift Reliability (kiểm soát thống kê chính thức)

1. **Permutation-null theo từng aspect riêng** — vì null threshold khác nhau rõ rệt giữa các aspect
   đã đo được: room $t_{95}=2{,}73$, breakfast $t_{95}=3{,}11$, pool $t_{95}=3{,}16$, staff
   $t_{95}=2{,}94$, clean $t_{95}=2{,}75$, bathroom $t_{95}=3{,}07$ — **không dùng chung một ngưỡng
   cho mọi aspect**.
2. **Hiệu chỉnh nhiều so sánh:** Benjamini-Hochberg FDR trên toàn lưới (hotel × aspect), báo cáo cả
   kết quả trước và sau hiệu chỉnh để minh bạch (con số 39,2% đo được là **chưa hiệu chỉnh**, dùng
   làm baseline so sánh, không dùng làm kết quả chính).
3. **Bootstrap CI** cho drift magnitude (resample review trong cửa sổ trước/sau).
4. **Cross-model check:** drift có còn ý nghĩa nếu đổi backbone/seed extractor không.
5. **Label-free agreement (khớp đúng tên bài báo):** so khớp 3 chuỗi độc lập — (i) chuỗi từ quad
   model, (ii) chuỗi `review_score` gốc, (iii) chuỗi keyword-proxy (đã dùng ở bước feasibility, base
   rate 17–68% tuỳ aspect) — kiểm định bằng permutation null. Đây là cơ chế **validate không cần
   nhãn** trên toàn bộ 678k review, và là khoảng trống chưa có tên/chưa hình thức hoá trong literature
   (gần nhất là "Beyond the Star Rating", 2026, mới chỉ làm ở mức tương quan thô).
6. **[Tuỳ chọn] Visual corroboration:** với 345 hotel photo-rich, kiểm định MMD hai mẫu (Failing
   Loudly, NeurIPS 2019) trên embedding ảnh giữa giai đoạn trước/sau changepoint đã phát hiện từ
   text, so với giai đoạn kiểm soát khớp mẫu.

---

## 3. Output đầu ra

Với mỗi cặp $(h, a)$ đủ điều kiện dữ liệu tối thiểu:

```
{
  "hotel_id": h,
  "aspect": a,
  "series": [(t, x_hat, ci_low, ci_high), ...],       # chuỗi ước lượng đã làm mượt
  "changepoint": {
    "t_star": ..., "t_stat": ..., "p_value_raw": ...,
    "p_value_fdr": ...,                                 # sau hiệu chỉnh — số chính
    "significant_after_fdr": bool
  },
  "drift_magnitude": {"value": ..., "ci_95": [..., ...]},
  "forecast": {
    "horizon": h, "point": ..., "interval_95": [..., ...],
    "mae_vs_naive_persistence": ..., "mae_vs_seasonal_naive": ...
  },
  "reliability": {
    "bootstrap_stable": bool, "cross_model_consistent": bool,
    "label_free_agreement_score": ..., "label_free_p_value": ...
  },
  "visual_corroboration": {"available": bool, "mmd_stat": null, "p_value": null}  # chỉ 345 hotel
}
```

Output tổng hợp cấp corpus: bảng thống kê mô tả (% hotel có drift đáng kể sau FDR, theo từng aspect,
theo cohort anchor/large/long-tail) + case study định tính (hotel `5937937` làm ví dụ neo: staff
giảm 45,8%→23,7% trong khi pool tăng 25,5%→39,1%, cùng thời điểm rating tổng lại tăng 7,67→8,39 —
minh chứng trực tiếp cho luận điểm "phân tích cấp aspect thấy điều mà rating tổng không thấy được").

---

## 4. Contribution và tính mới

1. **Kết hợp domain khách sạn + cấp độ aspect + kiểm định changepoint có thống kê** — chưa ai làm cả
   ba cùng lúc. Xia et al. (ACM TKDD 2020) có phương pháp changepoint cấp aspect nhưng trên sản phẩm,
   không phải khách sạn. Song et al. (J. Hospitality and Tourism Management 2022) có domain khách
   sạn và cấp aspect nhưng chỉ so sánh tĩnh trước/sau COVID, không có kiểm định changepoint liên tục.
2. **Mô hình trạng thái phân cấp theo hotel cho chuỗi sentiment cấp aspect** — không thấy trong cả
   literature NLP-drift lẫn literature hospitality-analytics đã khảo sát; giải quyết trực tiếp vấn đề
   phân phối lệch review/hotel (median 20, max 3.120) mà không phương pháp nào trong 25 bài đã khảo
   sát xử lý.
3. **Hợp nhất changepoint + drift magnitude thành một kiểm định trên residual dự báo**, tách riêng
   đánh giá forecast bằng walk-forward — giải quyết mâu thuẫn mục tiêu huấn luyện giữa "nhạy với thay
   đổi" và "mượt để dự báo tốt", đồng thời tái sử dụng chính phương pháp đã kiểm chứng hoạt động trên
   dữ liệu thật (không phải đề xuất chưa kiểm chứng).
4. **Nghi thức label-free validation hình thức hoá** (permutation null + FDR + đối chiếu 3 tín hiệu
   độc lập) — literature hiện tại (kể cả "Beyond the Star Rating", 2026) mới dừng ở tương quan thô,
   chưa có kiểm định thống kê hay tên gọi chính thức. Khoảng trống này được xác nhận trong khảo sát
   33 bài ABSA (0/33 báo cáo confidence interval, 0/33 chạy shuffled/permutation control).
5. **Trích xuất bán giám sát ở tỷ lệ 52:1** (456.656 review chưa nhãn / 8.796 review gold) trên đúng
   tập hotel đã có gold — không phải một domain adaptation chung chung mà là semi-supervision có mục
   tiêu cụ thể (phát hiện trend), nối tiếp trực tiếp dòng SVU-ASQP v6 đã có.
6. **[Phụ] Đối chiếu ảnh không đăng ký (unaligned)** cho 345 hotel — literature vision-change-
   detection có từng mảnh riêng lẻ (đối sánh ảnh không cần alignment, kiểm định phân phối embedding)
   nhưng chưa ai ghép lại để soi drift text-sentiment, theo khảo sát đã thực hiện.

---

## 5. Checklist công việc — theo đúng trạng thái dữ liệu hiện tại, có thứ tự phụ thuộc

### Phase 0 — Vệ sinh dữ liệu (bắt buộc trước khi chạy bất cứ gì)

- [ ] Đối chiếu 3.517 `direct_conflicting_review_ids` với 8.795 review đã có ngày trong HAMoS; loại
      bỏ hoặc sửa các bản ghi trùng
- [ ] Ép kiểu `hotel_id` thống nhất giữa hai nguồn (int vs str); xác nhận lại overlap = 3.387
- [ ] Parse 2 định dạng ngày về cùng chuẩn; xác nhận lại 678.147 review có ngày hợp lệ
- [ ] Chạy langid/fastText thật trên 678k pool (thay heuristic dấu tiếng Việt hiện tại)
- [ ] Chốt và ghi lại version encoder dùng cho toàn dự án (text, và ảnh nếu dùng nhánh phụ)

### Phase 1 — Trích xuất

- [ ] Đo baseline F1 của extractor hiện có (SVU-ASQP/AVP-ASQP checkpoint) trên HAMoS test (1.320
      review) — con số tham chiếu bắt buộc phải có trước khi chạy trên pool
- [ ] Cài đặt pseudo-label scorer kiểu ACL 2024 (không dùng temporal consistency — đã xác nhận vô
      hiệu)
- [ ] Chạy self-training trên 456.656 review thuộc 3.387 hotel khớp; log phân phối lớp mỗi vòng
- [ ] Lấy mẫu audit người gán nhãn 200–300 quad để hiệu chỉnh trọng số reliability $\alpha,\beta,\gamma$

### Phase 2 — Xác định cohort & tổng hợp chuỗi

- [ ] Chốt danh sách 388 anchor hotel (gold + ≥300 pool review) — bắt đầu toàn bộ pipeline ở đây
      trước khi mở rộng ra 452 hotel / toàn bộ 10.555
- [ ] Quyết định độ dài kỳ (tháng/quý) theo mật độ review từng hotel
- [ ] Tính $x_{h,a,t}$ cho từng cohort; tính $\bar{S}^{\text{global}}_t$ để detrend
- [ ] Tính tỷ trọng `traveller_type`/`country` theo kỳ làm biến kiểm soát

### Phase 3 — Mô hình hoá

- [ ] Cài đặt phương án tối thiểu (local-level + shrinkage) trước, thử trên ~20 hotel khối lượng lớn
      nhất trong tập anchor
- [ ] Cài đặt walk-forward harness (rolling-origin, span ~38 tháng)
- [ ] Cài đặt naive persistence + seasonal-naive làm baseline forecast bắt buộc
- [ ] Cài đặt kiểm định best-split trên residual dự báo (tái dùng code đã kiểm chứng ở audit, đổi
      input từ giá trị thô sang residual)

### Phase 4 — Tin cậy & kiểm định

- [ ] Cài đặt permutation-null **riêng theo từng aspect** (đã có ngưỡng đo được làm tham chiếu: room
      2,73 / breakfast 3,11 / pool 3,16 / staff 2,94 / clean 2,75 / bathroom 3,07)
- [ ] Cài đặt hiệu chỉnh Benjamini-Hochberg trên lưới hotel×aspect; báo cáo cả trước/sau hiệu chỉnh
- [ ] Cài đặt bootstrap CI cho drift magnitude
- [ ] Cài đặt label-free agreement: đối chiếu chuỗi từ model, chuỗi `review_score`, chuỗi keyword-
      proxy, kiểm định bằng permutation
- [ ] [Tuỳ chọn] Cài đặt MMD test trên embedding ảnh cho 345 hotel photo-rich

### Phase 5 — Viết bài / báo cáo

- [ ] Chạy lại probe vô hiệu hoá (mục 2.1 trong feasibility verdict) bằng **output thật của ASQP
      model** thay vì proxy `review_score`, để xác nhận temporal-instance consistency vẫn vô hiệu ở
      cấp aspect (không chỉ ở cấp rating tổng)
- [ ] Viết case study định tính cho hotel `5937937` làm ví dụ neo
- [ ] Bảng so sánh kết quả trước/sau FDR để minh bạch với reviewer
- [ ] Bảng ablation: có/không detrend, có/không stratify theo traveller_type — đo mức ảnh hưởng thật
      của từng kiểm soát nhiễu

**Thứ tự ưu tiên nếu thời gian hạn chế:** Phase 0 → Phase 1 (baseline F1 + audit nhỏ) → Phase 2 (chỉ
388 anchor hotel) → Phase 3 phương án tối thiểu → Phase 4 mục permutation-null + FDR (đây là phần
quyết định độ tin cậy của toàn bộ claim) → phần còn lại (visual, seasonal-naive, mixed-effects đầy
đủ) làm sau nếu còn thời gian.
