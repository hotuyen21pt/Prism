# PRISM — Method Specification Table (Q1 submission-ready)
### Provenance-anchored, Reliability-calibrated, Image-verified Sentiment Monitoring
**Composition-Adjusted Aspect Sentiment Drift from Semi-Supervised Multimodal Quad Extraction**

> Bản hợp nhất, 2026-08-21. Mọi con số trong tài liệu này **đã đo trực tiếp trên dữ liệu**, không có
> giá trị giả định. Module D **đã được kiểm chứng thực nghiệm end-to-end** (script:
> `scripts/probe_complaint_composition.py`); A–C là thiết kế có căn cứ đo lường nhưng chưa chạy.
> Tài liệu liên quan: `approach_temporal_trend.md` (bằng chứng cho D), `method_spec_aspect_drift.md`
> (audit đầy đủ), `method_proposal_q1.md` (lập luận novelty).

---

## 0. Tóm tắt điều hành

| Mục | Nội dung |
|---|---|
| **Bài toán** | Ước lượng **aspect-level sentiment drift** đáng tin từ ~1,15M pseudo-quad, khi không thể annotate để kiểm chứng, recall của extractor lệch theo cực tính/độ dài, và thành phần người viết review thay đổi mạnh theo thời gian |
| **Dữ liệu** | 1.949.604 review Booking.com (10.631 hotel, 2022-02→2025-03) + 23.995 gold quad multimodal (8.796 review, 9.219 ảnh, 3.399 hotel) |
| **Cửa sổ phân tích** | **2022-03 → 2025-02 (36 tháng)** |
| **Đơn vị quan sát** | segment (extraction) · (aspect, kỳ, cohort) (drift) |
| **4 module** | A: Temporal Quad Store · B: Provenance-Anchored Extraction · C: Cross-Modal Reliability · D: Dual-Bias Compositional Drift |
| **Trạng thái** | D ✅ **đã kiểm chứng** (8/13 aspect đổi kết luận) · A/B/C 📐 thiết kế |
| **Venue mục tiêu** | Information Fusion (Q1) · dự phòng IP&M (Q1) |
| **Điểm go/no-go** | Cuối Module C — nếu image verifier không đạt AUC ngưỡng, bỏ C, giữ A+B+D |

---

## 1. Bảng tổng quan module

| Module | Tên | Input | Output | Contribution | Novelty | Rủi ro | Trạng thái |
|---|---|---|---|---|---|---|---|
| **A** | Temporal Quad Store | Pool 1,95M raw + gold 23.995 quad | Corpus sạch 1,15M review có text + gold blocklist + 36 kỳ | Tài nguyên `HAMoS-Temporal` | 🟡 TB (dataset) | 🟢 Thấp | 📐 |
| **B** | Provenance-Anchored Quad Extraction | Segment + tag nguồn trường $\phi$ | Pseudo-quad $(a,c,o,s)$ + conf | Prior **bất đối xứng đã hiệu chỉnh** (93,1/64,1) áp theo từng thành phần quad | 🟡 TB-Cao | 🟢 Thấp | 📐 |
| **C** | Cross-Modal Reliability Calibration | 9.175 cặp (ảnh, review) gold; 114.988 review pool có ảnh | $w_q\in[0,1]$ cho **toàn bộ** quad, kể cả review không ảnh | Ảnh làm "annotator miễn phí" trên tập con → **hiệu chỉnh estimator text-only rồi suy rộng ra ngoài phạm vi có ảnh** | 🟢 **Cao** | 🟠 TB | 📐 |
| **D** | Dual-Bias Compositional Drift | Quad + $w$ + metadata | Chuỗi $\tilde\pi,\tilde\nu,E$ + changepoint + phân loại drift | **Tách lỗi đo của công cụ khỏi đổi thành phần tổng thể**; phân tích compositional trên thang CLR | 🟢 **Cao** | 🟢 Thấp | ✅ **kiểm chứng** |

---

## 2. Module A — Temporal Quad Store

### A.1 Input

| Nguồn | File | Quy mô | Trường dùng |
|---|---|---|---|
| Pool | `data/raw/hotel_booking_unlabeled.jsonl` | 1.949.604 dòng / 1,4 GB | `hotel_id`(str), `review_date`(VI text), `review_score`(str), `review_positive`, `review_negative`, `review_photo`(dict), `country`, `state`, `room`, `stars_rating`, `date`(ngày ở), `time_crawl` |
| Gold quad | `data/raw/{train,dev,test}.jsonl` | 8.816 / 1.895 / 1.890 segment · 23.995 quad | `instance_id`, `segment_id`, `text`, `quads[]` |
| Gold meta | `hamos-mabsa/data/metadata/reviews_with_dates.jsonl` | 8.796 | `source_review_id`, `hotel_id`(int), `review_date`(ISO), `languages`, `split` |
| Ảnh | `hamos-mabsa/data/images/` + `images.jsonl` | 9.219 file / 986 MB | `image_id`, `source_review_id`, `local_path` |

### A.2 Thao tác bắt buộc

| # | Thao tác | Chi tiết đo được | Vì sao bắt buộc |
|---|---|---|---|
| A1 | Parse ngày | Regex `ngày (\d+) tháng (\d+) năm (\d{4})`; 184/1.949.604 null | Không có định dạng ISO trong pool |
| A2 | **Cắt cửa sổ** | Bỏ 2025-03 (11.211 review, crawl dừng ngày 10) và 2022-02 (1.391, 549 hotel) | Tháng dở dang tự sinh drift giả ở biên |
| A3 | Dedup | 15.137 dòng (0,78%) trên khoá `(hotel_id, name, review_date, positive, negative)` | |
| A4 | **Gold blocklist** 🔴 | Gold ⊂ pool đã xác nhận (`H3977209_R00003`, 2022-09-05); 100% (3.399/3.399) hotel gold có trong pool | Không làm → self-training nuốt test set, mọi F1 vô hiệu |
| A5 | Ép kiểu `hotel_id` | pool `str` ↔ gold `int` | |
| A6 | Lọc text-bearing | 1.150.415/1.949.604 (**59,0%**) có text; 799.189 (41,0%) chỉ có score+title | Mẫu số của mọi tỷ lệ |
| A7 | Language ID thật | fastText/langid thay heuristic dấu tiếng Việt | Biến kiểm soát phải đáng tin |
| A8 | Đóng băng encoder | Ghi version cho cả text lẫn ảnh | Chống "model version drift" |
| A9 | Audit 73 ca rơi phủ định | ≥1 ca xác nhận (`H10933639_R00009`: gold "Nhân viên nhiệt tình"/positive ← gốc "nhân viên **không** nhiệt tình") | 5 ca kiểm thủ công: 4 đúng, 1 sai → cần chốt tần suất |

### A.3 Output

```jsonc
// temporal_quad_store/reviews.jsonl
{
  "review_uid":    "H13241654_20250202_0031",
  "hotel_id":      13241654,                    // đã ép kiểu int
  "period":        "2025-02",                   // (year, month)
  "text_pos":      "Khách sạn mới, nhân viên nhiệt tình cực kỳ",
  "text_neg":      null,
  "score":         10.0,                        // parse từ "Đạt điểm 10"
  "stratum":       ["VN", "Phòng gia đình", "L2"],   // bloc × traveller × length_bin
  "lang":          "vi",
  "has_photo":     false,
  "image_ids":     [],
  "in_gold":       false,                       // cờ blocklist
  "gold_split":    null                         // train/dev/test nếu in_gold
}
```

| Số liệu output | Giá trị |
|---|---|
| Review trong cửa sổ, có text | ~1,13M (từ 1.150.415 trừ phần ngoài cửa sổ) |
| Review có nội dung "chưa hài lòng" | **701.776** (đã đo) |
| Review có ảnh + text | 114.988 |
| Kỳ | 36 tháng |
| Hotel | 10.631 |
| Strata | **20** ô (`country_bloc`×`state`), mở rộng 60 ô nếu thêm `length_bin` |

### A.4 Cohort

| Cohort | Định nghĩa | n hotel | Dùng cho |
|---|---|---|---|
| **A-dense** | gold ∧ ≥1000 review pool | 276 | Drift cấp hotel, granularity **tháng** |
| **B-anchor** | gold ∧ ≥300 review pool | 1.221 | Drift cấp hotel, granularity **quý** |
| **T-unbiased** | **514 hotel thuộc test split** | 514 | **Cohort đánh giá drift không thiên vị** (extractor chưa từng thấy) |
| **Corpus** | toàn bộ | 10.631 | Drift cấp corpus — **nơi tín hiệu mạnh nhất** |

---

## 3. Module B — Provenance-Anchored Quad Extraction

### B.1 Cơ sở đo lường (đây là điều làm module này khác self-training thông thường)

Đo trực tiếp **trên nhãn gold**, n = 14.129 quad xác định được nguồn trường:

| Segment lấy từ | n | → positive | → neutral | → negative |
|---|---:|---:|---:|---:|
| `review_positive` | 11.733 | **93,1%** | 3,6% | 3,3% |
| `review_negative` | 2.396 | 21,7% | 14,2% | **64,1%** |

⚠️ **Bất đối xứng 93,1 / 64,1 phải mô hình hoá tường minh.** Nguyên nhân đã kiểm chứng bằng ví dụ:
khách viết nội dung tích cực trong ô "chưa hài lòng" (*"The owner was lovely although…"*,
*"However, dinner and breakfast were just great."*). Annotation gán positive ở đó là **đúng**.

### B.2 Input / Output

| | Nội dung |
|---|---|
| **Input** | Segment $x$ (mean 13 từ) + tag nguồn trường $\phi(x)\in\{\text{POS},\text{NEG}\}$ + `lang` + `length_bin` |
| **Model** | Generative ASQP (mT5/T5), fine-tune trên 8.816 segment train |
| **Output** | $\hat Q=\{(a,c,o,s)\}$ + $\text{conf}_\theta$ + cờ `provenance_conflict` |

### B.3 Công thức

$$P(s\mid x,\phi)\ \propto\ P_\theta(s\mid x)^{\lambda}\cdot P(s\mid\phi)^{1-\lambda},\qquad \lambda \text{ chọn trên dev}$$

$$P(s\mid\text{POS})=(0{,}931,\ 0{,}036,\ 0{,}033),\qquad P(s\mid\text{NEG})=(0{,}217,\ 0{,}142,\ 0{,}641)$$

| Quy tắc | Nội dung | Lý do |
|---|---|---|
| **Không dùng bộ lọc cứng** | Dùng posterior mềm | Bộ lọc cứng vứt đúng 21,7% ca "positive trong ô negative" — những ca khó và giàu thông tin nhất |
| **Taxonomy hard filter** | Loại quad có `taxonomy_code` ∉ 31 code | Ràng buộc cứng, tách khỏi score mềm |
| **Xung đột → hàng đợi audit** | $s$ mâu thuẫn mạnh với $\phi$ → không vứt, đưa vào audit | Nơi tập trung mỉa mai, phủ định, lỗi extractor |
| **Self-training** | 2–3 vòng; dừng theo dev macro-F1 hoặc ECE tăng | |
| **Curriculum ngôn ngữ** | Oversample gold tiếng Việt (gold 80,5% en vs pool ~67% non-vi-diacritic) | |

### B.4 Phạm vi taxonomy

| Nhóm | Số code | Xử lý |
|---|---|---|
| ≥300 quad gold | **14** (FAC_ROOM 5.444 … SER_SUPPORT 324) | Báo cáo **riêng từng code** |
| <300 quad gold | 17 (gồm BRA_CONSISTENCY 286, LOY_RECOMMEND 262) | **Gộp lên category cha** |
| <100 quad gold | 9 (BRA_REPUTE 22 … SER_COMM 80) | Không báo cáo riêng ở bất kỳ đâu |

⚠️ dev có `BRA_REPUTE` n=1, test có `AM_UTILITY` n=5 → **per-code F1 vô nghĩa với 17 code hiếm**.

---

## 4. Module C — Cross-Modal Reliability Calibration `[lõi novelty]`

### C.1 Vấn đề module này giải

Mọi tín hiệu text-based — model confidence, cross-model agreement, field provenance — **chia sẻ cùng
nguồn lỗi**. Hai model cùng train trên một gold sẽ cùng sai một kiểu; agreement cao **không** chứng
minh đúng. Đây là lỗ hổng circularity mà không tín hiệu text nào bịt được.

Ảnh **không chia sẻ nguồn lỗi đó** — nó là một sensor khác.

### C.2 Ba bước

| Bước | Input | Output | Ghi chú |
|---|---|---|---|
| **C1 · Học verifier** | 9.175 cặp (ảnh, review) gold + nhãn category | $V(\text{img}, c)\to[0,1]$ | Chỉ ở **mức category**, **mức review** |
| **C2 · Áp lên pool có ảnh** | 114.988 review pool có ảnh + text | Tín hiệu tin cậy **độc lập với text** | |
| **C3 · Cầu nối hiệu chỉnh** 🟢 | Đặc trưng **chỉ-dùng-text** trên tập C2 | $\hat r(q)$ áp được cho **1,03M review không ảnh** | **Đây là điểm mới thật sự** |

$$\hat r(q)=g\big(\text{conf}_\theta(q),\ \text{agree}(q),\ \phi(q),\ \ell(q),\ \text{lang}(q)\big)\ \approx\ \Pr\big[V(\text{img},c_q)=1\big]$$

$$w_q=\underbrace{\mathbb 1[\text{tax hợp lệ}]\cdot\mathbb 1[\text{span hợp lệ}]}_{\text{filter cứng}}\ \cdot\ \hat r(q)$$

### C.3 Giới hạn phạm vi (bắt buộc khai báo — đây là thứ làm claim đáng tin)

| Giới hạn | Số đo | Hệ quả |
|---|---|---|
| Đa số review chỉ 1 ảnh | **95,6%** (8.412/8.796) | Grounding ở **mức review**, không phải mức quad |
| 1 ảnh phải "gánh" nhiều aspect | **42,6%** review 1-ảnh có quad trải >1 taxonomy code | Ảnh **không** xác thực được toàn bộ quad |
| Cực tính lẫn lộn dưới 1 ảnh | **13,3%** | Ảnh khó xác thực $s$ ở mức quad |
| Review nhiều ảnh gán đúng | 349/384 = 90,9% trỏ ảnh khác nhau | Cỡ mẫu quá nhỏ để làm claim quad-level |

> **Phạm vi hợp lệ:** ảnh xác thực **kênh category $c$** và **cực tính thô ở mức review**.
> **Không** xác thực span $(a, o)$.

### C.4 Hiệu chỉnh dịch chuyển (bắt buộc — nếu bỏ thì C sụp)

| Nhóm pool có text | n | Số từ TB | Score TB |
|---|---:|---:|---:|
| **Có ảnh** | 114.988 | **56,7** | **8,90** |
| **Không ảnh** | 1.035.427 | **36,5** | **8,68** |

Gold **100%** có ảnh, pool chỉ **6,24%** → covariate shift nặng. Cầu nối C3 phải kèm **importance
weighting** theo `has_photo` propensity; báo cáo **có/không** hiệu chỉnh.

### C.5 Kiểm chứng cầu nối + go/no-go

| Kiểm tra | Ngưỡng chốt **trước** khi chạy | Nếu trượt |
|---|---|---|
| AUC của $V$ trên gold test | *(chốt trước)* | Bỏ Module C |
| Tương quan $\hat r$ ↔ human audit 300 quad | Spearman > *(chốt trước)* | Bỏ Module C |
| $\hat r$ giữ được hiệu lực sau IPW `has_photo` | Không suy giảm quá *(chốt trước)* | Báo cáo là negative result |

> **Nếu bỏ C:** lùi về A+B+D — vẫn đủ cho IP&M/KBS, chỉ mất mũi nhọn Information Fusion.
> Vì vậy **chạy C sớm**, không để cuối.

---

## 5. Module D — Dual-Bias Compositional Drift `[đã kiểm chứng]`

### D.1 Hai độ lệch, **cả hai đã chứng minh tồn tại**, cả hai **biến thiên theo thời gian**

| | Độ lệch | Bằng chứng đo được | Bản chất |
|---|---|---|---|
| **D-a** | **Recall lệch theo cực tính × độ dài** | Gold chỉ phủ **26%** văn bản gốc (5.355 cặp ghép: 23,6 vs 92,7 từ, token overlap **0,97**); phủ ô positive **42%** nhưng negative chỉ **25%** | **Lỗi đo của công cụ** |
| **D-b** | **Đổi thành phần reviewer** | reviewer VN **89,7% → 16,1%**; %có text 42,1→63,4%; độ dài 27,5→40,8 từ; hotel/tháng 549→8.535 | **Đổi thành phần tổng thể** |

> Hai thứ này **khác nhau về bản chất** và phải hiệu chỉnh riêng. Gộp chung là sai khái niệm —
> và sự phân biệt này tự nó là một đóng góp phương pháp cho temporal opinion mining.

### D.2 Ba đại lượng đo theo kỳ

| Ký hiệu | Tên | Công thức | Trả lời |
|---|---|---|---|
| $\tilde\pi_{c,t}$ | **Share of complaints** | tỷ trọng $c$ trong tổng lượt nhắc thuộc ô "chưa hài lòng" của kỳ $t$ | *Khách phàn nàn về cái gì nhiều hơn?* |
| $\tilde\nu_{c,t}$ | **Negativity rate** | $P(\text{neg}\mid \text{nhắc } c,\ t)$ | *Nhắc tới $c$ thì có chê nhiều hơn không?* |
| $E_{c,t}$ | **Opinion centroid** | $\frac{\sum_q w_q\,\psi(o_q)}{\sum_q w_q}$, $\psi$ = encoder đóng băng | *Lý do chê có đổi không?* |

**Vì sao dạng "share" chứ không phải count / điểm trung bình:**

| Thứ trôi theo thời gian | Biên độ | count | điểm TB | **share** |
|---|---|---|---|---|
| Hotel/tháng 549→8.535 | 15,6× | ❌ hỏng | ✅ | ✅ tự khử |
| %review có text 42→63% | +50% | ❌ hỏng | ✅ | ✅ tự khử |
| Độ dài 27,5→40,8 từ | +48% | ❌ hỏng | ⚠️ | ✅ tự khử |
| Recall chung của extractor | — | ❌ hỏng | ⚠️ | ✅ **triệt tiêu trong tỷ số** |
| Thành phần reviewer VN 89,7→16,1% | — | ❌ | ❌ | ⚠️ **cần D-b** |

Riêng "điểm trung bình" còn hỏng vì `review_score` gần phẳng (8,47–8,87 suốt 36 tháng) và lệch dương
nặng (điểm 10 = 44,9%) — không đủ độ phân giải.

### D.3 Ba bước xử lý

**Bước 1 — Hiệu chỉnh recall (D-a).**
$$\tilde N_{c,t,s}=\sum_{\ell}\frac{N_{c,t,s,\ell}}{\hat\rho^{\text{rec}}(s,\ell)}$$
$\hat\rho^{\text{rec}}$ ước lượng trên **tập D0** (~300 review annotate **đầy đủ**, xem §7).

**Bước 2 — Hiệu chỉnh thành phần (D-b), direct standardization.**
$$\tilde\pi_{c,t}=\sum_{g}\pi^{\text{ref}}_{g}\cdot\frac{n_{c,t,g}}{\sum_{c'}n_{c',t,g}}$$
$g$ = `country_bloc`(VN/WEST/ASIA/OTH) × `state`(5 loại khách) = **20 ô**; $\pi^{\text{ref}}$ = tỷ trọng
gộp toàn 36 tháng; ô $n<30$ bỏ qua. **Độ phủ đo được: trung bình 99,2%, thấp nhất 74,2%.**

**Bước 3 — CLR + tách mùa vụ.**
$$\text{clr}(\tilde\pi_t)_c=\log\frac{\tilde\pi_{c,t}}{\big(\prod_{c'}\tilde\pi_{c',t}\big)^{1/K}}$$

> **Vì sao bắt buộc CLR:** các $\tilde\pi_{c,t}$ **cộng lại bằng 1**. Nếu một aspect tăng thì các aspect
> khác **buộc phải** giảm → tương quan âm giả. Hồi quy từng aspect trên share là **sai thống kê**.
> CLR đưa từ simplex về không gian thực, làm mọi công cụ chuỗi thời gian tiêu chuẩn trở nên hợp lệ,
> và hiệu log-ratio **bất biến với tổng khối lượng** — đúng thứ cần khi hotel/tháng tăng 15,6×.

Sau CLR: khử hiệu ứng tháng-trong-năm, rồi ước lượng xu hướng / changepoint.

### D.4 Bốn loại drift — **không gộp thành một điểm số**

$$D^{\text{prev}}=\big|\Delta\,\text{clr}(\tilde\pi)_c\big|,\quad
D^{\text{val}}=\big|\Delta\tilde\nu_c\big|,\quad
D^{\text{sem}}=1-\cos(E_{c,t},E_{c,t-1}),\quad
D^{\text{dist}}=\mathrm{JS}(P_{c,t}\|P_{c,t-1})$$

| Loại drift | $D^{\text{prev}}$ | $D^{\text{val}}$ | $D^{\text{sem}}$ | Diễn giải |
|---|:---:|:---:|:---:|---|
| Sentiment drift | – | ✔ | – | Chê nặng hơn |
| **Opinion drift** | – | ✗ | ✔ | **Vẫn chê, nhưng lý do đổi** |
| Prevalence drift | ✔ | – | – | Được nhắc nhiều hơn |
| Emerging issue | ✔ | ✔ | ✔ | Vấn đề mới xuất hiện |

> **Phản biện công thức tổng hợp $D=\alpha D_{sent}+\beta D_{sem}+\gamma D_{vol}$:** **không dùng.**
> (i) thứ nguyên không tương thích (JS∈[0,1], $|\Delta\nu|$∈[0,2], cosine∈[0,2], log-ratio không chặn);
> (ii) $\alpha,\beta,\gamma$ **không thể hiệu chỉnh** vì không có gold drift → hyperparameter tuỳ ý;
> (iii) gộp lại là **vứt bỏ** đúng điểm mạnh của bài — khả năng *phân biệt* các loại drift.
> Thay bằng: 4 chiều tách rời, mỗi chiều một p-value riêng, phân loại theo **bảng chữ ký** ở trên.

### D.5 Suy diễn thống kê

| Bước | Nội dung |
|---|---|
| Changepoint | $c^\star=\arg\max_c \mathrm{Welch}\text{-}t(y_{1:c},\,y_{c+1:T})$ trên **residual dự báo** |
| Permutation null | **Riêng theo từng aspect** — ngưỡng khác nhau rõ rệt giữa các aspect |
| Đa so sánh | **BH-FDR** trên lưới (aspect × cohort × loại drift); báo cáo cả trước/sau |
| Khoảng tin cậy | Bootstrap **resample review trong strata** (giữ cấu trúc), $B=1000$ |
| Negative control | Xáo trộn timestamp → FPR thực nghiệm, phải ≈ $\alpha$ sau FDR |

### D.6 ✅ Kết quả kiểm chứng thực tế

Proxy keyword (lexicon sinh từ `aspect_term` gold), **701.776 review có nội dung "chưa hài lòng"**,
36 tháng, chưa dùng ASQP.

**(a) Hiệu chỉnh thành phần đổi kết luận ở 8/13 aspect**

| Aspect | THÔ slope/năm | t | ĐÃ HIỆU CHỈNH | t | Kết luận |
|---|---:|---:|---:|---:|---|
| **AM_FOOD** | +0,018 | +1,4 | **−0,053** | **−7,4** | 🟡 **bị che + đảo dấu** |
| SER_SUPPORT | +0,042 | +3,6 | −0,009 | −1,7 | 🔴 giả |
| **FAC_ENV** | −0,085 | **−4,4** | −0,008 | −1,1 | 🔴 **giả — 91% artifact** |
| AM_TRANSPORT | −0,031 | −4,8 | −0,006 | −0,6 | 🔴 giả |
| SER_ATTITUDE | +0,014 | +2,8 | +0,001 | +0,1 | 🔴 giả |
| FAC_VIEW_LOCATION | +0,031 | +3,1 | +0,023 | +1,6 | 🔴 giả |
| FAC_BATH | −0,002 | −0,4 | +0,022 | **+3,4** | 🟡 bị che |
| AM_ROOM_UTIL | −0,025 | −1,6 | +0,023 | +2,6 | 🟡 bị che |
| **AM_WIFI** | −0,121 | −6,9 | **−0,109** | **−6,7** | 🟢 **thật, bền vững** |
| AM_POOL | +0,126 | +6,0 | +0,037 | +3,5 | 🟢 thật (nhỏ hơn 71%) |
| FAC_BUILDING | +0,048 | +4,3 | +0,016 | +2,8 | 🟢 thật (nhỏ hơn 67%) |
| FAC_ROOM | +0,009 | +2,6 | +0,013 | +3,1 | 🟢 thật |
| FAC_CLIMATE | −0,024 | −0,9 | +0,053 | +2,1 | ⚪ dưới ngưỡng |

**5 giả bị loại · 3 bị che được phát hiện · 4 thật được xác nhận.**
Ca sạch nhất: `AM_FOOD` thô đi 7,67%→~10% (t=+1,4, "không có gì"); đã hiệu chỉnh đi
**12,74%→~9,6%** (t=**−7,4**). Đầu 2022 reviewer 89,7% người Việt, nhóm này phàn nàn về đồ ăn ở tỷ
trọng thấp hơn → kéo con số thô xuống giả tạo.

**(b) Xác thực ngoại vi — khôi phục đúng quy luật vật lý mà không được cho biết gì về mùa**

| Aspect | Đỉnh | Đáy | Đỉnh/đáy | Hợp lý? |
|---|---|---|---:|---|
| **FAC_CLIMATE** (điều hoà) | **T6** 2,54% | T1 1,15% | **2,22×** | ✅ đỉnh giữa hè |
| **AM_POOL** (hồ bơi) | **T8** 4,38% | T12 2,53% | **1,73×** | ✅ đỉnh mùa bơi |
| FAC_BATH (nước nóng) | T12 9,45% | T8 7,89% | 1,20× | ✅ đỉnh mùa lạnh |
| FAC_ROOM | T11 22,84% | T6 21,67% | **1,05×** | ✅ đúng là **không** có mùa |

> Đây là **bằng chứng validity mạnh nhất có được khi không có gold drift**: phương pháp tái tạo đúng
> chu kỳ đã biết trước, và cho **null phẳng đúng chỗ cần phẳng**. Nó cũng chứng minh khử mùa vụ là
> bắt buộc — biên độ mùa của điều hoà/hồ bơi lớn gấp ~4× các aspect khác.

⚠️ Bảng (a) **chưa hiệu chỉnh đa so sánh**, mới có t-stat OLS. Cần bootstrap CI + permutation null +
BH-FDR trước khi trích vào bài.

---

## 6. Bảng Contribution × Novelty

| # | Contribution | Literature hiện làm gì | Hạn chế | Ta làm gì | Novelty | Bằng chứng thực nghiệm |
|---|---|---|---|---|---|---|
| **C1** | **Cross-modal reliability calibration** | Multimodal ABSA dùng ảnh để **cải thiện extraction** | Chưa ai dùng ảnh để **ước lượng độ tin cậy pseudo-label**, càng chưa ai **suy rộng ra ngoài phạm vi có ảnh** | Ảnh = annotator miễn phí trên tập con → hiệu chỉnh estimator text-only → áp cho 1,03M review không ảnh | 🟢 **Cao** | AUC verifier; $\hat r$↔human audit; ablation −C |
| **C2** | **Dual-bias correction** | Temporal ABSA gộp mọi thay đổi thành "drift" | Không tách lỗi đo của công cụ khỏi đổi thành phần tổng thể | Hai hiệu chỉnh riêng biệt, mỗi cái có injection test riêng | 🟢 **Cao** | ✅ **8/13 aspect đổi kết luận**; injection composition + length |
| **C3** | **Provenance-anchored semi-supervision** | Self-training ASQP dùng model confidence (Zhang et al., ACL 2024) | Không có tín hiệu ngoài model; giả định đối xứng | Prior **bất đối xứng đã hiệu chỉnh trên gold** (93,1/64,1), áp theo từng thành phần quad | 🟡 TB-Cao | Ablation −B; F1 tách theo cực tính |
| **C4** | **Compositional treatment of aspect shares** | Phân tích tỷ lệ aspect bằng hồi quy trực tiếp trên share | Vi phạm ràng buộc simplex → tương quan âm giả | CLR + phân tích compositional | 🟡 TB | So CLR vs hồi quy thô trên share |
| **C5** | **Tài nguyên `HAMoS-Temporal`** | — | — | ~1,15M pseudo-quad có timestamp + reliability + ảnh, 36 tháng, 10.631 hotel; kèm giao thức đánh giá drift không cần gold | 🟡 TB | Reproducibility package |

**Luận điểm một câu cho abstract:** *chưa có công trình nào ước lượng sentiment drift cấp aspect ở quy
mô hàng triệu pseudo-label mà đồng thời (i) dùng một modality độc lập để hiệu chỉnh độ tin cậy vượt ra
ngoài phạm vi có modality đó, và (ii) tách bạch được lỗi đo của công cụ khỏi thay đổi thành phần tổng thể.*

---

## 7. Bảng Input / Output toàn hệ thống

| Cấp | Input | Output | Quy mô |
|---|---|---|---|
| **Hệ thống** | 1,95M raw review + 23.995 gold quad + 9.219 ảnh | Bảng drift có kiểm định cho (aspect × kỳ × cohort) | 36 kỳ × 6 category (+14 subcategory) × 4 cohort |
| **Module A** | raw jsonl | `temporal_quad_store/reviews.jsonl` | ~1,13M review sạch |
| **Module B** | segment + $\phi$ | pseudo-quad + conf | ~2,3M quad (ước tính 2,0 quad/review) |
| **Module C** | quad + ảnh | $w_q\in[0,1]$ | mọi quad |
| **Module D** | quad + $w$ + metadata | chuỗi + drift + p-value | xem §7.1 |

### 7.1 Output schema cuối

```jsonc
{
  "aspect":  "AM_FOOD",
  "level":   "taxonomy_code",          // hoặc "category"
  "cohort":  "T-unbiased",             // corpus | A-dense | B-anchor | T-unbiased
  "series": [
    {"period":"2022-03","pi_raw":0.0767,"pi_adj":0.1274,"clr":0.31,
     "nu_adj":0.42,"n":3120,"w_mean":0.81,"ci95":[0.118,0.137]}
  ],
  "seasonality": {"peak_month":"08","trough_month":"12","peak_trough_ratio":1.12},
  "trend": {"slope_clr_per_year":-0.053,"t_stat":-7.4,
            "p_raw":1.2e-8,"p_fdr":4.6e-8,"significant_after_fdr":true},
  "changepoint": {"t_star":"2023-05","welch_t":5.1,"p_perm":0.003,"p_fdr":0.011},
  "drift_signature": {"prevalence":true,"valence":false,"semantic":true,
                      "label":"opinion_drift"},
  "robustness": {"bootstrap_stable":true,"survives_composition_adj":true,
                 "survives_recall_adj":true,"cross_model_consistent":true,
                 "raw_vs_adj_verdict_changed":true},
  "provenance": {"n_quads":184203,"encoder_version":"...","extractor_ckpt":"..."}
}
```

---

## 8. Bảng Evaluation

| ID | Thí nghiệm | Đo cái gì | Có gold? | Ưu tiên |
|---|---|---|---|---|
| **E1** | Extraction F1 (exact-match quad) trên test 1.890 segment | Chất lượng extractor | ✅ | 🔴 |
| E1b | F1 tách theo **ngôn ngữ / độ dài / cực tính** | Nguồn của D-a | ✅ | 🔴 |
| E1c | **Chronological split probe** (train ≤2024-06 / test ≥2024-07) | Temporal generalization | ✅ | 🔴 |
| **E2** | AUC image verifier + $\hat r$ ↔ human audit + IPW `has_photo` | Module C sống hay chết | bán phần | 🔴 **go/no-go** |
| **E3a** | Injection **composition**: giữ nguyên sentiment, chỉ đổi tỷ trọng strata theo thời gian | Đề xuất **không** báo drift ↔ baseline **có** | ✅ tổng hợp | 🔴 **lõi** |
| **E3b** | Injection **length/recall**: kéo dài văn bản theo đúng xu hướng thật 27,5→40,8 | Như trên | ✅ tổng hợp | 🔴 **lõi** |
| E3c | Injection **valence / prevalence / opinion-semantic** | ROC theo $\delta\in\{2,5,10,20\}$%; minimum detectable effect | ✅ tổng hợp | 🔴 |
| **E4** | Negative control — xáo trộn timestamp | FPR thực nghiệm sau FDR | ✅ | 🔴 |
| **E5** | Cohort **train-hotel vs test-hotel** (2.381 vs 514) | **Mức lạc quan hoá của pseudo-label** | ✅ | 🟠 |
| E6 | Bootstrap stability (resample trong strata) | CI + độ bền changepoint | ✅ | 🟠 |
| E7 | Cross-model consistency (đổi backbone/seed) | Spearman giữa 2 bảng xếp hạng drift | ✅ | 🟠 |
| E8 | Rating external check (**held-out cohort, dùng 1 lần**) | Spearman kỳ vọng **0,3–0,7**, **chốt trước khi chạy** | weak | 🟠 |
| E9 | **Seasonality sanity check** | ✅ **đã đạt**: CLIMATE 2,22× (đỉnh T6), POOL 1,73× (đỉnh T8), ROOM 1,05× (phẳng) | vật lý | 🟢 **xong** |
| E10 | Event probe quanh **15/03/2022** (VN mở cửa du lịch) | Bằng chứng bổ trợ | ⚠️ trùng confounder | 🟡 |
| E11 | Human temporal micro-benchmark (~400 review, **mẫu tách biệt**) | Chiều + độ lớn drift | ✅ | 🟠 |

### Bảng Ablation

| Ablation | Chứng minh | Ưu tiên |
|---|---|---|
| **− composition adjustment (D-b)** | ✅ **đã có kết quả**: 8/13 aspect đổi kết luận | 🔴 |
| **− recall adjustment (D-a)** | Xu hướng giả do độ dài trôi | 🔴 |
| **− CLR** (hồi quy thô trên share) | Tương quan âm giả do ràng buộc simplex | 🔴 |
| **− deseasonalization** | ✅ có bằng chứng: biên độ mùa CLIMATE/POOL gấp ~4× | 🔴 |
| **− cross-modal reliability (C)** | Giá trị của Module C | 🔴 |
| **− provenance prior (B)** | Giá trị của weak supervision đặc thù | 🔴 |
| − reliability weighting ($w\equiv1$) | Uncertainty-aware aggregation có đáng công không | 🟠 |
| − self-training (chỉ gold) | Giá trị của scaling 131:1 | 🟠 |
| Strata 20 ô vs 60 ô (thêm `length_bin`) | Robustness của hiệu chỉnh | 🟠 |
| Tháng vs quý | Độ nhạy với lựa chọn thiết kế | 🟡 |

### Bảng Baseline

| Module | Baseline | Bắt buộc |
|---|---|---|
| Extraction | Fine-tune generative ASQP, **không** self-training | ✅ số tham chiếu |
| Extraction | Self-training vanilla (chỉ lọc confidence) | ✅ **baseline chính cần thắng** |
| Extraction | LLM zero-shot & few-shot | ✅ reviewer 2026 sẽ hỏi |
| Extraction | Pipeline extract-classify 2 giai đoạn; keyword/lexicon | ✅ sàn dưới |
| Drift | **Document-level rating trend** (không dùng ABSA) | ✅ **quan trọng nhất — không thắng thì bài mất lý do tồn tại** |
| Drift | **Chính phương pháp nhưng KHÔNG hiệu chỉnh** | ✅ bằng chứng trực tiếp cho C2 |
| Drift | Moving average · EWMA · CUSUM · Page-Hinkley | ✅ |
| Drift | PELT / Binary segmentation (`ruptures`) | ✅ |
| Drift | JS / Wasserstein giữa các kỳ | ✅ |
| Drift | Diachronic embedding shift | ✅ baseline cho $D^{\text{sem}}$ |

---

## 9. Bảng Rủi ro

| # | Rủi ro | Mức | Giảm thiểu | Trạng thái |
|---|---|---|---|---|
| R1 | Composition drift nhầm thành quality drift | 🔴 | D-b + E3a | ✅ **đã giải quyết & chứng minh** |
| R2 | Leakage gold ⊂ pool | 🔴 | A4 blocklist | 📐 |
| R3 | Recall lệch theo độ dài sinh drift giả | 🔴 | D-a + E3b | 📐 (cần D0) |
| R4 | Circular validation | 🔴 | Quy tắc **một-lần-một-phía**: `review_score` **chỉ** validate, không bao giờ lọc | 📐 |
| R5 | Module C thất bại | 🟠 | Go/no-go C.5; lùi về A+B+D | 📐 |
| R6 | Covariate shift ảnh (100% gold vs 6,24% pool) | 🟠 | IPW `has_photo`, báo cáo có/không | 📐 |
| R7 | Survivorship bias (hotel/tháng 549→8.535) | 🟠 | Chỉ giữ hotel có mật độ ổn định; báo cáo độ nhạy | 📐 |
| R8 | 17/31 code không đủ mẫu | 🟠 | Gộp lên category; khai báo rõ | ✅ đã quyết |
| R9 | Extractor yếu trên tiếng Việt | 🟠 | Oversample; F1 tách ngôn ngữ | 📐 |
| R10 | Rating tương quan quá cao → bài mất lý do tồn tại | 🟡 | E8 chốt ngưỡng trước; case study aspect ngược chiều rating | 📐 |
| R11 | Lỗi rơi phủ định trong gold | 🟡 | A9 audit 73 ca | 📐 |

### Giới hạn phải khai báo trong paper

1. Grounding ảnh ở **mức review**, không phải mức quad (95,6% single-image; 42,6% trải >1 code).
2. Gold **100% có ảnh** vs pool 6,24% → covariate shift, đã hiệu chỉnh nhưng không triệt tiêu.
3. Gold phủ **26%** văn bản gốc; D-a giảm nhẹ nhưng không xoá được hệ quả.
4. **Không có drift ground truth** — bằng chứng từ injection + hội tụ nhiều tín hiệu, không phải một con số.
5. Một nền tảng (Booking.com), một quốc gia → khái quát hoá hạn chế.
6. **Không có `reviewer_id`** → không loại được reviewer lặp.
7. Span 2022-02→2025-03 **không chứa cú sốc COVID** → thiếu một nguồn validation ngoại sinh mạnh.
8. Implicit aspect chỉ 3,3%, implicit opinion 1,2% → **không** làm claim về implicit.

---

## 10. Bảng Roadmap

| Phase | Việc | Output | Tuần | Chặn |
|---|---|---|---|---|
| **P0** | A1–A8 sửa dữ liệu + blocklist; A9 audit 73 ca | `temporal_quad_store/` | 1 | 🔴 chặn mọi thứ |
| **P0b** | **D0: annotate ~300 review ĐẦY ĐỦ** (toàn bộ văn bản pool) | $\hat\rho^{\text{rec}}(s,\ell)$ | 1,5 | 🔴 chặn D-a |
| **P1** | Module B seed extractor + E1/E1b/E1c | F1 tham chiếu | 1,5 | |
| **P2** | **Module C + E2 → go/no-go** ⚠️ **làm sớm** | $w_q$ hoặc quyết định bỏ C | 2,5 | 🔴 quyết định scope |
| **P3** | Module B self-training (provenance-anchored) | ~2,3M pseudo-quad | 2 | |
| **P4** | Module D full (thay keyword bằng quad có trọng số) | Chuỗi $\tilde\pi,\tilde\nu,E$ | 1,5 | ✅ khung đã chạy |
| **P5** | E3a/E3b/E3c injection + E4 + E5 + E6 | Bảng kết quả chính | 2 | |
| **P6** | Ablation + baseline | Bảng so sánh | 2 | |
| **P7** | Viết + đóng gói `HAMoS-Temporal` | Bản thảo + resource | 3 | |

**Đường găng:** P0 → P0b → P1 → **P2 (go/no-go sớm)** → P3 → P4 → P5.
**Cắt trước tiên nếu thiếu thời gian:** forecasting, BSTS phân cấp, drift cấp hotel cohort B, E10, E11.

---

## 11. Bảng Venue

| Tạp chí | Q | Phù hợp | Lý do |
|---|---|---|---|
| **Information Fusion** | Q1 | ⭐ **Tốt nhất** | Bài đúng nghĩa *fuse* text + ảnh + metadata + provenance thành ước lượng độ tin cậy; C1 là fusion đúng chủ đề lõi |
| **Information Processing & Management** | Q1 | ⭐ Rất tốt | Trích xuất quy mô lớn + phương pháp luận đánh giá; IP&M chuộng giao thức đánh giá mới |
| Knowledge-Based Systems | Q1 | Tốt | Framework tích hợp |
| Expert Systems with Applications | Q1 | Tốt | Nếu nhấn ứng dụng vận hành |
| Decision Support Systems | Q1 | Khá | Cần nhấn hàm ý quản trị |
| Int. J. of Hospitality Management | Q1 | Khá | Phải đổi trọng tâm sang domain, rút gọn method |

> **Nếu bỏ Module C** (go/no-go trượt): chuyển mục tiêu sang **IP&M / KBS** với A+B+D.
> C2 (dual-bias, đã có bằng chứng 8/13) + C3 + C4 + C5 vẫn đủ tải một bài Q1.
