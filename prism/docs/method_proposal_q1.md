# Đề xuất phương pháp cho bài báo Q1
### PRISM — Provenance-anchored, Reliability-calibrated, Image-verified Sentiment Monitoring

> Tài liệu này viết sau **audit vòng 2** (2026-08-21), khi đã có đủ `train/dev/test.jsonl` chứa quad.
> Nó bổ sung và **sửa** một số kết luận trong `method_spec_aspect_drift.md` v2.0.
> Mọi con số đều đo trực tiếp; chỗ nào là ước lượng đều ghi rõ.
> *PRISM là tên làm việc, không phải yêu cầu.*

---

## 1. Audit vòng 2 — điều gì đã thay đổi

### 1.1 Đã xác nhận: split được thiết kế rất tốt

`train/dev/test.jsonl` = 8.816 / 1.895 / 1.890 **segment-level instance**, tổng 12.601, 23.995 quad.

| Kiểm tra | Kết quả |
|---|---|
| Overlap review giữa các split | **0** |
| **Overlap hotel giữa các split** | **0** — split **hotel-disjoint** (2.381 / 504 / 514 hotel) |
| Phân bố sentiment mỗi split | pos 79,4% · neg 15,4% · neu 5,2% — **giống nhau tới 0,1%** |
| Phân bố ngôn ngữ | en ~80,5% · vi ~16,2% · other ~3,3% ở cả 3 split |
| Phân bố thời gian | 2022 ~12,4% · 2023 ~30,7% · 2024 ~46% · 2025 ~11% ở cả 3 split |
| Độ phủ taxonomy | **31/31 code có mặt ở cả 3 split** |
| Overlap text chính xác | 39 / 43 / 13 segment — đều là cụm generic ("breakfast was good") |

Đây là **stratified hotel-disjoint split** — chất lượng thiết kế cao hơn hẳn random split thường gặp.
Nó có hai hệ quả trực tiếp mà nên khai thác:

- **Đánh giá là generalization tới hotel chưa từng thấy** — đúng setting cần cho việc suy rộng ra 10.631 hotel của pool. Đây là điểm mạnh nên nói rõ trong paper.
- **Cho sẵn một cohort drift không thiên vị:** 514 hotel test chưa hề được extractor nhìn thấy. Dùng đúng nhóm này làm cohort đánh giá drift chính → pseudo-label trên đó không bị lạc quan hoá. Không cần thiết kế gì thêm, chỉ cần dùng đúng.

⚠️ Nhưng: dev có `BRA_REPUTE` **n=1**, test có `AM_UTILITY` **n=5**. **Per-code F1 cho ~9–17 code hiếm là vô nghĩa về mặt thống kê** — chỉ báo cáo ở mức category cho nhóm này.

### 1.2 Phát hiện mới #1 — Dataset là **multimodal thật**, không phải metadata trang trí

- **9.219 ảnh đã tải về đĩa (986 MB)**, không phải chỉ URL.
- **100% quad có `primary_image_id`**; 9.175 ảnh được dùng làm primary.
- 100% gold review có ≥1 ảnh.

Đây là tài sản lớn nhất và ở vòng 1 tôi đã đánh giá thấp nó. Nhưng phải kèm giới hạn đo được:

| Giới hạn | Số đo | Hệ quả |
|---|---|---|
| Hầu hết review chỉ có 1 ảnh | 8.412/8.796 = **95,6%** | Grounding thực chất ở **mức review**, không phải mức quad |
| Review nhiều ảnh thì gán đúng | 349/384 = **90,9%** trỏ tới ảnh khác nhau | Annotation có ý thức về alignment, nhưng cỡ mẫu quá nhỏ (349) để làm claim quad-level |
| 1 ảnh phải "gánh" nhiều aspect | **42,6%** review 1-ảnh có quad trải trên >1 taxonomy code | Ảnh **không thể** xác thực toàn bộ quad của review |
| Cực tính lẫn lộn dưới 1 ảnh | **13,3%** review 1-ảnh có >1 sentiment | Ảnh khó xác thực `s` ở mức quad |

> **Kết luận có kiểm chứng:** ảnh dùng được để xác thực **kênh aspect category `c`** và **cực tính thô ở mức review**, **không** dùng được để xác thực span `(a, o)` ở mức quad. Mọi claim multimodal phải giới hạn đúng phạm vi này. Nói rõ điều này chính là thứ làm cho contribution đáng tin thay vì bị reviewer bắt lỗi.

### 1.3 Phát hiện mới #2 — Gold chỉ chú thích **26%** văn bản gốc `[nghiêm trọng]`

Ghép 5.355 cặp gold↔pool theo (hotel, ngày, ≥60% token overlap):

| | Gold `review_text` | Pool `positive`+`negative` |
|---|---|---|
| Số từ trung bình | **23,6** | **92,7** |
| Trung vị | 14 | 72 |

- **Tỷ lệ gold/pool = 0,26.** Token overlap = **0,97** → gold là **tập con thật sự** của văn bản gốc, không phải bản diễn đạt khác.
- **85,6%** cặp có gold ngắn hơn >20%.
- Độ phủ **không đối xứng theo cực tính**: gold phủ **42%** trường `review_positive` nhưng chỉ **25%** trường `review_negative` (trung vị 30% vs **11%**).

**Ba hệ quả, hệ quả thứ ba là nguy hiểm nhất:**

1. 23.995 quad được chú thích trên ~1/4 lượng văn bản sẵn có. Nội dung ý kiến thật trong chính các review đó lớn hơn khoảng 4×.
2. Phân bố sentiment gold (79,4% pos) **bị lệch dương một phần do quy trình cắt**, không thuần tuý phản ánh thực tế — vì nội dung negative bị cắt nhiều hơn (25% vs 42%).
3. **Recall của extractor sẽ phụ thuộc cực tính VÀ phụ thuộc độ dài. Mà độ dài review lại drift theo thời gian (27,5 → 40,8 từ). Tức là độ lệch ước lượng tự nó thay đổi theo thời gian → sinh ra drift giả.** Đây là mối đe doạ validity trực tiếp và định lượng được, không phải lo xa.

### 1.4 Phát hiện mới #3 — Field provenance: đo lại đúng cách, và nó **bất đối xứng**

Ở vòng 1 tôi ước lượng độ tin cậy của trường pos/neg qua `review_score` (proxy). Giờ đo trực tiếp
**trên nhãn gold** (14.129 quad có xác định được nguồn trường):

| Segment lấy từ | → gold positive | → gold neutral | → gold negative |
|---|---|---|---|
| `review_positive` (n=11.733) | **93,1%** | 3,6% | 3,3% |
| `review_negative` (n=2.396) | 21,7% | 14,2% | **64,1%** |

**Đây là bản sửa quan trọng cho v2.0.** Trường positive là prior rất mạnh (93,1%); trường negative
**chỉ 64,1%** — yếu hơn nhiều so với ước lượng bằng score. Lý do đã kiểm chứng bằng ví dụ: khách
thường viết nội dung tích cực trong ô "điều chưa hài lòng" ("*The owner was lovely although…*",
"*However, dinner and breakfast were just great.*"). Annotation gán positive ở đó là **đúng**.

⚠️ Bất đối xứng 93,1% vs 64,1% **phải được mô hình hoá tường minh**. Giả định đối xứng — điều mà một
thiết kế thông thường sẽ mặc nhiên làm — sẽ bơm nhiễu vào đúng lớp thiểu số (negative) vốn là lớp
mang tín hiệu drift quan trọng nhất.

### 1.5 Phát hiện mới #4 — Gold **không phải** mẫu ngẫu nhiên của pool

Gold **100%** có ảnh; pool chỉ **6,24%** có ảnh. Review có ảnh khác biệt hệ thống:

| Nhóm (pool, có text) | n | Số từ TB | Trung vị | Score TB |
|---|---|---|---|---|
| **Có ảnh** | 114.988 | **56,7** | 39 | **8,90** |
| **Không ảnh** | 1.035.427 | **36,5** | 24 | **8,68** |

→ Có **covariate shift** giữa tập huấn luyện và tập suy luận. Phải xử lý bằng importance weighting
hoặc ít nhất phải đo và khai báo. Đây là điểm reviewer Q1 chắc chắn sẽ hỏi.

### 1.6 Một lỗi nhãn đã xác nhận (chưa định lượng được quy mô)

`H10933639_R00009`: gold segment `"Nhân viên nhiệt tình"` → quad `SER_ATTITUDE / nhiệt tình / **positive**`.
Văn bản gốc trong `review_negative`: *"nhân viên **không** nhiệt tình"*. **Từ phủ định bị rơi khi cắt
segment → nhãn bị đảo cực.**

Tôi tìm được **73 trường hợp ứng viên** (có từ phủ định ngay trước segment). Kiểm tra thủ công 5 ca:
**4 ca là annotation đúng** (từ đứng trước là liên từ đối lập: *although*, *however*), **1 ca là lỗi đảo
cực thật**. Vậy đây là lỗi **có thật nhưng tần suất thấp** — cần audit 73 ứng viên đó để chốt con số,
**không nên suy rộng khi chưa audit**.

---

## 2. Bảng tài sản & nợ

| Tài sản | Số đo | Giá trị |
|---|---|---|
| Gold ASQP multimodal | 23.995 quad · 12.601 segment · 9.219 ảnh | Hiếm — ASQP hầu hết là text-only |
| Split hotel-disjoint phân tầng | 2.381/504/514 hotel, 31/31 code mỗi split | Đánh giá generalization sạch + cohort drift không thiên vị sẵn có |
| Pool quy mô lớn có thời gian | 1.949.604 review · 10.631 hotel · 38 tháng | Đủ cho drift cấp corpus/cohort |
| Pool có ảnh | 121.584 (114.988 có cả text) | Cho phép mở rộng nhánh multimodal |
| Field provenance | POS 93,1% · NEG 64,1% (đo trên gold) | Weak label **đã hiệu chỉnh**, hiếm gặp |
| Metadata kiểm soát nhiễu | `country`,`state`,`room`,`stars_rating`,`date` | Cho phép hiệu chỉnh thành phần |

| Nợ | Số đo | Mức |
|---|---|---|
| Gold chỉ phủ 26% văn bản, lệch theo cực tính (42% vs 25%) | ratio 0,26 · overlap 0,97 | 🔴 |
| Composition drift | reviewer VN **89,7% → 16,1%**; độ dài 27,5→40,8 từ | 🔴 |
| Gold ⊄ mẫu ngẫu nhiên pool (100% ảnh vs 6,24%) | 56,7 vs 36,5 từ | 🟠 |
| Grounding ảnh chỉ ở mức review | 95,6% single-image | 🟠 |
| Code hiếm | dev `BRA_REPUTE` n=1; test `AM_UTILITY` n=5 | 🟠 |
| Leakage gold⊂pool | xác nhận vòng 1 | 🔴 (đã có cách xử lý) |
| Lỗi rơi phủ định | ≥1 ca xác nhận / 73 ứng viên | 🟡 |

---

## 3. Vì sao hướng ở v2.0 chưa đủ cho Q1

v2.0 đề xuất: semi-supervised ASQP + composition-adjusted drift + label-free validation. Chắc chắn,
nhưng ba điểm yếu với chuẩn Q1:

1. **Label-free validation vẫn chưa thật sự "label-free".** Nó dựa vào `review_score` + human audit.
   Score thì tương quan với field provenance (không độc lập); human audit thì không scale. Vẫn còn
   một lỗ hổng khái niệm chưa bịt.
2. **Không dùng ảnh** — bỏ phí đúng thứ làm dataset này khác biệt, và đúng thứ có thể bịt lỗ hổng ở (1).
3. **Chưa biết về vấn đề recall 26%** — nên chưa có cơ chế xử lý một nguồn drift giả đã được chứng minh tồn tại.

Q1 (IP&M, Information Fusion, KBS, ESWA, DSS) đòi một **framework tích hợp** giải quyết một vấn đề
được nêu rõ, cộng thực nghiệm dày. Ba mảnh còn thiếu ở trên ghép lại vừa đúng thành một framework như vậy.

---

## 4. Phương pháp đề xuất — PRISM

**Câu hỏi trung tâm:** *Làm sao ước lượng aspect-level sentiment drift đáng tin từ hàng triệu
pseudo-quad, khi (i) không thể annotate để kiểm chứng, (ii) recall của extractor lệch theo cực tính và
độ dài, và (iii) chính thành phần người viết review cũng thay đổi theo thời gian?*

Ba vế của câu hỏi này đều **đo được là có thật trên dữ liệu** — không phải khó khăn giả định. Bốn
thành phần dưới đây, mỗi thành phần trả lời một vế, vế cuối là giao thức đánh giá.

```text
                    ┌──────────────── M1 ────────────────┐
   Pool 1.15M       │  Provenance-Anchored Extraction     │
   text-bearing ───▶│  prior bất đối xứng P(s|field)      │──▶ pseudo-quads
   + field tag      │  93,1% / 64,1% — hiệu chỉnh trên gold│    + conf
                    └─────────────────────────────────────┘
                                     │
              ┌──────────────────────┴───────────────────────┐
              ▼                                              ▼
   ┌───────── M2 ──────────┐                    ┌─────────── M3 ───────────┐
   │ Cross-Modal Reliability│                    │ Dual-Bias Correction     │
   │ 115k review CÓ ẢNH:    │                    │  (a) recall theo cực tính│
   │  ảnh xác thực kênh c   │                    │      × độ dài            │
   │        ↓ CẦU NỐI       │───── w(q) ───────▶ │  (b) thành phần reviewer │
   │ hiệu chỉnh sang 1.03M  │                    │      (72 strata)         │
   │ review KHÔNG ảnh       │                    └──────────────────────────┘
   └────────────────────────┘                                 │
                                                              ▼
                                          ┌────────── M4 ──────────┐
                                          │ Drift + giao thức đánh │
                                          │ giá bằng injection     │
                                          └────────────────────────┘
```

### M1 — Provenance-Anchored Quad Extraction

Pool mang một tín hiệu mà corpus unlabeled thông thường không có: mỗi đoạn văn bản đến từ ô
"điều hài lòng" hoặc ô "điều chưa hài lòng". Đây **không phải heuristic** — nó đã được hiệu chỉnh
trên gold (§1.4), và nó **bất đối xứng**.

- Đặt $\phi(i)\in\{\text{POS},\text{NEG}\}$ là nguồn trường của đoạn $i$. Prior đã đo:
  $P(s\mid \text{POS})=(0{,}931,\ 0{,}036,\ 0{,}033)$, $P(s\mid \text{NEG})=(0{,}217,\ 0{,}142,\ 0{,}641)$
  trên $(\text{pos},\text{neu},\text{neg})$.
- Kết hợp với model qua **posterior**, không phải bộ lọc cứng:
  $$P(s\mid x,\phi)\ \propto\ P_\theta(s\mid x)^{\lambda}\cdot P(s\mid\phi)^{1-\lambda}$$
  $\lambda$ chọn trên dev. Bộ lọc cứng sẽ vứt đúng 21,7% trường hợp "positive nằm trong ô negative" —
  vốn là những ca khó và giàu thông tin nhất.
- Quad có $s$ **mâu thuẫn mạnh** với prior → **không vứt**, đưa vào hàng đợi audit. Đây là nơi tập
  trung mỉa mai, phủ định, và lỗi extractor.

**Novelty:** self-training cho ASQP hiện dùng confidence của model (Zhang et al., ACL 2024). Ở đây có
một prior **ngoài model, đo được, bất đối xứng, và chỉ phủ lên một thành phần của quad**. Việc phân rã
quad theo thành phần rồi đưa tín hiệu ngoài vào đúng thành phần mà nó nói được — đó là điểm mới, và
nó tổng quát hoá cho mọi nền tảng review có cấu trúc pro/con (Booking, TripAdvisor, Amazon).

### M2 — Cross-Modal Reliability Calibration `[lõi novelty]`

Đây là câu trả lời cho vấn đề bạn quan tâm nhất: **làm sao biết hàng triệu pseudo-quad đáng tin.**

Vấn đề của mọi tín hiệu text-based (confidence, cross-model agreement, provenance): chúng **chia sẻ
cùng nguồn lỗi**. Hai model cùng train trên một gold sẽ cùng sai một kiểu; agreement cao không chứng
minh đúng. Ảnh **không chia sẻ nguồn lỗi đó** — đó là một sensor khác.

**Bước 1 — Học verifier trên gold.** Trên 9.175 cặp (ảnh, review) gold, huấn luyện
$V(\text{image}, c)\rightarrow[0,1]$: ảnh có chứng thực rằng category $c$ được nói tới không.
Chỉ ở mức **category**, và chỉ ở mức **review** — đúng phạm vi §1.2 cho phép.

**Bước 2 — Áp lên phần pool có ảnh.** 114.988 review có cả text và ảnh → thu được tín hiệu tin cậy
**độc lập với text** cho pseudo-quad trên tập con này.

**Bước 3 — Cầu nối hiệu chỉnh (điểm mới thật sự).** Trên tập con có ảnh, fit ánh xạ từ các đặc trưng
**chỉ-dùng-text** sang xác suất được ảnh xác thực:

$$\hat r(q)=g\big(\text{conf}_\theta(q),\ \text{agree}(q),\ \phi(q),\ \ell(q),\ \text{lang}(q)\big)
\ \approx\ \Pr\big[V(\text{img},c_q)=1\big]$$

rồi **chuyển $\hat r$ sang 1.03M review không có ảnh**, nơi không thể xác thực trực tiếp.
Ảnh đóng vai một **"annotator miễn phí" trên một tập con**, và tập con đó dạy ta cách quy đổi tín hiệu
text thành ước lượng độ tin cậy có ý nghĩa — trên toàn corpus.

⚠️ **Bước hiệu chỉnh dịch chuyển (bắt buộc):** review có ảnh **khác** review không ảnh (56,7 vs 36,5 từ;
score 8,90 vs 8,68 — §1.5). Cầu nối phải kèm **importance weighting** theo `has_photo` propensity,
và phải báo cáo kết quả có/không hiệu chỉnh. Bỏ qua bước này thì cả M2 sụp.

**Kiểm chứng cầu nối (không thể thiếu):** $\hat r$ có tương quan với **human audit 300 quad** không?
Nếu có → cầu nối hợp lệ và ta có công cụ ước lượng độ tin cậy ở quy mô mà annotation không với tới.
Nếu không → báo cáo trung thực như negative result và lùi về M1+M3. Ngưỡng chấp nhận nên **chốt trước**.

**Novelty:** literature multimodal ABSA dùng ảnh để **cải thiện extraction**. Ở đây ảnh dùng để **ước
lượng độ tin cậy của pseudo-label** và — quan trọng hơn — để **hiệu chỉnh một estimator text-only rồi
suy rộng ra ngoài phạm vi có ảnh**. Tôi chưa thấy công trình nào làm việc này. Nó cũng đúng chủ đề lõi
của *Information Fusion*.

### M3 — Dual-Bias Correction

Hai độ lệch, **cả hai đều đã chứng minh tồn tại trên dữ liệu này**, và cả hai đều **biến thiên theo
thời gian** — nên cả hai đều sinh drift giả.

**(a) Hiệu chỉnh recall theo cực tính × độ dài.** Vì gold phủ 42% văn bản positive nhưng chỉ 25%
văn bản negative (§1.3), recall của extractor lệch theo cực tính. Ước lượng
$\hat\rho^{\text{rec}}(s,\ell)$ trên tập hiệu chỉnh (review được annotate **đầy đủ**, xem §7 D0), rồi
hiệu chỉnh đếm:
$$\tilde N_{c,t,s}=\sum_{\ell} \frac{N_{c,t,s,\ell}}{\hat\rho^{\text{rec}}(s,\ell)}$$
Nếu không làm: độ dài review tăng 27,5→40,8 từ theo thời gian ⇒ recall thay đổi theo thời gian ⇒
**xu hướng giả**.

**(b) Hiệu chỉnh thành phần reviewer.** Direct standardization trên 72 strata
(`country_bloc`×`state`×`length_bin`), như v2.0 §11.2. Cần vì reviewer VN đi từ **89,7% → 16,1%**.

Hai hiệu chỉnh này **khác nhau về bản chất** và phải tách riêng: (a) sửa **lỗi đo của công cụ**,
(b) sửa **đổi thành phần tổng thể**. Gộp chung là sai khái niệm. Sự phân biệt này tự nó là một đóng
góp về mặt phương pháp cho temporal opinion mining.

### M4 — Drift estimation + giao thức đánh giá

Giữ nguyên phần đã thiết kế ở v2.0 §8 và §12 (4 chiều drift tách bạch, bảng chữ ký, permutation null
riêng theo aspect, BH-FDR, bootstrap CI), cộng thêm:

- **Injection kiểm tra recall-bias:** giữ nguyên sentiment thật, chỉ **kéo dài văn bản** theo thời gian
  giống hệt xu hướng thật (27,5→40,8). Phương pháp đã hiệu chỉnh **phải không báo drift**; baseline
  **phải báo drift**. Song song với composition-injection ở v2.0.
- **Cohort đánh giá không thiên vị:** dùng **514 hotel test-split** làm cohort chính (extractor chưa
  từng thấy) và so với 2.381 hotel train-split. Chênh lệch giữa hai nhóm = **ước lượng trực tiếp mức
  lạc quan hoá của pseudo-label**. Đây là thứ mà không dataset nào khác cho làm được miễn phí, và là
  một đóng góp nhỏ nhưng rất thuyết phục về mặt đo lường.

---

## 5. Bốn contribution để tuyên bố

| # | Contribution | Novelty | Rủi ro | Bằng chứng thực nghiệm tương ứng |
|---|---|---|---|---|
| **C1** | **Cross-modal reliability calibration**: ảnh làm annotator miễn phí trên tập con, hiệu chỉnh estimator text-only rồi suy rộng ra corpus không ảnh | 🟢 Cao — chưa thấy tiền lệ | 🟠 TB (phụ thuộc verifier học được) | Tương quan $\hat r$ ↔ human audit; ablation bỏ M2 |
| **C2** | **Dual-bias correction** cho temporal opinion mining: tách hiệu chỉnh recall công cụ khỏi hiệu chỉnh thành phần tổng thể | 🟢 Cao — cả hai đã chứng minh cần thiết bằng số đo | 🟢 Thấp | Hai injection test cho câu trả lời nhị phân sạch |
| **C3** | **Provenance-anchored semi-supervision** với prior bất đối xứng đã hiệu chỉnh (93,1%/64,1%), áp theo từng thành phần quad | 🟡 TB-Cao | 🟢 Thấp | Ablation bỏ M1; F1 theo cực tính |
| **C4** | **Tài nguyên**: HAMoS-Temporal — pseudo-quad có timestamp + reliability + ảnh, trên 38 tháng / 10.631 hotel; kèm giao thức đánh giá drift không cần gold | 🟡 TB | 🟢 Thấp | Bản thân dataset + reproducibility package |

**Luận điểm một câu cho abstract:** *chưa có công trình nào ước lượng sentiment drift cấp aspect ở quy
mô hàng triệu pseudo-label mà đồng thời (i) dùng một modality độc lập để hiệu chỉnh độ tin cậy vượt ra
ngoài phạm vi có modality đó, và (ii) tách bạch được lỗi đo của công cụ khỏi thay đổi thành phần tổng thể.*

---

## 6. Venue

| Tạp chí | Q | Độ phù hợp | Lý do |
|---|---|---|---|
| **Information Fusion** | Q1 (IF cao) | ⭐ **Tốt nhất** | Bài đúng nghĩa *fuse* text + ảnh + metadata + provenance thành ước lượng độ tin cậy. C1 là fusion đúng chủ đề lõi |
| **Information Processing & Management** | Q1 | ⭐ Rất tốt | Trích xuất quy mô lớn + phương pháp luận đánh giá; IP&M chuộng bài có giao thức đánh giá mới |
| **Knowledge-Based Systems** | Q1 | Tốt | Framework tích hợp |
| **Expert Systems with Applications** | Q1 | Tốt | Nếu nhấn ứng dụng vận hành khách sạn |
| **Decision Support Systems** | Q1 | Khá | Cần nhấn mạnh hàm ý quản trị nhiều hơn |
| **Int. J. of Hospitality Management** | Q1 | Khá | Cần đổi trọng tâm sang domain; phần method sẽ phải rút gọn |

**Khuyến nghị: Information Fusion, phương án dự phòng IP&M.**

---

## 7. Thực nghiệm cần chạy

**D0 — Tập hiệu chỉnh recall `[bắt buộc, chặn M3a]`**
Annotate **đầy đủ** ~300 review (toàn bộ văn bản pool, không cắt) → ước lượng
$\hat\rho^{\text{rec}}(s,\ell)$. Đây là công việc annotation mới **không thể tránh** — nó là thứ duy
nhất cho phép hiệu chỉnh recall. Chọn mẫu phân tầng theo độ dài × cực tính.

**D1 — Sửa dữ liệu.** Gold blocklist khỏi pool; audit 73 ứng viên rơi phủ định; cắt cửa sổ
2022-03→2025-02; dedup 15.137 dòng.

**E1 — Extraction.** F1 exact-match trên test (1.890 segment); tách theo ngôn ngữ / độ dài / cực tính;
chronological split probe; per-category (không per-code cho 17 code hiếm).

**E2 — Cross-modal verifier.** AUC của $V$ trên gold test; tương quan $\hat r$ ↔ human audit 300 quad;
**có/không importance weighting** cho `has_photo`.

**E3 — Injection (đánh giá lõi).** Bốn loại: valence · prevalence · opinion-semantic · **composition** ·
**length/recall**. Hai loại cuối cho câu trả lời nhị phân: phương pháp đề xuất *không* báo drift ↔
baseline *có* báo.

**E4 — Negative control.** Xáo trộn timestamp → FPR thực nghiệm sau FDR.

**E5 — Train-vs-test hotel cohort.** Đo mức lạc quan hoá của pseudo-label.

**E6 — Ablation.** −M1 · −M2 · −M3a · −M3b · −deseasonalization · −reliability weighting · chỉ-gold.

**E7 — Baselines.** Document-level rating trend (**quan trọng nhất** — không thắng nó thì bài mất lý do
tồn tại) · LLM zero/few-shot · self-training vanilla · CUSUM/Page-Hinkley · PELT · JS/Wasserstein ·
diachronic embedding shift.

---

## 8. Giới hạn phải khai báo (không giấu — đây là điều tạo uy tín)

1. Grounding ảnh ở **mức review**, không phải mức quad (95,6% single-image; 42,6% review trải >1 code).
2. Gold **100% có ảnh** trong khi pool 6,24% → covariate shift; đã hiệu chỉnh nhưng không triệt tiêu.
3. Gold phủ **26%** văn bản gốc; M3a giảm nhẹ nhưng không xoá được hệ quả.
4. **Không có drift ground truth**; bằng chứng đến từ injection + hội tụ nhiều tín hiệu, không phải một
   con số duy nhất.
5. Một nền tảng (Booking.com), một quốc gia → tính khái quát hạn chế.
6. **Không có `reviewer_id`** → không loại được reviewer lặp.
7. 17/31 taxonomy code không đủ mẫu để báo cáo riêng.
8. Span 2022-02→2025-03 **không chứa cú sốc COVID** → thiếu một nguồn validation ngoại sinh mạnh.

---

## 9. Roadmap

| Giai đoạn | Việc | Thời lượng ước tính | Chặn |
|---|---|---|---|
| **P0** | D1 sửa dữ liệu + blocklist + audit 73 ca phủ định | 1 tuần | 🔴 chặn mọi thứ |
| **P0b** | **D0 annotate 300 review đầy đủ** | 1,5 tuần | 🔴 chặn M3a |
| **P1** | Seed extractor + E1 | 1,5 tuần | |
| **P2** | M1 provenance-anchored self-training | 2 tuần | |
| **P3** | M2 verifier + cầu nối + E2 | 2,5 tuần | ⚠️ rủi ro cao nhất — làm sớm |
| **P4** | M3 dual-bias + aggregation | 1,5 tuần | |
| **P5** | M4 drift + E3/E4/E5 | 2 tuần | |
| **P6** | E6 ablation + E7 baselines | 2 tuần | |
| **P7** | Viết + đóng gói tài nguyên | 3 tuần | |

**Đường găng:** P0 → P0b → P1 → **P3 (làm sớm để biết C1 có sống không)** → P2 → P4 → P5.

> **Điểm quyết định (go/no-go) đặt ở cuối P3.** Nếu verifier ảnh không đạt AUC hợp lý trên gold test,
> **bỏ C1** và lùi về bài dựa trên C2+C3+C4 — vẫn đủ cho IP&M/KBS, chỉ mất mũi nhọn Information Fusion.
> Chốt ngưỡng AUC **trước khi chạy**.
