# Method Spec — Aspect-Level Sentiment Drift in Hotel Reviews
### Semi-Supervised Quad Extraction at Scale with Label-Free Validation

> **Phiên bản 2.0 — viết lại sau full data audit ngày 2026-08-21.**
> Bản 1.0 được viết dựa trên pool `hotel_booking_data2.jsonl` (678.219 dòng). Pool hiện tại
> (`hotel_booking_unlabeled.jsonl`) có **1.949.604 dòng / 10.631 hotel** — gần **2,9×** bản cũ.
> **Mọi con số định lượng trong bản 1.0 đã lỗi thời và đã được thay bằng số đo lại.**
> Mọi phát biểu dưới đây đều kèm số đo thực tế; chỗ nào là suy đoán/ước lượng đều được đánh dấu rõ.
>
> **📌 Vòng 2 (2026-08-21):** sau khi có `train/dev/test.jsonl` chứa quad, một số kết luận đã được sửa
> (đánh dấu "vòng 2" trong văn bản) và bổ sung **B5, B6** dưới đây. Đề xuất phương pháp mới cho bài Q1
> nằm ở tài liệu riêng: **`method_proposal_q1.md`**.

---

## 1. Executive assessment

**Kết luận: `Feasible with modifications` — nhưng phải sửa 4 vấn đề mang tính chặn (blocking) trước khi chạy bất kỳ experiment nào.**

Dữ liệu đủ mạnh một cách bất thường cho hướng nghiên cứu này: 1,95M review có timestamp trên 38 tháng
liên tục, 23.995 quad gold trên cùng tập hotel, taxonomy 2 tầng 6 category / 31 subcategory. Đây là
quy mô hiếm trong literature temporal ABSA. Nhưng bốn phát hiện sau **thay đổi bản chất bài toán**, không
chỉ là chi tiết kỹ thuật:

| # | Phát hiện | Bằng chứng đo được | Hệ quả |
|---|---|---|---|
| **B1** | **Reviewer-population drift lấn át quality drift** | Tỷ lệ reviewer Việt Nam: **89,7% (2022-03) → 16,1% (2025-01)**. Tỷ lệ review có text: **42,1% → 63,4%**. Độ dài trung bình: **27,5 → 40,8 từ** | Drift cấp corpus đo theo cách ngây thơ sẽ **đo sự đổi thành phần khách, không phải đổi chất lượng khách sạn**. Detrend bằng rating trung bình toàn nền tảng (bản 1.0) **không đủ** — nó chỉ khử mức, không khử thành phần |
| **B2** | **Gold ⊂ Pool → leakage** | Gold review `H3977209_R00003` (2022-09-05) xuất hiện nguyên văn trong pool. **100% (3.399/3.399)** hotel gold nằm trong pool | Self-training trên pool sẽ nuốt cả **test set**. Mọi F1 báo cáo sau self-training đều vô hiệu nếu không loại trừ tường minh |
| **B3** | **Field provenance là weak label mạnh nhưng BẤT ĐỐI XỨNG** | *Đo lại vòng 2 trên nhãn gold (14.129 quad), thay cho ước lượng bằng score ở vòng 1:* segment từ `review_positive` → gold positive **93,1%**; segment từ `review_negative` → gold negative **chỉ 64,1%** (21,7% lại là positive — khách viết nội dung tích cực trong ô negative) | (a) Độ khó thật nằm ở **(a, c, o)**; (b) `review_score` **KHÔNG** độc lập với polarity field → **circular validation**; (c) **bất đối xứng 93,1/64,1 phải mô hình hoá tường minh**, không giả định đối xứng |
| **B4** | **Survivorship / truncation bias theo thời gian** | Số hotel/tháng: **549 (2022-02) → 8.535 (2025-02)**. 549 hotel có review đầu tiên ở đúng 2022-02, 2.025 hotel ở 2022-03 | Crawl một lần (2025-03) chỉ giữ lại N review gần nhất/hotel → dữ liệu 2022 là **mẫu sống sót**, không phải mẫu ngẫu nhiên của 2022 |
| **B5** *(vòng 2)* | **Gold chỉ chú thích 26% văn bản gốc, lệch theo cực tính** | 5.355 cặp ghép: gold **23,6 từ** vs pool **92,7 từ**, token overlap **0,97** (gold là tập con thật sự). Phủ trường positive **42%** nhưng negative chỉ **25%** | Recall extractor lệch theo cực tính **và** theo độ dài; mà độ dài review drift 27,5→40,8 từ ⇒ **độ lệch tự thay đổi theo thời gian ⇒ sinh drift giả**. Cần hiệu chỉnh recall (xem `method_proposal_q1.md` M3a) |
| **B6** *(vòng 2)* | **Gold không phải mẫu ngẫu nhiên của pool** | Gold **100%** có ảnh, pool chỉ **6,24%**. Review có ảnh: **56,7 từ / score 8,90** vs không ảnh **36,5 từ / 8,68** | Covariate shift train→inference. Phải importance-weight theo `has_photo` propensity và báo cáo có/không hiệu chỉnh |

**Đề xuất trọng tâm:** biến B1 từ điểm yếu thành **contribution chính**. Bài toán đúng không phải
"đo drift" mà là **"đo drift đã hiệu chỉnh thành phần" (composition-adjusted aspect drift)** — tách
`ai-nói` khỏi `nói-gì`. Literature temporal ABSA hầu như bỏ qua điều này; dữ liệu ở đây bắt buộc phải xử lý.

---

## 2. Data audit — số đo thực tế

### 2.1 Pool chưa gán nhãn — `data/raw/hotel_booking_unlabeled.jsonl`

**1.949.604 review · 10.631 hotel · 1,4 GB · 0 dòng JSON lỗi.**

| Trường | Kiểu | Độ phủ | Giá trị cho nghiên cứu drift |
|---|---|---|---|
| `hotel_id` | **str** | 100% | Đơn vị phân tích cấp hotel. **Lưu ý: gold dùng `int` → phải ép kiểu khi join** |
| `review_date` | str tiếng Việt `"Ngày đánh giá: ngày D tháng M năm YYYY"` | 99,99% (184 null) | **Trục thời gian chính.** Phải parse regex; không có định dạng ISO |
| `review_score` | str `"Đạt điểm 9,0"` | 100% | Weak signal + biến detrend. **Dùng dấu phẩy thập phân.** Lệch dương mạnh: điểm 10 chiếm **44,9%** |
| `review_positive` | str \| null | 57,6% có text | Text nguồn **+ nhãn polarity ngầm** (xem B3) |
| `review_negative` | str \| null | 36,2% có text | Text nguồn **+ nhãn polarity ngầm**. Nguồn negative quad chính |
| `review_title` | str | 99,9% | Ngắn, phần lớn generic ("Tuyệt vời") — giá trị thấp |
| `state` | str | 99,7% | **Loại khách**: Cặp đôi 40,8% · Gia đình 24,0% · Khách lẻ 18,9% · Nhóm 16,0%. **Biến kiểm soát confounder chủ lực** |
| `country` | str | 99,7% | **Quốc tịch reviewer**: VN 29,9% · Pháp 6,8% · Úc 6,6% · Đức 6,5% · Anh 6,4%. **Biến kiểm soát confounder quan trọng nhất** (xem B1) |
| `room` | str | 99,7% | Loại phòng — cho phép kiểm soát "đổi mix sản phẩm" thay vì "đổi chất lượng" |
| `stars_rating` | float \| null | 89,9% | Hạng sao hotel — biến phân tầng (stratify) |
| `review_photo` | dict `{url: caption}` | **6,24%** (121.584 review) | Nhánh multimodal tuỳ chọn. *Bản 1.0 ghi 11,5% — không còn đúng với pool mới* |
| `date` | str `"1 đêm · Tháng 2/2025"` | 99,7% | **Ngày lưu trú** (khác ngày review) — cho phép tính độ trễ review, kiểm soát seasonality theo mùa lưu trú thật |
| `time_crawl` | str | 100% | Tất cả ~2025-03 → xác nhận đây là **crawl một lần**, nguồn gốc của B4 |
| `name` | str | 99,7% | Tên reviewer. **Không có `reviewer_id`** → không track được reviewer lặp |
| `source_file` | str | 100% | Truy vết nguồn |

**Không tồn tại trong dataset (phải nói rõ với reviewer):** `reviewer_id`, city/địa lý hotel dạng
cấu trúc, giá phòng, thông tin renovation/đổi chủ, review response của hotel, helpfulness/vote,
platform khác Booking.com.

#### Chất lượng thời gian

- **Span: 2022-02 → 2025-03 (38 tháng liên tục, không có tháng rỗng).**
- Phân bố năm: 2022 **16,63%** · 2023 **32,82%** · 2024 **40,86%** · 2025 **9,68%**.
- **2025-03 chỉ có 11.211 review** (crawl dừng ngày 10) → **phải cắt bỏ tháng cuối**, nếu không sẽ tạo một "drift" giả ở biên.
- Duplicate chính xác: **15.137 dòng (0,78%)** trên khoá `(hotel_id, name, review_date, positive, negative)` — thấp, chỉ cần dedup một lần.
- Review/hotel: mean **183,4** · median **66** · p90 **457** · max **8.599** → lệch phải rất mạnh.
- Số tháng riêng biệt/hotel: mean **19,7** · median **19** · max 37. Chỉ **962 hotel** phủ ≥36 tháng; **4.025 hotel** phủ ≥24 tháng.

#### Kết luận về temporal granularity

| Cấp phân tích | Granularity khuyến nghị | Lý do (số đo) |
|---|---|---|
| **Corpus-level** | **Tháng** | ~60.5k quad/tháng ước tính — dư sức, kể cả subcategory hiếm |
| **Cohort-level** (nhóm hotel) | **Tháng** | Cohort ≥50 hotel luôn đủ dày |
| **Hotel-level, hotel ≥1000 review** (323 hotel) | **Tháng** | ~26 review/tháng → FAC_ROOM ~7,5 quad/tháng — vừa đủ |
| **Hotel-level, hotel 300–1000 review** (1.370 hotel) | **Quý** | ~8 review/tháng → FAC_ROOM ~2,3 quad/tháng, quá thưa cho tháng |
| **Hotel-level, hotel <300 review** (8.938 hotel) | **Không phân tích riêng lẻ** | Median 66 review / 38 tháng = 1,7 review/tháng. Chỉ dùng qua partial pooling |

**Không dùng granularity tuần** ở bất kỳ cấp nào dưới corpus — và ngay cả corpus-level, tuần làm
seasonality (chu kỳ ngày trong tuần, kỳ nghỉ) lấn át tín hiệu.

### 2.2 Tập gold — `hamos-mabsa/data/`

⚠️ **Cập nhật vòng 2 (2026-08-21):** `tabsa/data/raw/` nay **đã có** `train.jsonl` (8.816),
`dev.jsonl` (1.895), `test.jsonl` (1.890) — segment-level instance **có quad đầy đủ**, trùng khớp với
`hamos-mabsa/data/splits/single_image/`. Cảnh báo cũ đã được giải quyết.
File `hotel_absa_labeled.jsonl` (8.796 dòng) vẫn **không chứa quad** — nó chỉ là bảng metadata
review/ngày/ngôn ngữ; **không dùng nó làm nguồn nhãn**.

**Split là hotel-disjoint + phân tầng** (kiểm chứng vòng 2): 2.381/504/514 hotel, overlap hotel giữa
các split = **0**; sentiment/ngôn ngữ/thời gian giống nhau tới 0,1%; 31/31 taxonomy code có mặt ở cả
3 split. ⚠️ nhưng dev có `BRA_REPUTE` n=1, test có `AM_UTILITY` n=5 → **per-code F1 vô nghĩa với ~17
code hiếm**, chỉ báo cáo ở mức category.

**23.995 quad · 12.601 segment · 8.796 review · 3.399 hotel · 9.219 ảnh (986 MB, đã tải về đĩa).**

**Dataset là multimodal thật** *(vòng 2)*: **100% quad có `primary_image_id`**, 9.175 ảnh được dùng làm
primary. Nhưng grounding thực chất ở **mức review**, không phải mức quad: **95,6%** review chỉ có 1 ảnh;
**42,6%** review 1-ảnh có quad trải trên >1 taxonomy code; **13,3%** có cực tính lẫn lộn dưới cùng 1 ảnh.
→ Ảnh dùng được để xác thực **kênh category `c`** và cực tính thô ở mức review, **không** dùng được để
xác thực span `(a, o)` ở mức quad.

| Chiều | Số đo | Nhận định cho drift |
|---|---|---|
| Category (6) | FACILITY **55,1%** · AMENITY **20,5%** · SERVICE **10,7%** · EXPERIENCE **8,3%** · LOYALTY **3,2%** · BRANDING **2,2%** | Cả 6 đều đủ dữ liệu cho drift cấp corpus |
| Subcategory (31) | FAC_ROOM **22,7%** (5.444) → BRA_REPUTE **0,09%** (22) | **Đuôi dài nghiêm trọng.** 9/31 code có <100 quad; 17/31 code có <300 quad gold |
| Sentiment | positive **79,4%** · negative **15,4%** · neutral **5,2%** | Mất cân bằng nặng. Neutral (1.248) sẽ là lớp yếu nhất của extractor |
| `aspect_implicit` | **3,3%** (782) | **Rất thấp** so với chuẩn ACOS (~30%). Dataset thiên về explicit |
| `opinion_implicit` | **1,2%** (283) | Cực thấp — implicit opinion **không đủ để làm claim riêng** |
| Quad/review | mean **2,73** · median 2 · max 36 | Cơ sở để ước lượng sản lượng quad trên pool |
| Quad/segment | mean **1,90** · median 1 · max 23 | |
| Ngôn ngữ | en **7.473** · vi **1.032** · other **205** · đa ngữ **86** | **Nghịch đảo so với pool!** (pool ~30% có dấu tiếng Việt) |
| Split (review) | train 6.156 / dev 1.320 / test 1.320 | |
| Split (segment) | train 8.816 / dev 1.895 / test 1.890 = 12.601 ✓ | Nhất quán |
| Thời gian gold | 38 tháng, 8.795/8.796 có ngày | 2022-02: 4 review → 2025-01: 549 review |

**Khả thi của subcategory-level drift:** *khả thi ở cấp corpus, không khả thi ở cấp hotel.*
Với ~2,3M pseudo-quad ước tính, ngay cả code hiếm như `BRA_REPUTE` vẫn được ~54 quad/tháng corpus-wide.
Nhưng gold chỉ có 22 mẫu để huấn luyện code đó → **extractor sẽ không học nổi**. Đây là nút thắt
thật: **giới hạn không nằm ở mật độ thời gian mà ở dữ liệu huấn luyện của lớp hiếm.**

> **Quyết định scoping:** làm drift cho **6 category + 14 subcategory có ≥300 quad gold**
> (FAC_ROOM, FAC_VIEW_LOCATION, AM_POOL, FAC_BUILDING, SER_ATTITUDE, AM_FOOD, FAC_BATH,
> EXP_OVERALL, AM_ROOM_UTIL, FAC_ENV, LOY_RETURN, FAC_INTERIOR, EXP_VALUE, SER_SUPPORT).
> 17 code còn lại (kể cả BRA_CONSISTENCY 286, LOY_RECOMMEND 262 sát ngưỡng): gộp lên category cha, báo cáo là giới hạn.

### 2.3 Overlap gold ↔ pool

- **3.399/3.399 (100%)** hotel gold có mặt trong pool.
- **1.249.424 review pool** (64,1%) thuộc hotel có gold → đây là universe làm việc.
- Cohort:

| Cohort | Định nghĩa | Số hotel |
|---|---|---|
| **A — Anchor-dense** | có gold **và** ≥1000 review pool | **276** |
| **B — Anchor** | có gold **và** ≥300 review pool | **1.221** |
| **C — Anchor-wide** | có gold **và** ≥100 review pool | **2.313** |
| **D — Pool-large** | ≥300 review pool (không cần gold) | 1.693 |
| **E — Toàn pool** | tất cả | 10.631 |

*Bản 1.0 ghi "388 anchor hotel" — với pool mới con số đúng là **1.221** (≥300) hoặc **276** (≥1000).*

---

## 3. Data suitability for temporal ASQP

| Yêu cầu | Trạng thái | Bằng chứng |
|---|---|---|
| Timestamp mọi review | ✅ | 99,99% |
| Span đủ dài | ✅ | 38 tháng liên tục |
| Mật độ đủ cho aggregation | ✅ corpus/cohort · ⚠️ hotel · ❌ hotel×subcategory×tháng | xem §2.1 |
| Gold quad để train extractor | ✅ | 23.995 quad |
| Gold trên cùng domain/hotel | ✅ | overlap 100% |
| Metadata kiểm soát confounder | ✅ hiếm có | `country`, `state`, `room`, `stars_rating`, `date` (ngày ở) |
| Weak signal độc lập | ⚠️ | `review_score` **không độc lập** với polarity field (B3) |
| Gold cho drift | ❌ | **Không tồn tại và không thể có** — xem §12 |
| Sự kiện ngoại sinh để validate | ⚠️ | Span 2022-02→2025-03 **bỏ lỡ cú sốc COVID chính**; VN mở cửa hoàn toàn 2022-03-15 nằm ngay đầu chuỗi |

**Điểm quan trọng ít ai nhận ra:** span bắt đầu 2022-02 nghĩa là dataset **không** chứa cú sốc COVID.
Điều này **tốt** (không bị một sự kiện khổng lồ chi phối mọi kết quả) nhưng cũng **mất** một nguồn
validation ngoại sinh mạnh. Sự kiện dùng được duy nhất: **VN mở cửa du lịch quốc tế 15/03/2022** —
và nó khớp chính xác với B1 (làn sóng khách quốc tế đẩy tỷ lệ reviewer VN từ 89,7% xuống).
**Đây vừa là confounder lớn nhất vừa là cơ hội validation tốt nhất.**

---

## 4. Vấn đề & khoảng trống trong dữ liệu

### 4.1 Blocking (phải xử lý trước mọi experiment)

1. **B2 — Leakage gold↔pool.** Xây `blocklist` gồm hash chuẩn hoá của toàn bộ 8.796 gold review
   (và cả `(hotel_id, review_date, prefix text)`), loại khỏi pool **trước** khi self-training.
   Báo cáo số dòng bị loại. Không làm bước này → mọi F1 sau self-training vô nghĩa.
2. **B1 — Composition drift.** Bắt buộc mọi ước lượng drift phải **hiệu chỉnh thành phần** (§11).
3. **B4 — Truncation bias.** Hoặc (a) giới hạn phân tích vào cửa sổ hotel thực sự quan sát được, hoặc
   (b) chỉ dùng hotel có mật độ ổn định qua thời gian. **Khuyến nghị (b)**: chỉ giữ hotel có ≥1
   review ở ≥80% số tháng trong cửa sổ phân tích.
4. **Cắt 2025-03** (11.211 review, crawl dở dang) và cân nhắc cắt 2022-02 (1.391 review, 549 hotel).
   → **Cửa sổ phân tích chốt: 2022-03 → 2025-02 (36 tháng chẵn).**

### 4.2 Non-blocking nhưng phải khai báo

5. **41,0% review không có text nào** (799.189 dòng chỉ có score+title). Không dùng được cho ASQP.
   **Universe text thật = 1.150.415 review (59,0%)**. Và tỷ lệ này **tự nó drift** (42%→63%) — chính là B1.
6. **Language mismatch gold↔pool**: gold 85% English, pool ~30% có dấu tiếng Việt. Extractor huấn luyện
   trên gold sẽ **yếu hơn trên phần tiếng Việt của pool**. Cần đo F1 tách theo ngôn ngữ trên gold test.
7. **`review_text` gold có dấu hiệu ghép lossy** từ `positive`+`negative` (so sánh
   `H3977209_R00003` với dòng pool tương ứng: gold mất cụm "giá ok", chèn "mà đến phòng không có, hơi
   thất vọng"). **Cần audit thủ công ~50 mẫu** để xác nhận mức độ; nếu phổ biến thì có
   train/inference distribution mismatch (gold = văn bản ghép, pool = 2 field tách).
8. Language ID hiện chỉ là heuristic dấu tiếng Việt → chạy fastText/langid thật trên 1,15M review có text.
9. **Không có `reviewer_id`** → không loại được reviewer lặp, không mô hình hoá được reviewer effect.

---

## 5. Review phương pháp hiện tại (bản 1.0)

**Giữ lại (đúng và có giá trị):**
- Permutation-null **riêng theo từng aspect** — đúng về mặt thống kê, ngưỡng đo được khác nhau rõ rệt.
- **BH-FDR** trên lưới hotel×aspect — bắt buộc, giữ nguyên.
- Bootstrap CI cho drift magnitude.
- Đóng băng **một checkpoint encoder duy nhất** — chống "model version drift", chi tiết tinh tế và đúng.
- Kết luận **temporal-instance consistency vô hiệu làm bộ lọc** (11,97% vs 12,11% khi shuffle) — negative
  result có giá trị, giữ và báo cáo.
- Tách **forecast** ra đánh giá riêng bằng walk-forward + naive persistence baseline.
- Hợp nhất changepoint & drift magnitude thành một kiểm định trên residual.

**Phải sửa (có lỗi thực chất):**

| Vấn đề trong bản 1.0 | Phản biện | Sửa |
|---|---|---|
| **Circular validation trong §2.5 "label-free agreement"** | Đối chiếu 3 chuỗi: (i) quad model, (ii) `review_score`, (iii) keyword-proxy. Nhưng **(i) được huấn luyện/lọc bằng chính text mà polarity của nó đã được `review_score` xác định** (B3: pos-only → 9,52; neg-only → 5,33), và (iii) chạy trên cùng text đó. **Ba chuỗi này không độc lập.** Sự đồng thuận cao là tất yếu, không phải bằng chứng | Phân tách vai trò: `review_score`/polarity-field chỉ dùng **một lần**, ở **một phía**. Xem §10 |
| **Detrend bằng $\bar S^{global}_t$ là không đủ** | Rating toàn nền tảng gần phẳng (8,47–8,87). Nhưng thành phần khách đổi cực mạnh (VN 89,7%→16,1%). Trừ một hằng số không khử được composition shift | Thay bằng **post-stratification / IPW** trên `country`×`state`×`has_text`×`length_bin` (§11.2) |
| **Mọi con số cohort đã lỗi thời** | 388 anchor → thực tế **1.221**; pool 678k → **1,95M**; photo 11,5% → **6,24%**; hotel 10.555 → **10.631** | Đã cập nhật toàn bộ |
| **Không đề cập leakage gold⊂pool** | Đây là lỗ hổng nghiêm trọng nhất chưa được nhận diện | §4.1 mục 1 |
| **Không đề cập 41% review không có text** | Ảnh hưởng trực tiếp tới mẫu số của mọi tỷ lệ | §4.2 mục 5 |
| **Reliability $w_i=\sigma(\alpha\,\text{conf}+\beta\,\text{agree}+\gamma\,\text{tax})$ trộn lẫn thứ nguyên** | `tax` là ràng buộc **cứng** (hợp lệ/không) — nhét vào tổng tuyến tính rồi qua sigmoid là sai kiểu. Ngoài ra `conf` của model sinh chuỗi và `agree` tương quan mạnh (cùng nguồn lỗi) | Tách filter cứng khỏi score mềm; hiệu chỉnh trên audit người (§10) |
| **Forecast head** | Với chuỗi gần phẳng, naive persistence rất mạnh; forecast có nguy cơ là contribution rỗng | **Hạ xuống phụ lục.** Không làm claim chính |

---

## 6. Recommended problem formulation

Ba bài toán cần được phân biệt rạch ròi (đây là phần reviewer sẽ soi kỹ nhất):

- **Không phải ABSA tĩnh:** ABSA cho $x \to Q$ tại một thời điểm; ở đây $t$ là biến bậc nhất.
- **Không phải sentiment trend:** trend cấp document mất cấu trúc $(a,c,o,s)$; không trả lời được
  *"vì sao"* thay đổi.
- **Không phải topic trend:** topic đo *nói về gì*; ở đây đo *nói về nó như thế nào*, và tách bạch
  hai chiều đó (prevalence drift vs sentiment drift).
- **Không phải concept drift trong classification:** concept drift quan tâm $P(y|x)$ đổi làm model
  hỏng — lấy model làm trung tâm. Ở đây drift là **đối tượng nghiên cứu**, không phải sự cố cần vá.
- **Không phải temporal sentiment analysis thông thường:** điểm khác biệt là ước lượng
  **đã hiệu chỉnh thành phần** và có **kiểm định thống kê**, trên pseudo-label được định lượng độ tin cậy.

### Formal statement

Cho corpus $\mathcal{D}=\{(x_i,h_i,t_i,m_i)\}$ với $m_i$ = metadata (country, traveller type,
room, has_text, length). Extractor $f_\theta$ sinh quad $\hat q=(a,c,o,s)$ kèm reliability $w\in[0,1]$.

Với mỗi $(c,t)$ định nghĩa **hai đại lượng tách biệt** — điểm mấu chốt của formulation:

$$
\underbrace{\pi_{c,t}=\Pr(c \in Q \mid \text{review at } t)}_{\textbf{prevalence}},
\qquad
\underbrace{\rho_{c,t}=\Pr(s=\text{neg} \mid c \in Q,\ t)}_{\textbf{valence}}
$$

**Estimand mục tiêu — composition-adjusted, không phải trung bình thô:**

$$
\tilde\rho_{c,t}=\sum_{g\in\mathcal{G}} \pi^{\text{ref}}_{g}\cdot \rho_{c,t,g}
$$

trong đó $\mathcal{G}$ = các ô phân tầng (country-bloc × traveller type × length-bin) và
$\pi^{\text{ref}}_g$ là **thành phần tham chiếu cố định** (chọn = thành phần trung bình toàn kỳ).
Đây là **direct standardization** — trả lời được câu hỏi *"nếu thành phần khách không đổi thì
sentiment về aspect $c$ có đổi không?"*, tức tách $\Delta(\text{ai nói})$ khỏi $\Delta(\text{nói gì})$.

**Drift** = thay đổi có ý nghĩa thống kê của $\tilde\rho_{c,t}$ (hoặc $\pi_{c,t}$, hoặc phân phối
opinion), **sau khi kiểm soát seasonality và đã hiệu chỉnh đa so sánh** — không phải mọi dao động.

---

## 7. Proposed architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 0 · DATA FOUNDATION                                            │
│  1.95M pool ─┬─ parse date VI-regex ─ cắt 2025-03 & 2022-02          │
│              ├─ dedup (15,137 dòng)                                  │
│              ├─ GOLD BLOCKLIST  ◄── chặn leakage B2  [BLOCKING]      │
│              ├─ langid thật (fastText)                               │
│              └─ lọc text-bearing → 1.15M review                      │
│  Gold 23,995 quad ── giữ nguyên split gốc (train/dev/test)           │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1 · SEED EXTRACTOR                                             │
│  Fine-tune ASQP generative trên 8,816 segment train                  │
│  Báo cáo F1 exact-match trên test (1,890 seg) — TÁCH THEO NGÔN NGỮ   │
│  ⚠ đây là con số tham chiếu; mọi bước sau không được làm giảm        │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2 · SEMI-SUPERVISED EXTRACTION (1.15M review)                  │
│  ┌── field-provenance prior: pos-field⇒s≈pos, neg-field⇒s≈neg  (B3)  │
│  ├── teacher ensemble (≥2 seed khác nhau)                            │
│  ├── taxonomy hard filter (31 code hợp lệ)                           │
│  └── self-training có kiểm soát, dừng theo dev macro-F1 + ECE        │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3 · RELIABILITY  w_i ∈ [0,1]   (calibrated, KHÔNG dùng rating) │
│  conf (temp-scaled) · agreement · field-consistency · span validity  │
│  ── hiệu chỉnh trên 300-quad human audit ──                          │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4 · COMPOSITION-ADJUSTED AGGREGATION       ◄── contribution    │
│  post-stratify country-bloc × traveller × length-bin                 │
│  → π̃(c,t) prevalence   ρ̃(c,t) valence   E(c,t) opinion embedding    │
│  weighted by w_i; STL deseasonalize                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 5 · DRIFT DETECTION + INFERENCE                                │
│  best-split Welch-t trên residual · permutation null theo aspect     │
│  · BH-FDR · bootstrap CI (resample review, giữ nguyên strata)        │
│  ├ sentiment drift  ├ prevalence drift  ├ opinion drift  ├ emerging  │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 6 · LABEL-FREE VALIDATION (bằng chứng ĐỘC LẬP, dùng 1 lần)     │
│  held-out rating channel · synthetic drift injection · event probe   │
│  · negative control (shuffle) · human temporal micro-benchmark       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Mathematical formulation

**Ký hiệu.** Review $i$ tại kỳ $t$, hotel $h$, strata $g(i)\in\mathcal G$. Pseudo-quad
$\hat q$ với reliability $w$. $\mathbb 1[\cdot]$ = chỉ báo.

**(1) Prevalence có trọng số tin cậy, hiệu chỉnh thành phần**

$$
\hat\pi_{c,t}=\sum_{g}\pi^{\text{ref}}_{g}\cdot
\frac{\sum_{i\in t,g} \mathbb 1[\exists \hat q_i : c(\hat q_i)=c]\cdot \bar w_i}
     {\sum_{i\in t,g} \bar w_i}
$$

**(2) Valence (net sentiment) hiệu chỉnh thành phần**

$$
\hat\rho_{c,t}=\sum_{g}\pi^{\text{ref}}_{g}\cdot
\frac{\sum_{\hat q\in \mathcal Q_{c,t,g}} w_{\hat q}\,\phi(s_{\hat q})}
     {\sum_{\hat q\in \mathcal Q_{c,t,g}} w_{\hat q}},
\qquad \phi(\text{pos})=+1,\ \phi(\text{neu})=0,\ \phi(\text{neg})=-1
$$

**(3) Opinion semantic centroid** (encoder đóng băng $\psi$, ghi version)

$$
E_{c,t}=\frac{\sum_{\hat q\in\mathcal Q_{c,t}} w_{\hat q}\,\psi(o_{\hat q})}
              {\sum_{\hat q\in\mathcal Q_{c,t}} w_{\hat q}}
$$

**(4) Khử mùa vụ.** Với mỗi chuỗi $y_{c,t}\in\{\hat\pi,\hat\rho\}$, tách STL:
$y_{c,t}=T_{c,t}+S_{c,t}+R_{c,t}$; mọi kiểm định drift chạy trên $T+R$ (đã bỏ $S$).
*Bắt buộc*: `state` (loại khách) có chu kỳ mùa rõ (tỷ lệ Cặp đôi tụt vào tháng 6–7, mùa gia đình).

**(5) Bốn loại drift — định nghĩa tách bạch, không gộp thành một điểm số**

$$
\begin{aligned}
D^{\text{val}}(c,t) &= \big|\hat\rho_{c,t}-\hat\rho_{c,t-1}\big| &&\text{(sentiment drift)}\\
D^{\text{prev}}(c,t) &= \big|\log \hat\pi_{c,t}-\log\hat\pi_{c,t-1}\big| &&\text{(prevalence drift)}\\
D^{\text{sem}}(c,t) &= 1-\cos\!\big(E_{c,t},E_{c,t-1}\big) &&\text{(opinion semantic drift)}\\
D^{\text{dist}}(c,t) &= \mathrm{JS}\big(P_{c,t}\,\|\,P_{c,t-1}\big) &&\text{(phân phối sentiment)}
\end{aligned}
$$

> **Phản biện formulation tổng hợp $D=\alpha D_{sent}+\beta D_{sem}+\gamma D_{vol}$ (đề bài §7):**
> **Không nên dùng.** Ba lý do: (i) **thứ nguyên không tương thích** — JS ∈[0,1], $|\Delta\rho|$∈[0,2],
> cosine-distance ∈[0,2], log-ratio không chặn; cộng tuyến tính là so sánh táo với cam;
> (ii) **$\alpha,\beta,\gamma$ không thể hiệu chỉnh** vì không có gold drift → thành hyperparameter tuỳ ý,
> reviewer sẽ hỏi ngay "vì sao 0,3 mà không phải 0,5"; (iii) **mất tính diễn giải** — đúng điểm bán hàng
> của bài này là *phân biệt được* sentiment drift với opinion drift, gộp lại là vứt bỏ nó.
>
> **Thay bằng:** giữ 4 chiều tách rời, mỗi chiều có **p-value riêng** từ permutation null riêng,
> rồi phân loại drift theo **profile chữ ký** (signature):
>
> | Loại drift | $D^{\text{val}}$ | $D^{\text{prev}}$ | $D^{\text{sem}}$ |
> |---|---|---|---|
> | Sentiment drift | ✔ sig | – | – |
> | Opinion drift (đổi lý do, giữ cực) | ✗ ns | – | ✔ sig |
> | Prevalence drift | – | ✔ sig | – |
> | Emerging issue | ✔ sig | ✔ sig | ✔ sig |
>
> Đây vừa là **định nghĩa hình thức**, vừa là **kết quả phân loại có thể kiểm chứng** — mạnh hơn nhiều
> so với một con số vô hướng.

**(6) Kiểm định changepoint (giữ từ bản 1.0, áp trên chuỗi đã hiệu chỉnh)**

$$
c^\star=\arg\max_c \ \mathrm{Welch}\text{-}t\big(y_{1:c},\,y_{c+1:T}\big),\qquad
p=\Pr\nolimits_{\text{perm}}\big(t^{\text{null}}\ge t^{\text{obs}}\big)
$$

Permutation null **riêng theo từng aspect** (ngưỡng đo được khác nhau rõ rệt), sau đó **BH-FDR**
trên toàn lưới (aspect × cohort × loại drift).

---

## 9. Semi-supervised learning strategy

**Tỷ lệ thực tế: 1.150.415 review chưa nhãn / 8.796 review gold ≈ 131:1** (bản 1.0 ghi 52:1 — lỗi thời).

### 9.1 Điều chỉnh cốt lõi: khai thác field provenance (B3)

Đây là điểm khác biệt lớn nhất so với self-training ASQP tiêu chuẩn, và nó **đến từ dữ liệu chứ không
từ literature**. Vì `review_positive` / `review_negative` mang polarity gần như xác định
(9,52 vs 5,33; 1,2% vs 59,0% có score ≤6), ta có **prior mạnh và rẻ cho thành phần $s$**:

$$
P(s\mid \text{POS})=(0{,}931,\ 0{,}036,\ 0{,}033),\qquad
P(s\mid \text{NEG})=(0{,}217,\ 0{,}142,\ 0{,}641)
$$

trên $(\text{pos},\text{neu},\text{neg})$ — **đã đo trực tiếp trên nhãn gold vòng 2** (n=11.733 quad
từ trường POS, n=2.396 từ trường NEG), không còn là ước lượng qua `review_score`.

⚠️ **Sửa quan trọng so với vòng 1:** trường negative **yếu hơn nhiều** so với ước lượng bằng score
(64,1% chứ không phải ~93%). Nguyên nhân đã kiểm chứng: khách thường viết nội dung tích cực trong ô
"điều chưa hài lòng" (*"The owner was lovely although…"*). Vì vậy **không dùng bộ lọc cứng** — dùng
posterior $P(s\mid x,\phi)\propto P_\theta(s\mid x)^{\lambda}P(s\mid\phi)^{1-\lambda}$, nếu không sẽ
vứt đúng 21,7% ca khó và giàu thông tin nhất.

**Hệ quả cho pipeline:**
- Field provenance là tín hiệu **độc lập với confidence của model** (khác với `agree` vốn tương quan
  cao với `conf`) — nhưng phải dùng dạng **posterior mềm**, không phải bộ lọc cứng (xem cảnh báo trên).
- Ngân sách khó khăn dồn hết cho **(a, c, o)** — đúng chỗ bài toán thật sự khó.
- Quad có $s$ **mâu thuẫn** field → **không vứt**, mà đưa vào **hàng đợi human audit**: đây chính là
  nơi tập trung các trường hợp mỉa mai, phủ định, hoặc lỗi extractor — mẫu vàng cho error analysis.

### 9.2 Chiến lược khuyến nghị (đủ mới, vẫn khả thi)

| Thành phần | Quyết định | Lý do |
|---|---|---|
| Self-training | ✅ **Có**, 2–3 vòng | Chuẩn, an toàn, đã có tiền lệ (Zhang et al. ACL 2024) |
| Teacher–student | ✅ **Có** — 2 teacher khác seed + khác backbone | Cần cho `agree`; chi phí chấp nhận được |
| Cross-model agreement | ✅ **Có** | Nhưng phải nhớ: 2 model cùng train trên 1 gold → **lỗi hệ thống chung**, agreement cao ≠ đúng. Phải hiệu chỉnh bằng human audit |
| **Field-provenance weak supervision** | ✅ **Có — thành phần đặc thù của bài này** | Rẻ, mạnh, **độc lập với model** |
| Consistency regularization | ⚠️ **Bỏ** | Augmentation cho ASQP (span-level) dễ phá span; ROI thấp |
| LLM-assisted pseudo-labeling | ⚠️ **Hạn chế** | Không dùng để **sinh** pseudo-label đại trà (chi phí 1,15M review + không tái lập). Dùng làm **teacher thứ 3 trên mẫu** và làm judge (§10) |
| Ensemble | ✅ Nhẹ (2–3 model) | |
| Weak supervision (Snorkel-style) | ✅ **Có**, dạng nhẹ | Labeling function = field provenance + taxonomy keyword + span validity |
| Curriculum theo ngôn ngữ | ✅ **Nên** | Gold 85% en, pool ~30% vi → oversample gold tiếng Việt, đo F1 tách theo ngôn ngữ |

### 9.3 Error propagation — đánh giá trung thực

**Rủi ro thật và không nhỏ.** Cơ chế nguy hiểm nhất **không** phải nhiễu ngẫu nhiên (nhiễu ngẫu nhiên
trung bình hoá tốt qua ~60k quad/tháng), mà là **bias có tương quan với thời gian**:

> Reviews dài hơn theo thời gian (27,5→40,8 từ) và đổi ngôn ngữ (VN 89,7%→16,1%). Nếu extractor có
> recall cao hơn trên câu dài / trên tiếng Anh, thì **prevalence sẽ tăng giả tạo theo thời gian**
> thuần tuý do đặc tính model — và trông y hệt một aspect prevalence drift thật.

**Đây là mối đe doạ số một đối với validity của toàn bài.** Bắt buộc:
- **Length-stratified & language-stratified recall probe** trên gold test: đo F1 theo bin độ dài và
  theo ngôn ngữ. Nếu F1 phụ thuộc độ dài/ngôn ngữ → **đưa length-bin và language vào strata $\mathcal G$**
  (đã có trong §6) và báo cáo tường minh.
- **Negative control bắt buộc:** chạy toàn bộ pipeline trên timestamp **đã xáo trộn**. Mọi drift phát
  hiện được ở đó là false positive rate thực nghiệm.

### 9.4 Confidence nên tính ở mức nào

**Component-level, rồi tổng hợp — không phải quad-level thuần.** Lý do: quad-level log-prob của model
sinh chuỗi bị chi phối bởi độ dài chuỗi và tần suất token, và **trộn lẫn 4 quyết định có độ khó rất
khác nhau** ($s$ gần như free, $c$ trung bình, $a/o$ khó nhất). Đề xuất:

$$
w_{\hat q}=\underbrace{\mathbb 1[\text{tax hợp lệ}]\cdot\mathbb 1[\text{span hợp lệ}]}_{\text{filter cứng}}
\cdot \sigma\Big(\beta_0+\beta_1 z^{\text{conf}}_{a,o}+\beta_2 z^{\text{agree}}+\beta_3 z^{\text{field}}+\beta_4 z^{\text{len}}\Big)
$$

với $\beta$ **fit bằng logistic regression trên 300-quad human audit** (nhãn = quad đúng/sai), *không*
đặt tay. Đây là điểm sửa quan trọng so với bản 1.0: filter cứng tách khỏi score mềm, và trọng số
được **học từ ground truth thật** thay vì gán tuỳ ý.

---

## 10. Label-free validation framework

### 10.1 Nguyên tắc chống circular validation

> **Quy tắc một-lần-một-phía:** mỗi tín hiệu chỉ được dùng **hoặc** để tạo/lọc pseudo-label,
> **hoặc** để validate — **không bao giờ cả hai.**

Bản 1.0 vi phạm quy tắc này. Bảng phân vai bắt buộc:

| Tín hiệu | Vai trò | Ghi chú |
|---|---|---|
| Model confidence | **Tạo** | Không bao giờ là bằng chứng đúng |
| Cross-model agreement | **Tạo** | Lỗi hệ thống chung — không phải bằng chứng độc lập |
| Field provenance (pos/neg field) | **Tạo** | §9.1 |
| `review_score` | **Validate** — và **chỉ** validate | **Không** được dùng làm feature lọc pseudo-label. Vì B3 buộc nó tương quan mạnh với field provenance, phải dùng **held-out theo hotel**: hiệu chỉnh trên cohort A, validate trên cohort B |
| Keyword-proxy | **Validate** (yếu) | Chỉ dùng như sanity check, không làm bằng chứng chính |
| Human audit (300 quad) | **Tạo** (fit $\beta$) | Đã tiêu vào việc fit → **không** dùng lại để validate |
| Human temporal benchmark (riêng, ~400 review) | **Validate** | **Mẫu tách biệt, phải thu riêng** |
| LLM-as-a-judge | **Validate** (phụ) | §10.3 |
| Synthetic drift injection | **Validate** | Bằng chứng mạnh nhất — có ground truth thật |
| Shuffled-time negative control | **Validate** | Đo FPR thực nghiệm |

### 10.2 Về temporal consistency

**Không dùng temporal consistency làm bộ lọc pseudo-label** — hai lý do, một cũ một mới:
1. *(bản 1.0, đã đo)* Vô hiệu: disagreement 11,97% vs 12,11% khi shuffle → chênh lệch bằng nhiễu.
2. *(quan trọng hơn)* **Nó phá huỷ chính đối tượng nghiên cứu.** Lọc quad "bất thường so với $t\pm1$"
   chính là lọc bỏ tín hiệu drift thật. Đây là dạng circularity tinh vi nhất: làm mượt dữ liệu rồi
   kết luận dữ liệu mượt. **Phải nêu tường minh trong paper** — reviewer tinh ý sẽ tìm đúng lỗi này,
   và việc mình chủ động loại trừ nó là một điểm cộng về rigor.

### 10.3 LLM-as-a-Judge — dùng ở đâu

- **Có dùng**, nhưng **chỉ ở Phase 6**, trên **mẫu phân tầng ~1.000 quad**, không phải toàn corpus.
- **Không** dùng làm gold. Dùng làm **tín hiệu đồng thuận thứ ba** bên cạnh human audit.
- Bắt buộc khai báo: model ID + version, prompt nguyên văn, temperature = 0, seed, ngày chạy.
- Bắt buộc đo **agreement giữa LLM judge và human** trên phần chồng lấn (~200 quad) và báo cáo Cohen's κ.
  **Nếu κ < 0,6 → bỏ LLM judge khỏi paper**, không cứu vãn.
- Bias phải nêu: LLM thiên vị văn bản tiếng Anh (gold 85% en, pool ~70% non-vi-diacritic) → **báo cáo κ
  tách theo ngôn ngữ**.

### 10.4 Đánh giá công thức tổng hợp $C(q)=\sum w_i C_i$ (đề bài §5)

**Không hợp lý ở dạng đã nêu**, ba lỗi:
1. **$C_{\text{temporal}}$ phải bị loại hoàn toàn** — xem §10.2, nó khử mất tín hiệu cần đo.
2. **$C_{\text{rating}}$ không được nằm trong công thức** — nó là bằng chứng validate duy nhất
   tương đối độc lập; tiêu nó vào việc tạo label là mất trắng khả năng validate (§10.1).
3. **Trọng số $w_i$ không thể xác định** nếu không có nhãn — mà một khi đã có human audit để fit
   thì nên fit bằng logistic regression (§9.4) chứ không đặt tay.

**Phiên bản tốt hơn:** filter cứng × sigmoid(logistic đã fit trên human audit), chỉ gồm các tín hiệu
thuộc nhóm "Tạo" ở bảng §10.1. Đơn giản hơn, có cơ sở thống kê, và **giữ được** `review_score` cùng
temporal signal làm bằng chứng validate độc lập.

---

## 11. Aspect sentiment drift methodology

### 11.1 Temporal aspect representation

$$
z_{c,t}=\big[\ \hat\pi_{c,t},\ \ \hat\rho_{c,t},\ \ P^{+}_{c,t},P^{0}_{c,t},P^{-}_{c,t},\ \
E_{c,t},\ \ V_{c,t},\ \ \bar w_{c,t},\ \ U_{c,t},\ \ \mathbf{g}_t\ \big]
$$

| Thành phần | Nguồn dữ liệu | Bắt buộc? |
|---|---|---|
| $\hat\pi$ prevalence (đã hiệu chỉnh) | quad + `country`,`state` | ✅ |
| $\hat\rho$ valence (đã hiệu chỉnh) | quad `sentiment` | ✅ |
| $P^{+/0/-}$ phân phối 3 lớp | quad `sentiment` | ✅ (JS distance cần phân phối, không cần scalar) |
| $E$ centroid embedding opinion | `opinion_term` + encoder đóng băng | ✅ — **đây là chiều tạo nên opinion drift** |
| $V$ khối lượng review | đếm | ✅ (mẫu số + biến kiểm soát) |
| $\bar w$ reliability trung bình | Phase 3 | ✅ — **nếu $\bar w$ tự nó drift theo thời gian, mọi kết luận bị nghi ngờ; phải plot** |
| $U$ uncertainty (bootstrap SE) | resample | ✅ |
| $\mathbf g_t$ vector thành phần (tỷ trọng strata) | `country`,`state`,`room`,length | ✅ — **phải report như một chuỗi riêng để chứng minh đã kiểm soát** |
| $R_t$ rating trung bình | `review_score` | ❌ **Cố ý loại** khỏi representation — giữ làm kênh validate độc lập (§10.1) |

**Khuyến nghị cho hotel review cụ thể:** cặp $(\hat\pi,\hat\rho)$ là lõi — vì nó tách bạch
*"nhắc tới nhiều hơn"* khỏi *"nói xấu hơn"*, hai thứ mà rating tổng gộp làm một và không phân biệt được.
$E$ là chiều tạo ra tính mới. $\mathbf g_t$ là chiều tạo ra tính đáng tin.

### 11.2 Quy trình hiệu chỉnh thành phần (chi tiết thực thi)

1. Định nghĩa strata $\mathcal G$ = `country_bloc`(6: VN / Tây Âu / Bắc Mỹ / Úc-NZ / Đông Á / Khác)
   × `state`(4) × `length_bin`(3) = **72 ô**. Với ~30k text-review/tháng → trung bình ~420/ô,
   đủ để ước lượng ổn định ở cấp corpus.
2. $\pi^{\text{ref}}_g$ = tỷ trọng ô $g$ tính trên **toàn bộ 36 tháng gộp** (cố định, khai báo rõ).
3. Ô có $n<30$ trong một tháng → **gộp lên mức thô hơn** (bỏ chiều `length_bin` trước, rồi `state`).
4. Báo cáo **song song** cả chuỗi thô và chuỗi đã hiệu chỉnh. **Độ chênh giữa hai chuỗi chính là
   thước đo mức độ nghiêm trọng của confounder** — và đây là một figure có sức thuyết phục rất cao
   cho reviewer.
5. Robustness: lặp lại bằng **IPW** (logistic propensity trên metadata) thay cho post-stratification;
   kết luận phải bền qua cả hai.

### 11.3 Mô hình hoá cấp hotel

Giữ **partial pooling** từ bản 1.0 (đúng hướng), điều chỉnh cohort:
- Hotel cohort A (**276** hotel, ≥1000 review): ước lượng riêng, granularity **tháng**.
- Hotel cohort B (**1.221** hotel, ≥300 review): granularity **quý**, ước lượng có shrinkage.
- Còn lại: **không ước lượng riêng**, chỉ đóng góp vào chuỗi corpus/cohort.
- Empirical-Bayes shrinkage về trung bình aspect toàn corpus (phương án tối thiểu). BSTS phân cấp
  để ở bản mở rộng — **không đưa vào critical path**.

---

## 12. Evaluation protocol

**Sự thật phải nói thẳng: không có và không thể có drift ground truth.** Vì vậy evaluation phải là
**hội tụ nhiều bằng chứng độc lập yếu**, không phải một con số.

### E1 · Extraction quality (có gold — đây là chỗ duy nhất có số thật)
- Exact-match quad P/R/F1 trên gold test (1.890 segment), **và** F1 từng thành phần $(a,c,o,s)$.
- **Tách theo ngôn ngữ** (en / vi) và **theo length-bin** — bắt buộc, vì §9.3.
- **Chronological split bổ sung:** ngoài split gốc, chạy thêm split theo thời gian
  (train ≤2024-06, test ≥2024-07). **Nếu F1 tụt mạnh → temporal generalization kém → mọi kết luận
  drift phải hạ mức tự tin.** Đây là kiểm tra không thể thiếu cho một bài về temporal.

### E2 · Synthetic drift injection (bằng chứng mạnh nhất — có ground truth thật)
Trên corpus thật, chèn drift có kiểm soát và đo detection:
- **Valence injection:** trong cửa sổ $[t_0,T]$, lật $\delta\%$ quad của aspect $c$ từ pos→neg,
  $\delta\in\{2,5,10,20\}$ → đo detection rate & độ lệch ước lượng magnitude.
- **Prevalence injection:** oversample/undersample review chứa $c$.
- **Opinion injection:** thay opinion term bằng cụm đồng cực nhưng khác nghĩa (slow → crowded)
  → **kiểm tra $D^{\text{sem}}$ bắt được trong khi $D^{\text{val}}$ thì không** — đây chính là
  bằng chứng cho claim "phân biệt được 4 loại drift", và là experiment quan trọng nhất của bài.
- **Composition injection (đặc thù bài này):** giữ nguyên sentiment, chỉ đổi tỷ trọng strata theo thời gian
  → **phương pháp đã hiệu chỉnh phải KHÔNG báo drift, baseline thô PHẢI báo drift.** Đây là
  experiment chứng minh contribution chính.
- Báo cáo **ROC/AUC theo $\delta$** và **minimum detectable effect**.

### E3 · Negative control
Xáo trộn timestamp (giữ nguyên mọi thứ khác), chạy full pipeline → **FPR thực nghiệm**.
Phải ≈ mức $\alpha$ sau FDR. Nếu cao hơn → pipeline có lỗi.

### E4 · Bootstrap stability
Resample review **trong từng strata** (giữ cấu trúc), $B=1000$ → CI cho $D$, và
$\text{Stability}=\Pr(\text{cùng dấu \& cùng changepoint} \pm 1 \text{ kỳ})$.

### E5 · Cross-model consistency
Đổi backbone + seed extractor → drift có tái lập không. Báo cáo Spearman giữa hai bảng xếp hạng drift.

### E6 · External validation bằng rating (held-out, dùng một lần)
Trên cohort **chưa dùng để hiệu chỉnh gì cả**: chuỗi $\hat\rho_{c,t}$ có tương quan với chuỗi
`review_score` không? **Lưu ý diễn giải đúng:** tương quan **vừa phải** mới là kết quả tốt —
tương quan quá cao nghĩa là quad không thêm thông tin gì so với rating (bài mất lý do tồn tại);
tương quan bằng 0 nghĩa là quad có thể là nhiễu. **Phải nêu trước ngưỡng kỳ vọng** (đề xuất: Spearman
0,3–0,7) **trước khi chạy**, để tránh HARKing.

### E7 · Event-based probe
**VN mở cửa du lịch quốc tế 15/03/2022** là sự kiện ngoại sinh duy nhất dùng được trong span.
Giả thuyết kiểm chứng được: aspect liên quan khách quốc tế (`SER_COMM` giao tiếp, `AM_FOOD`,
`FAC_VIEW_LOCATION`) phải có prevalence drift quanh 2022 Q2–Q3. **Cảnh báo:** đây cũng chính là B1,
nên nó **vừa là validation vừa là confounder** — phải trình bày trung thực, dùng làm bằng chứng
*bổ trợ* chứ không phải bằng chứng chính.

### E8 · Human temporal micro-benchmark
~400 review phân tầng (2 kỳ × 5 aspect × 40 review), 2 annotator, đo κ. Dùng để kiểm tra chiều và
độ lớn của drift ở vài cặp $(c,t)$ nổi bật. **Mẫu này phải tách hoàn toàn khỏi 300-quad audit ở §9.4.**

---

## 13. Baselines

### 13.1 Quad extraction

| Nhóm | Baseline | Bắt buộc? |
|---|---|---|
| Supervised | Fine-tune generative ASQP (T5/mT5) trên gold, **không** self-training | ✅ Đây là số tham chiếu |
| Pipeline | Extract-classify 2 giai đoạn | ✅ Rẻ, chứng minh giá trị của end-to-end |
| Semi-supervised | Self-training vanilla (chỉ lọc theo confidence) | ✅ **Baseline chính để đánh bại** |
| Semi-supervised | Self-training + field-provenance filter (**đề xuất**) | — |
| LLM | Zero-shot & few-shot (GPT-class / Claude-class) trên gold test | ✅ Reviewer 2026 sẽ hỏi |
| LLM | LLM làm teacher trên mẫu 10k → so chi phí/chất lượng | ⚠️ Tuỳ ngân sách |
| Weak | Keyword/lexicon matching theo taxonomy | ✅ Sàn dưới, rất rẻ |

### 13.2 Drift detection

| Nhóm | Baseline | Ghi chú |
|---|---|---|
| Trivial | Raw difference giữa kỳ liên tiếp | Sàn dưới |
| Trivial | **Document-level rating trend** (không dùng ABSA) | ✅ **Baseline quan trọng nhất** — nếu không đánh bại được nó thì cả bài không có lý do tồn tại |
| Smoothing | Moving average, EWMA | ✅ |
| Streaming | CUSUM, Page-Hinkley | ✅ Kinh điển, rẻ, reviewer quen thuộc |
| Streaming | ADWIN | ⚠️ Thiết kế cho stream cấp instance, hơi lệch setting; đưa vào nếu còn thời gian |
| Change-point | PELT / Binary segmentation (`ruptures`) | ✅ So với best-split Welch-t đề xuất |
| Distribution | JS / Wasserstein giữa các kỳ | ✅ |
| Embedding | Semantic shift kiểu diachronic word embedding | ✅ Baseline cho $D^{\text{sem}}$ |
| **Ablation-baseline** | **Chính phương pháp nhưng KHÔNG hiệu chỉnh thành phần** | ✅ **Bắt buộc** — đây là bằng chứng trực tiếp cho contribution chính |

---

## 14. Ablation studies

| Ablation | Chứng minh điều gì | Ưu tiên |
|---|---|---|
| **− composition adjustment** | Contribution chính. Kỳ vọng: số drift "phát hiện được" giảm mạnh, và trên composition-injection test baseline thô cho FPR cao | 🔴 **Cao nhất** |
| **− field-provenance filter** | Giá trị của weak supervision đặc thù dữ liệu này | 🔴 Cao |
| **− reliability weighting** ($w_i\equiv1$) | Uncertainty-aware aggregation có đáng công không | 🔴 Cao |
| **− cross-model agreement** | Đóng góp riêng của ensemble (tách khỏi conf) | 🟡 TB |
| **− deseasonalization** | Bao nhiêu "drift" thực chất là mùa vụ | 🔴 Cao |
| **− FDR correction** | Mức thổi phồng nếu bỏ hiệu chỉnh đa so sánh | 🟡 TB (nhưng phải có bảng trước/sau) |
| **− semantic opinion channel** | Có bao nhiêu opinion drift bị bỏ sót nếu chỉ nhìn sentiment | 🔴 Cao — chứng minh 4-way taxonomy có ý nghĩa |
| **− self-training** (chỉ gold) | Giá trị của scaling 131:1 | 🔴 Cao |
| **− temporal consistency** *(đã loại)* | Báo cáo negative result: filter này vô hiệu (11,97% vs 12,11%) **và** nguy hiểm về mặt khái niệm | 🟢 Thấp nhưng nên viết |
| Granularity tháng vs quý | Độ nhạy của kết luận với lựa chọn thiết kế | 🟡 TB |
| Strata definition (72 ô vs thô hơn) | Robustness của hiệu chỉnh | 🟡 TB |

---

## 15. Research Gap Matrix

| Hướng hiện có | Thường làm gì | Hạn chế | Đóng góp khả dĩ của ta | Bằng chứng từ dữ liệu |
|---|---|---|---|---|
| **ASQP** | Trích xuất tĩnh trên SemEval/ACOS (vài nghìn mẫu) | Không có chiều thời gian; corpus nhỏ | ASQP có timestamp ở quy mô 1,15M review | 38 tháng, 23.995 gold quad |
| **Semi-supervised ABSA** | Cải thiện label efficiency, báo cáo F1 trên test có nhãn | **Không ai đánh giá được độ tin cậy của pseudo-label ở quy mô không thể annotate** | Reliability calibrated + phân vai chống circularity | Tỷ lệ 131:1 |
| **Sentiment trend / hospitality analytics** | Trung bình rating theo thời gian, hoặc topic model + polarity | Mất cấu trúc $(a,c,o,s)$; **không kiểm soát thành phần reviewer** | Tách $\hat\pi$ / $\hat\rho$ / $E$; hiệu chỉnh thành phần | VN 89,7%→16,1% chứng minh đây là lỗ hổng thật, không phải giả định |
| **Concept drift NLP** | Phát hiện $P(y|x)$ đổi để **bảo vệ model** | Lấy model làm trung tâm; drift là sự cố cần vá | Drift là **đối tượng nghiên cứu**, có kiểm định thống kê chính thức | Permutation null theo aspect + BH-FDR |
| **Temporal ABSA** | Phân tích sentiment theo thời gian trên corpus nhỏ | Trích xuất không scale; hiếm khi có kiểm định | Semi-supervised temporal ASQP có kiểm định | 1,95M review |
| **Opinion mining** | Trích xuất opinion term | Ngữ nghĩa thời gian yếu | **Opinion semantic drift**: cực tính giữ nguyên nhưng lý do đổi | Injection test E2 chứng minh phân biệt được |
| **Diachronic semantics** | Word meaning change qua thế kỷ | Cấp từ vựng, không gắn aspect/sentiment | Semantic drift **có điều kiện theo aspect**, thang tháng | $E_{c,t}$ |
| **Survey/causal inference** | Direct standardization, IPW | Chưa được đưa vào NLP temporal analysis | **Nhập kỹ thuật standardization vào temporal ABSA** | 72 strata từ `country`×`state`×length |

**Gap sắc nét nhất (một câu):** *chưa ai đo drift sentiment cấp aspect trên pseudo-label quy mô lớn
mà đồng thời (i) định lượng được độ tin cậy của chính pseudo-label đó không cần annotate toàn bộ, và
(ii) tách được thay đổi chất lượng thật khỏi thay đổi thành phần người viết review.*

---

## 16. Risks and mitigation

| # | Rủi ro | Mức | Giảm thiểu |
|---|---|---|---|
| R1 | Composition drift bị nhầm thành quality drift | 🔴 **Rất cao — đã xác nhận tồn tại** | §11.2 + ablation + composition-injection test (E2) |
| R2 | Leakage gold⊂pool làm hỏng mọi F1 | 🔴 Cao (đã xác nhận) | Blocklist bắt buộc ở Phase 0; báo cáo số dòng loại |
| R3 | Length/language-dependent recall tạo prevalence drift giả | 🔴 Cao | Length-stratified probe; đưa length vào strata; negative control |
| R4 | Circular validation | 🔴 Cao (bản 1.0 mắc) | Bảng phân vai §10.1, quy tắc một-lần-một-phía |
| R5 | Truncation/survivorship bias (549→8.535 hotel/tháng) | 🟠 TB-Cao | Lọc hotel có mật độ ổn định; báo cáo độ nhạy |
| R6 | Subcategory hiếm không đủ gold để học (9/31 code <100 quad, 17/31 <300) | 🟠 TB | Giới hạn scope xuống 14 code + 6 category; khai báo rõ |
| R7 | Extractor yếu trên tiếng Việt (gold 85% en) | 🟠 TB | Oversample; báo cáo F1 tách ngôn ngữ; cân nhắc bổ sung 500 gold tiếng Việt |
| R8 | Reviewer đòi drift ground truth | 🟠 TB | E2 synthetic injection + E8 micro-benchmark trả lời trực diện |
| R9 | Neutral class quá yếu (5,2%) | 🟡 Thấp-TB | Báo cáo F1 riêng; cân nhắc gộp về nhị phân cho drift, giữ 3 lớp cho extraction |
| R10 | Rating tương quan quá cao → bài mất lý do tồn tại | 🟡 Thấp-TB | Nêu ngưỡng kỳ vọng **trước** khi chạy (E6); chuẩn bị case study aspect ngược chiều rating |
| R11 | Chi phí tính toán 1,15M review qua ensemble | 🟡 Thấp | Chạy cohort A/B trước (276/1.221 hotel); mở rộng sau |
| R12 | 41% review không text làm lệch mẫu số | 🟡 Thấp | Universe = 1,15M; `has_text` là một chiều strata |

---

## 17. Recommended paper scope

### Option A — Safe

- **RQ:** Sentiment cấp aspect với hotel VN có drift theo thời gian không, và ASQP bán giám sát có
  phát hiện được tin cậy hơn rating tổng không?
- **Scope:** corpus + cohort A (276 hotel), 6 category, granularity tháng, chỉ sentiment + prevalence drift.
- **Novelty:** thấp-TB. **Khó:** thấp. **Rủi ro:** thấp.
- Phù hợp: workshop / applied venue / tạp chí hospitality-analytics.

### Option B — Balanced ⭐ **KHUYẾN NGHỊ**

- **RQ1:** Làm sao ước lượng độ tin cậy của hàng triệu pseudo-quad mà không annotate toàn bộ?
- **RQ2:** Làm sao tách **thay đổi chất lượng thật** khỏi **thay đổi thành phần người review**
  khi đo aspect sentiment drift?
- **RQ3:** Có phân biệt được sentiment drift với opinion drift (cực tính giữ, lý do đổi) không?
- **Scope:** 1,15M review text-bearing · 6 category + 14 subcategory · corpus + cohort A&B ·
  4 loại drift tách bạch · **composition-adjusted estimand** · permutation null + BH-FDR ·
  synthetic injection làm evaluation lõi.
- **Contributions:** (1) label-free reliability có phân vai chống circularity;
  (2) **composition-adjusted aspect drift** — nhập direct standardization vào temporal ABSA;
  (3) phân loại 4 chiều drift với chữ ký kiểm chứng được bằng injection;
  (4) HAMoS-Temporal: 1,15M pseudo-quad có timestamp + reliability (tài nguyên công bố được).
- **Novelty:** cao và **có thể chứng minh bằng experiment** (E2 composition-injection). **Khó:** TB.
- **Rủi ro:** TB — đã có bằng chứng đo được cho mọi giả định chính.
- Phù hợp: ACL/EMNLP/NAACL main hoặc findings; LREC cho phần dataset.

### Option C — Ambitious

- Thêm: mô hình BSTS phân cấp đầy đủ, drift forecasting, nhánh multimodal (121.584 review có ảnh),
  emerging-issue discovery không giám sát (aspect ngoài taxonomy), cross-hotel causal comparison.
- **Novelty:** rất cao. **Khó:** cao. **Rủi ro:** 🔴 cao.
- Vấn đề cụ thể: forecasting nhiều khả năng thua naive persistence (chuỗi gần phẳng 8,47–8,87);
  nhánh ảnh chỉ 6,24% độ phủ (giảm từ 11,5% ở pool cũ) và không có alignment → khó ra kết quả có ý nghĩa;
  emerging issue ngoài taxonomy không có cách nào validate.

### Khuyến nghị: **Option B**

Ba lý do:
1. **Contribution chính được dữ liệu ép ra, không phải nghĩ ra.** Composition drift (VN 89,7%→16,1%,
   text 42%→63%, độ dài 27,5→40,8) lớn tới mức **bất kỳ ai làm bài này mà bỏ qua nó đều sai** — nên
   giải quyết nó vừa là điều bắt buộc vừa là điểm mới. Đây là dạng novelty vững nhất.
2. **Mọi claim đều có experiment kiểm chứng được**, đặc biệt composition-injection test (E2) cho
   bằng chứng nhị phân sạch: phương pháp đề xuất không báo drift ↔ baseline báo drift.
3. **Option C có ít nhất hai nhánh nhiều khả năng ra kết quả rỗng** (forecasting, multimodal) và sẽ
   tiêu thời gian mà không đổi được acceptance. Giữ chúng làm future work.

---

## 18. Implementation roadmap

### Phase 0 — Data foundation `[BLOCKING · ~1 tuần]`
- [ ] Trỏ pipeline vào `hamos-mabsa/data/annotations/quads.jsonl` (**không** dùng
      `hotel_absa_labeled.jsonl` — file này **không có quad**)
- [ ] Parse `review_date` regex tiếng Việt trên 1.949.604 dòng; xác nhận lại 184 null
- [ ] **Cắt cửa sổ 2022-03 → 2025-02 (36 tháng)**; loại 2025-03 (11.211, crawl dở) và 2022-02 (1.391)
- [ ] Dedup 15.137 dòng trùng
- [ ] **Xây gold blocklist** (hash chuẩn hoá + `(hotel_id, date)`), loại khỏi pool, **báo cáo số dòng loại** 🔴
- [ ] Ép kiểu `hotel_id` str↔int; xác nhận lại overlap 3.399/3.399
- [ ] fastText langid trên 1.150.415 review có text
- [ ] Chốt & ghi version encoder (một checkpoint duy nhất cho toàn dự án)
- [ ] Chốt danh sách cohort A (276) / B (1.221) / C (2.313)
- [ ] **Audit 50 mẫu** kiểm tra giả thuyết `review_text` gold bị ghép lossy (§4.2 mục 7)

### Phase 1 — Seed extractor `[~1,5 tuần]`
- [ ] Fine-tune ASQP generative trên 8.816 segment train; đo trên test (1.890)
- [ ] Bảng F1: tổng thể · theo thành phần $(a,c,o,s)$ · **theo ngôn ngữ** · **theo length-bin** 🔴
- [ ] **Chronological split probe** (train ≤2024-06 / test ≥2024-07) 🔴
- [ ] Temperature scaling trên dev (1.895 segment); báo cáo ECE

### Phase 2 — Semi-supervised extraction `[~2 tuần]`
- [ ] Field-provenance prior; đo lại $P(s\mid\text{field})$ **trên human audit**, không trên score
- [ ] Teacher ensemble ≥2; taxonomy hard filter (31 code)
- [ ] Self-training 2–3 vòng; log phân phối lớp mỗi vòng; dừng theo dev macro-F1 / ECE
- [ ] Chạy inference trên cohort A trước (~50k review), rồi B, rồi mở rộng

### Phase 3 — Reliability `[~1 tuần]`
- [ ] Lấy **300 quad** phân tầng, annotate người → fit logistic $\beta$ (§9.4)
- [ ] Sinh $w_i$ cho toàn bộ pseudo-quad
- [ ] **Plot $\bar w_t$ theo thời gian** — nếu reliability tự nó drift, phải xử lý trước khi đi tiếp 🔴

### Phase 4 — Composition-adjusted aggregation `[~1 tuần]`
- [ ] Xây 72 strata; tính $\pi^{\text{ref}}$; quy tắc gộp ô $n<30$
- [ ] Tính $\hat\pi,\hat\rho,E,U$ cho từng (aspect × kỳ × cohort)
- [ ] STL deseasonalize
- [ ] **Figure chuỗi thô vs chuỗi hiệu chỉnh** — figure chủ lực của bài 🔴

### Phase 5 — Drift detection `[~1,5 tuần]`
- [ ] Best-split Welch-t trên residual; permutation null **riêng từng aspect**
- [ ] BH-FDR toàn lưới; bảng trước/sau hiệu chỉnh
- [ ] Bootstrap CI (resample **trong strata**)
- [ ] Phân loại 4 loại drift theo bảng chữ ký (§8)

### Phase 6 — Evaluation `[~2 tuần]`
- [ ] **E2 synthetic injection**: valence / prevalence / opinion / **composition** 🔴 *(experiment quan trọng nhất)*
- [ ] E3 shuffled-time negative control → FPR
- [ ] E5 cross-model consistency
- [ ] **E6 nêu ngưỡng kỳ vọng Spearman TRƯỚC khi chạy** (chống HARKing) 🔴
- [ ] E7 event probe quanh 2022-03-15
- [ ] E8 human temporal micro-benchmark (~400 review, mẫu tách biệt)

### Phase 7 — Ablation & viết `[~2 tuần]`
- [ ] Bảng ablation ưu tiên 🔴 ở §14
- [ ] Case study định tính: aspect drift ngược chiều rating tổng (bằng chứng trực quan cho luận điểm chính)
- [ ] Đóng gói **HAMoS-Temporal** để công bố

**Đường găng nếu thiếu thời gian:**
`Phase 0 (blocklist) → Phase 1 (F1 + chronological probe) → Phase 2 (cohort A) → Phase 4 (hiệu chỉnh) → E2 composition-injection → Phase 5`.
Sáu bước này đủ tạo thành một bài hoàn chỉnh. Multimodal, forecasting, BSTS phân cấp, cohort C — cắt trước tiên.

---

## Phụ lục A — Tra cứu nhanh số liệu đã đo (2026-08-21)

```
POOL   hotel_booking_unlabeled.jsonl
  1.949.604 review · 10.631 hotel · 2022-02→2025-03 (38 tháng) · 184 null date · 0 JSON lỗi
  duplicate 15.137 (0,78%)
  text: both 678.194 (34,8%) · pos-only 444.508 (22,8%) · neg-only 27.713 (1,4%) · none 799.189 (41,0%)
  → universe text-bearing = 1.150.415 (59,0%)
  photo 121.584 (6,24%) · stars_rating có ở 89,9%
  review/hotel: mean 183,4 · median 66 · p90 457 · max 8.599
  tháng phủ/hotel: mean 19,7 · median 19 · ≥24 tháng: 4.025 · ≥36 tháng: 962
  state: Cặp đôi 40,8% · Gia đình 24,0% · Khách lẻ 18,9% · Nhóm 16,0%
  country: VN 29,9% · Pháp 6,8% · Úc 6,6% · Đức 6,5% · Anh 6,4%
  score: điểm 10 = 44,9%; trung bình theo tháng dao động 8,47–8,87

GOLD   hamos-mabsa/data/annotations/quads.jsonl
  23.995 quad · 12.601 segment · 8.796 review · 3.399 hotel · 38 tháng
  category(6): FACILITY 55,1 · AMENITY 20,5 · SERVICE 10,7 · EXPERIENCE 8,3 · LOYALTY 3,2 · BRANDING 2,2
  taxonomy(31): FAC_ROOM 5.444 … BRA_REPUTE 22   (11 code <100 quad)
  sentiment: pos 79,4% · neg 15,4% · neu 5,2%
  implicit: aspect 3,3% (782) · opinion 1,2% (283)
  quad/review mean 2,73 (max 36) · quad/segment mean 1,90 (max 23)
  ngôn ngữ: en 7.473 · vi 1.032 · other 205 · đa ngữ 86
  split review 6.156/1.320/1.320 · split segment 8.816/1.895/1.890

OVERLAP  gold hotel ⊂ pool: 3.399/3.399 (100%) · pool review thuộc hotel gold: 1.249.424
  cohort A (gold & ≥1000): 276 · B (gold & ≥300): 1.221 · C (gold & ≥100): 2.313

CONFOUNDER (đo theo tháng)
  reviewer VN:   89,7% (2022-03) → 23,0% (2023-03) → 19,7% (2024-03) → 16,1% (2025-01)
  % có text:     42,1% (2022-02) → 63,4% (2025-03)
  độ dài TB:     27,5 từ → 40,8 từ
  hotel/tháng:   549 (2022-02) → 8.535 (2025-02)
  first-month:   549 hotel bắt đầu 2022-02 · 2.025 hotel bắt đầu 2022-03  (dấu hiệu truncation)

FIELD PROVENANCE (mẫu 400k)
  pos-only  n=88.380  score TB 9,52  median 10  chỉ 1,2% có score ≤6
  neg-only  n= 5.839  score TB 5,33  median  5     59,0% có score ≤6
  both      n=137.196 score TB 8,28
  → polarity field là nhãn sentiment gần-xác-định  ⇒ (a) s gần như miễn phí; (b) review_score
    KHÔNG độc lập với nó ⇒ không được dùng cả hai cùng phía
```
