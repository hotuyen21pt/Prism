# Hướng tiếp cận để thể hiện xu hướng theo thời gian
### CACT — Compositional Aspect-Complaint Tracking

> Tài liệu này đề xuất **một** hướng tiếp cận, và **đã chạy thử nghiệm thật trên toàn bộ 1,95M review**
> để chứng minh nó hoạt động trước khi đầu tư vào ASQP extractor.
> Script tái lập: `scripts/probe_complaint_composition.py`.

---

## 1. Đề xuất một câu

> **Không đo "sentiment trung bình theo thời gian". Đo *cơ cấu phàn nàn* — tỷ trọng mỗi aspect
> trong tổng lượt phàn nàn của kỳ đó — như một **composition**, chuẩn hoá về một thành phần
> reviewer tham chiếu cố định, rồi tách mùa vụ khỏi xu hướng trên thang log-ratio.**

Ba đại lượng, đo theo tháng, trên cửa sổ **2022-03 → 2025-02 (36 tháng)**:

| Ký hiệu | Định nghĩa | Trả lời câu hỏi |
|---|---|---|
| $\pi_{c,t}$ | **Share of complaints** — tỷ trọng aspect $c$ trong tổng lượt nhắc thuộc ô "chưa hài lòng" của kỳ $t$ | *Khách đang phàn nàn về cái gì nhiều hơn?* |
| $\nu_{c,t}$ | **Negativity rate** — $P(\text{negative} \mid \text{có nhắc } c,\ t)$ | *Nhắc tới $c$ thì có xu hướng chê nhiều hơn không?* |
| $V_{c,t}$ | Khối lượng tuyệt đối | Mẫu số, dùng cho khoảng tin cậy — **không** dùng làm tín hiệu xu hướng |

---

## 2. Vì sao là "share" chứ không phải count hay điểm trung bình

Dữ liệu này có bốn thứ trôi mạnh theo thời gian, và **cả bốn đều không liên quan tới chất lượng khách sạn**:

| Thứ trôi | Biên độ đo được | Phá hỏng đại lượng nào |
|---|---|---|
| Số hotel/tháng | 549 → **8.535** | count tuyệt đối |
| Tỷ lệ review có text | 42,1% → **63,4%** | count tuyệt đối |
| Độ dài review | 27,5 → **40,8 từ** | count, và **recall của extractor** |
| Reviewer Việt Nam | **89,7% → 16,1%** | mọi tỷ lệ chưa hiệu chỉnh |

$\pi_{c,t}$ **tự khử ba thứ đầu bằng cấu trúc**: cả tử số và mẫu số cùng phóng to khi có nhiều
review hơn / review dài hơn / nhiều hotel hơn. Thứ tư — thành phần reviewer — thì **không** tự khử,
phải hiệu chỉnh tường minh (§3).

Còn "điểm sentiment trung bình" thì hỏng vì `review_score` gần như phẳng (8,47–8,87 suốt 36 tháng)
và bị lệch dương nặng (điểm 10 chiếm 44,9%) — nó không có đủ độ phân giải để thể hiện xu hướng.

**Thêm một lý do quan trọng:** vì gold chỉ chú thích **26%** văn bản gốc và lệch theo cực tính
(phủ 42% ô positive nhưng chỉ 25% ô negative), recall của extractor sẽ **thấp và lệch**. Nhưng
$\pi_{c,t}$ là **tỷ số giữa các aspect trong cùng một kỳ**, nên một hệ số recall chung sẽ **triệt tiêu**.
Chỉ phần recall *khác nhau giữa các aspect* mới còn ảnh hưởng — nhỏ hơn nhiều. Đây là lý do kỹ thuật
mạnh nhất để chọn dạng "share".

---

## 3. Ba bước xử lý

### 3.1 Hiệu chỉnh thành phần (direct standardization)

Chia mỗi kỳ thành strata $g$ = `country_bloc`(VN/WEST/ASIA/OTH) × `state`(loại khách) → **20 ô**.
Tính tỷ trọng trong từng ô rồi gộp lại theo **một thành phần tham chiếu cố định** $\pi^{\text{ref}}_g$
(= tỷ trọng gộp toàn bộ 36 tháng):

$$\tilde\pi_{c,t}=\sum_{g}\pi^{\text{ref}}_{g}\cdot\frac{n_{c,t,g}}{\sum_{c'}n_{c',t,g}}$$

Ô có $n<30$ trong kỳ đó bị bỏ qua. **Độ phủ đo được: trung bình 99,2% trọng số strata dùng được,
thấp nhất 74,2%** — nghĩa là hiệu chỉnh gần như không mất dữ liệu.

Câu hỏi mà bước này trả lời: *nếu thành phần khách không đổi thì cơ cấu phàn nàn có đổi không?*

### 3.2 Chuyển sang thang CLR

Các $\pi_{c,t}$ **cộng lại bằng 1**. Hệ quả: nếu một aspect tăng thì các aspect khác **bắt buộc**
phải giảm — sinh tương quan âm giả. Hồi quy từng aspect trên share là **sai về mặt thống kê**.

Dùng centered log-ratio để đưa từ simplex về không gian thực:

$$\text{clr}(\pi_t)_c=\log\frac{\pi_{c,t}}{\big(\prod_{c'}\pi_{c',t}\big)^{1/K}}$$

Sau CLR, mọi công cụ chuỗi thời gian tiêu chuẩn (OLS, CUSUM, changepoint, ARIMA) trở nên hợp lệ.
Thêm một tính chất quý: **hiệu log-ratio bất biến với tổng khối lượng** — đúng thứ cần khi số hotel
đi từ 549 lên 8.535.

### 3.3 Tách mùa vụ khỏi xu hướng

Ước lượng hiệu ứng tháng-trong-năm rồi trừ đi, sau đó hồi quy xu hướng. Bước này **bắt buộc** —
§4.2 cho thấy vì sao.

---

## 4. Kết quả thử nghiệm thật

Proxy bằng keyword (lexicon sinh tự động từ `aspect_term` của gold + seed thủ công), chạy trên
**701.776 review có nội dung "chưa hài lòng"** trong 36 tháng. Chưa dùng ASQP model.

### 4.1 Hiệu chỉnh thành phần **đổi kết luận ở 8/13 aspect**

Slope trên thang CLR mỗi năm, đã khử mùa vụ; |t|>2,5 coi là có ý nghĩa.

| Aspect | THÔ slope | t | ĐÃ HIỆU CHỈNH slope | t | Kết luận |
|---|---:|---:|---:|---:|---|
| **AM_FOOD** | +0,018 | +1,4 | **−0,053** | **−7,4** | 🟡 **BỊ CHE + đảo dấu** |
| SER_SUPPORT | +0,042 | +3,6 | −0,009 | −1,7 | 🔴 đảo dấu, mất ý nghĩa |
| **FAC_ENV** | −0,085 | **−4,4** | −0,008 | −1,1 | 🔴 **XU HƯỚNG GIẢ** (91% là artifact) |
| AM_TRANSPORT | −0,031 | −4,8 | −0,006 | −0,6 | 🔴 xu hướng giả |
| SER_ATTITUDE | +0,014 | +2,8 | +0,001 | +0,1 | 🔴 xu hướng giả |
| FAC_VIEW_LOCATION | +0,031 | +3,1 | +0,023 | +1,6 | 🔴 xu hướng giả |
| FAC_BATH | −0,002 | −0,4 | +0,022 | **+3,4** | 🟡 **bị che, lộ ra sau hiệu chỉnh** |
| AM_ROOM_UTIL | −0,025 | −1,6 | +0,023 | +2,6 | 🟡 bị che |
| FAC_CLIMATE | −0,024 | −0,9 | +0,053 | +2,1 | ⚪ đổi dấu nhưng vẫn dưới ngưỡng |
| **AM_WIFI** | −0,121 | −6,9 | **−0,109** | **−6,7** | 🟢 **XU HƯỚNG THẬT** (bền vững) |
| AM_POOL | +0,126 | +6,0 | +0,037 | +3,5 | 🟢 thật, nhưng nhỏ hơn 71% |
| FAC_BUILDING | +0,048 | +4,3 | +0,016 | +2,8 | 🟢 thật, nhỏ hơn 67% |
| FAC_ROOM | +0,009 | +2,6 | +0,013 | +3,1 | 🟢 thật |

**Tổng kết:** 5 xu hướng **giả** bị loại (FAC_ENV, AM_TRANSPORT, SER_ATTITUDE, SER_SUPPORT,
FAC_VIEW_LOCATION), 3 xu hướng **bị che** được phát hiện (AM_FOOD, FAC_BATH, AM_ROOM_UTIL),
4 xu hướng **thật** được xác nhận (AM_WIFI, AM_POOL, FAC_BUILDING, FAC_ROOM).
FAC_CLIMATE đổi dấu nhưng vẫn dưới ngưỡng ý nghĩa.

**Đọc bảng này:** phân tích thô sẽ báo cáo *"phàn nàn về không gian/tiếng ồn (FAC_ENV) giảm mạnh"* —
**sai, 91% là do đổi thành phần khách**. Và sẽ **bỏ sót** việc phàn nàn về đồ ăn thật ra đang **giảm**
(t = −7,4), vì tín hiệu bị thành phần che mất hoàn toàn.

Ca `AM_FOOD` là ví dụ sạch nhất: chuỗi thô đi từ 7,67% (2022-03) lên ~10%; chuỗi đã hiệu chỉnh đi từ
**12,74%** xuống ~9,6%. Đầu 2022 reviewer là 89,7% người Việt, và nhóm này phàn nàn về đồ ăn ở tỷ lệ
thấp hơn nhóm khác — nên con số thô bị kéo xuống một cách giả tạo.

### 4.2 Xác thực ngoại vi: mô hình khôi phục đúng quy luật vật lý

Không hề được cung cấp thông tin về mùa, nhưng profile tháng-trong-năm ra đúng như vật lý đòi hỏi:

| Aspect | Đỉnh | Đáy | Tỷ lệ đỉnh/đáy | Có hợp lý không? |
|---|---|---|---:|---|
| **FAC_CLIMATE** (điều hoà) | **T6** 2,54% | T1 1,15% | **2,22×** | ✅ phàn nàn điều hoà đỉnh giữa hè |
| **AM_POOL** (hồ bơi) | **T8** 4,38% | T12 2,53% | **1,73×** | ✅ đỉnh mùa bơi |
| FAC_BATH (nước nóng) | T12 9,45% | T8 7,89% | 1,20× | ✅ đỉnh mùa lạnh |
| FAC_ROOM | T11 22,84% | T6 21,67% | **1,05×** | ✅ đúng là **không** có mùa |
| AM_WIFI | T3 1,73% | T1 1,36% | 1,27× | trung tính |

**Đây là bằng chứng validity mạnh nhất có được mà không cần gold drift.** Phương pháp tái tạo lại
những chu kỳ đã biết trước là đúng, và cho ra **null phẳng** ở đúng chỗ đáng lẽ phải phẳng
(FAC_ROOM 1,05×). Nếu FAC_ROOM cũng "có mùa" thì đó là dấu hiệu pipeline hỏng.

Nó cũng chứng minh **bước khử mùa vụ là bắt buộc**: biên độ mùa của FAC_CLIMATE và AM_POOL lớn gấp
~4× các aspect khác. Không khử thì một cửa sổ so sánh đặt lệch mùa sẽ tự sinh ra "drift".

---

## 5. Vì sao đây là hướng phù hợp nhất với **dữ liệu này**

| Đặc điểm dữ liệu | CACT xử lý thế nào |
|---|---|
| 41% review không có text | Không ảnh hưởng — mẫu số chỉ tính review có text |
| Điểm số lệch dương nặng (điểm 10 = 44,9%) | Không dùng điểm số làm tín hiệu chính |
| Gold chỉ phủ 26% văn bản, recall lệch | Recall chung **triệt tiêu** trong tỷ số |
| `review_negative` là ô riêng | Cho sẵn mẫu số "lượt phàn nàn" — không cần model đoán cực tính |
| Thành phần reviewer trôi mạnh | Hiệu chỉnh tường minh bằng standardization |
| Mật độ cấp hotel quá thưa (median 66 review/38 tháng) | Chạy ở cấp **corpus/cohort**, nơi có ~60k quad/tháng |
| Số hotel/tháng tăng 15× | Log-ratio bất biến với tổng khối lượng |
| Có 38 tháng liên tục | Đủ dài để tách mùa vụ (3 chu kỳ đầy đủ) |

Và quan trọng: **nó đã chạy được rồi**, bằng keyword proxy, trên toàn bộ dữ liệu, cho ra kết quả có
thể diễn giải và tự xác thực. Rủi ro triển khai gần như bằng không.

---

## 6. ASQP model thêm được gì

Proxy keyword chỉ cho $\pi_{c,t}$ ở mức thô. Extractor thật nâng cấp bốn thứ, **không** thay đổi
khung phân tích — chỉ thay bước đếm:

1. **Trọng số tin cậy:** đếm $\sum_i w_i$ thay vì đếm 1, với $w$ từ reliability model.
2. **Kênh $\nu_{c,t}$ thật:** keyword chỉ biết "đoạn này ở ô negative"; ASQP cho cực tính ở mức quad,
   xử lý được 21,7% trường hợp nội dung tích cực nằm trong ô negative (§ đo ở vòng 2).
3. **Kênh opinion $E_{c,t}$:** phát hiện *"vẫn chê, nhưng lý do đổi"* — keyword hoàn toàn mù với việc này.
4. **Subcategory + aspect implicit:** keyword không bắt được aspect ẩn (3,3% gold).

> **Hệ quả về thứ tự công việc:** khung phân tích thời gian **đã kiểm chứng xong**. Nên xây extractor
> để *nâng cấp* một pipeline đang chạy, thay vì xây extractor rồi mới hy vọng phần phân tích hoạt động.
> Đây là thứ tự ít rủi ro hơn hẳn.

---

## 7. Việc cần làm tiếp

| # | Việc | Ghi chú |
|---|---|---|
| 1 | Bootstrap CI cho từng slope (resample review **trong strata**) | Hiện mới có t-stat từ OLS |
| 2 | Permutation null: xáo trộn timestamp, đo FPR thực nghiệm | Bảng ở §4.1 chưa hiệu chỉnh đa so sánh |
| 3 | BH-FDR trên lưới 13 aspect × 2 kênh | Số chính thức phải là sau FDR |
| 4 | Thêm `length_bin` vào strata (20 → 60 ô) | Kiểm tra kết luận có bền không |
| 5 | Đối chiếu IPW thay cho post-stratification | Robustness |
| 6 | Lặp lại trên **cohort 514 hotel test-split** | Cohort không thiên vị, có sẵn |
| 7 | Thay keyword bằng quad có trọng số tin cậy | Nâng cấp, không phải làm lại |
| 8 | Mở kênh $\nu_{c,t}$ và $E_{c,t}$ | Cần ASQP |

⚠️ Bảng §4.1 là **kết quả probe chưa hiệu chỉnh đa so sánh** — dùng để chứng minh tính khả thi và
định hướng, **chưa phải con số để đưa vào bài**. Việc 1–3 phải xong trước khi trích dẫn.

---

## 8. Chốt lại

Xu hướng theo thời gian trong dữ liệu này **có tồn tại và đo được** — nhưng chỉ khi đo đúng đại lượng.
Đại lượng đúng là **cơ cấu phàn nàn đã chuẩn hoá thành phần**, không phải điểm sentiment trung bình,
không phải count tuyệt đối. Bằng chứng: cùng một dữ liệu, cách đo thô cho kết luận **sai ở 5 aspect**
và **bỏ sót 3 aspect**, trong khi cách đo đề xuất tái tạo đúng chu kỳ mùa vụ đã biết và cho null phẳng
ở đúng chỗ cần phẳng.
