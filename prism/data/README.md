# data/ — dữ liệu thô (không commit; 1,4 GB)

```text
data/raw/
├── hotel_booking_unlabeled.jsonl   # pool 1.949.604 review Booking.com · 10.631 hotel
│                                   # 2022-02 → 2025-03 · crawl một lần 2025-03
├── train.jsonl                     # 8.816 segment · gold quad · hotel-disjoint
├── dev.jsonl                       # 1.895 segment
├── test.jsonl                      # 1.890 segment   (tổng 23.995 quad)
└── hotel_absa_labeled.jsonl        # ⚠️ CHỈ là metadata review — KHÔNG chứa quad,
                                    #    không dùng làm nguồn nhãn
```

Gold metadata/ảnh nằm ở repo `hamos-mabsa` (cấu hình `PRISM_HAMOS_ROOT`,
mặc định `../hamos-mabsa`): `reviews_with_dates.jsonl`, `quads.jsonl`,
`segments.jsonl`, `images.jsonl`, `data/images/` (9.219 ảnh, 986 MB).

Số liệu audit đầy đủ: `docs/method_spec_aspect_drift.md` §2 và Phụ lục A.
Quy tắc bất biến: **mọi dòng pool trùng gold (`in_gold=true`) không bao giờ được
train/self-train/infer** — blocklist dựng ở Module A.
