# TABSA — Temporal Aspect-Based Sentiment Analysis trên review khách sạn

Nghiên cứu **aspect-level sentiment drift** trên 1,95M review Booking.com
(10.631 hotel, 2022-02 → 2025-03) với 23.995 gold quad multimodal (HAMoS).
Phương pháp: **PRISM** — trích xuất quad bán giám sát có provenance prior,
hiệu chỉnh độ tin cậy cross-modal bằng ảnh, và ước lượng drift **đã hiệu chỉnh
thành phần reviewer** với kiểm định thống kê (permutation null + BH-FDR).

## Cấu trúc repo

```text
├── README.md                # file này
├── pyproject.toml           # packaging (src-layout) + cấu hình pytest
├── requirements.txt         # deps nặng cho Module B/C (torch, transformers, open_clip)
├── data/                    # dữ liệu thô — KHÔNG commit (1,4 GB); xem data/README.md
│   └── raw/                 #   pool 1,95M + gold splits train/dev/test
├── docs/                    # tài liệu phương pháp (mọi số liệu đã audit)
│   └── archive/             #   bản cũ giữ để đối chiếu
├── models/                  # checkpoint huấn luyện — KHÔNG commit
├── outputs/                 # mọi sản phẩm pipeline — KHÔNG commit
│   ├── store/  extract/  reliability/  drift/  probe/
├── kaggle_notebooks/        # notebook chạy từng step trên Kaggle (01→11 + full)
├── scripts/                 # entrypoint Kaggle (kaggle_pipeline.py) & tiện ích (tải ảnh, probe)
├── src/
│   └── prism/               # package pipeline 4 module (A store · B extractor ·
│                            #   C reliability · D drift) + smoke_test
└── tests/                   # unit test (stdlib, không cần dữ liệu/torch)
```

## Bắt đầu nhanh

Trên Windows PowerShell, dùng Python 3.9+ và cài dependency:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Đặt repo `hamos-mabsa` cạnh repo này, hoặc trỏ tới vị trí khác:

```powershell
$env:PRISM_HAMOS_ROOT='C:\path\to\hamos-mabsa'
```

Repo HAMoS phải có metadata gold, annotation quad và thư mục ảnh theo mô tả
trong [data/README.md](data/README.md). Các bước train/infer còn cần tải model
Hugging Face và tạo checkpoint trong `models/`.

```powershell
$env:PYTHONPATH='src'

# 1. unit test (~0,2s — không cần dữ liệu, không cần torch)
python -m unittest discover tests

# 2. smoke test end-to-end trên dữ liệu thật (~2 phút, không cần torch) — PHẢI PASS
python -m prism.smoke_test
```

Sau đó làm theo thứ tự chuẩn trong **[src/prism/README.md](src/prism/README.md)** §7
(đường găng: store → data → train → eval → verifier go/no-go → infer → reliability
→ drift → injection). Cài môi trường cho phần cần GPU/torch: xem §0 của file đó.

Đường dẫn cấu hình tập trung tại `src/prism/config.py`; override bằng biến môi trường
`PRISM_TABSA_ROOT` · `PRISM_HAMOS_ROOT` · `PRISM_WORK_DIR` · `PRISM_MODEL_DIR`.

## Chạy trên Kaggle (GPU) — mỗi cell 1 step

Các bước cần GPU (`train · selftrain · infer`) chạy trên Kaggle. Dùng orchestrator
`scripts/kaggle_pipeline.py`: **mỗi cell chạy đúng 1 step** để lỗi step nào chỉ hỏng
step đó. Script tự tìm input theo tên file khắp `/kaggle/input` (kể cả output notebook
đã attach), stage vào đúng chỗ, và ghi output vào `/kaggle/working` (→ output notebook,
attach cho step sau). I/O đầy đủ từng step: **[docs/pipeline_io.md](docs/pipeline_io.md)**.

Thứ tự (bất biến): `store → data → train → selftrain → infer → photos →
c_verifier → c_apply_verifier → c_bridge → c_apply → drift`. Không đảo
`selftrain → infer` (infer chính thức phải dùng checkpoint tốt nhất từ self-train).

**Cell setup (đầu mỗi notebook):**

```python
!rm -rf /kaggle/working/Prism
!git clone --depth 1 https://github.com/hotuyen21pt/Prism.git /kaggle/working/Prism
P = "/kaggle/working/Prism/prism/scripts/kaggle_pipeline.py"
```

**Chuỗi notebook** (input notebook sau = output notebook trước, cộng store dùng chung):

| Notebook | Attach | Cell chạy |
|---|---|---|
| 1 · selftrain `[GPU]` | prism-dataset, dev-jsonl, store, checkpoint train | `!python {P} --step selftrain --cohort B-anchor --infer-batch 2 --infer-score-batch 4` |
| 2 · infer `[GPU]` | notebook-1 (ckpt student), store | `!python {P} --step infer --cohort T-unbiased --batch 2 --score-batch 4` |
| 3 · C+D `[CPU, net]` | notebook-2 (pool_quads), store, hamos-mabsa | `photos → c_verifier → c_apply_verifier → c_bridge → c_apply → drift` (mỗi step 1 cell, cùng `--cohort`) |

Sau mỗi notebook GPU bấm **Save Version (Save & Run All)** để giữ output.

**Hoặc chạy hết trong 1 cell** — tự bỏ qua step đã có output dưới `/kaggle/working`:

```python
!python {P} --step all --cohort T-unbiased      # thêm --force để chạy lại tất cả
```

Chưa có `hamos-mabsa` trên Kaggle thì bỏ Module C, chạy thẳng drift (tự fallback dùng
`pool_quads` với `w=conf_seq`): `!python {P} --step drift --cohort corpus`.

> Bước `train` (seed teacher) và `data`/`store` thường chạy local rồi upload output làm
> dataset; vẫn chạy được trên Kaggle qua `--step train` / `--step data` / `--step store`
> nếu attach dữ liệu thô + `hamos-mabsa`.

## Tài liệu phương pháp

| File | Nội dung |
|---|---|
| `docs/method_table_q1.md` | Bảng method hợp nhất, trạng thái từng module |
| `docs/method_proposal_q1.md` | Lập luận novelty PRISM + audit vòng 2 + venue |
| `docs/method_spec_aspect_drift.md` | Audit dữ liệu đầy đủ, formulation, evaluation protocol |
| `docs/approach_temporal_trend.md` | CACT keyword-probe — bằng chứng khả thi cho Module D |

Dữ liệu gold (metadata + 9.219 ảnh) lấy từ repo `hamos-mabsa` bên cạnh
(mặc định `../hamos-mabsa`, đổi bằng `$env:PRISM_HAMOS_ROOT`). Nếu chưa có repo
này, smoke test và các bước cần gold sẽ dừng ở bước kiểm tra dữ liệu đầu vào.
