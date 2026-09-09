# PRISM — Module A (store) + Module B-data (text-to-text)
# Hai bước này chỉ dùng thư viện chuẩn Python (json/collections/gzip/re) ->
# không cần cài numpy/torch. Image gọn, build trong vài giây.
FROM python:3.12-slim

WORKDIR /app

# Chỉ copy source code. data/ và outputs/ được MOUNT lúc chạy (xem README lệnh docker run).
COPY src/ ./src/
# tests/ + scripts/ đi kèm: image sinh ra kết quả PHẢI chạy được
#   python -m unittest discover tests   và   python -m prism.smoke_test
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# config.py suy ra repo root = parents[2] = /app; set tường minh cho chắc.
ENV PYTHONPATH=/app/src \
    PRISM_TABSA_ROOT=/app \
    PYTHONUNBUFFERED=1

# Mặc định chạy Module A; đổi bằng cách truyền lệnh khác cho `docker run ... <cmd>`.
CMD ["python", "-m", "prism.module_a_store"]
