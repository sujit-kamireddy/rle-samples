FROM python:3.12-slim

WORKDIR /app
COPY rle_openenv/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rle_openenv/__init__.py ./rle_openenv/__init__.py
COPY rle_openenv/server/ ./rle_openenv/server/
COPY rle/server/__init__.py ./rle/server/__init__.py
COPY rle/server/compliance.py ./rle/server/compliance.py
COPY rle/server/vendor/grader.py ./rle/server/vendor/grader.py

EXPOSE 8000
CMD ["uvicorn", "rle_openenv.server.app:app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
