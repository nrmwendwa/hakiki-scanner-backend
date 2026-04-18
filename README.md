# Hakiki Scanner API

A FastAPI backend for multimodal content verification. Routes image, document, and text uploads through a Gemini-powered pipeline and returns a deterministic verdict (real / suspicious / fake) from a rule-based decision engine.

## Features

- **Multimodal validation** — images, PDFs/text docs, and raw text
- **Gemini-powered** — OCR, image analysis, claim extraction, and claim verification all go through the Gemini LLM gateway
- **Online claim verification** — cross-checks extracted claims against web search results
- **Deterministic decisions** — signals (claim truthfulness, image authenticity, source trust) are fused by a rule-based engine, not an LLM
- **Auto API docs** — interactive Swagger UI at `/docs`

## Quick Start

### Prerequisites

- Python 3.10+
- A Gemini API key ([aistudio.google.com](https://aistudio.google.com/app/apikey))

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and set GEMINI_API_KEY
   ```

3. **Start the API**
   ```bash
   python main.py
   ```

   The API will start at `http://localhost:8000`.

4. **Test the API**
   - Interactive docs: http://localhost:8000/docs
   - Health check: `curl http://localhost:8000/health`
   - API info: `curl http://localhost:8000/info`

## API Endpoints

### Health & Info

- **GET `/health`** — Health check
  ```json
  { "status": "ok", "version": "1.2.0" }
  ```

- **GET `/info`** — API and LLM configuration

### Validation

- **POST `/validate`** — Validate an uploaded image or document
  ```bash
  curl -X POST -F "file=@face.jpg" http://localhost:8000/validate
  curl -X POST -F "file=@article.pdf" http://localhost:8000/validate
  ```

- **POST `/validate-text`** — Validate a text statement
  ```bash
  curl -X POST http://localhost:8000/validate-text \
    -H "Content-Type: application/json" \
    -d '{"text": "Statement to verify"}'
  ```

Both endpoints return a `DecisionResult` with the input type, verdict, confidence, and per-signal evidence.

## Configuration

Environment variables in `.env`:

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
WORKERS=4

# LLM (Gemini)
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-1.5-flash

# File upload
MAX_UPLOAD_SIZE_MB=10

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Logging
LOG_LEVEL=INFO
```

Optional LLM routing overrides (defaults route all tasks to Gemini):

```bash
LLM_ROUTE_OCR=gemini-vision
LLM_ROUTE_IMAGE_ANALYSIS=gemini-vision
LLM_ROUTE_CLAIM_EXTRACTION=gemini-text
LLM_ROUTE_CLAIM_VERIFICATION=gemini-text
LLM_ROUTE_REASONING=gemini-text
LLM_TIMEOUT_S=30
LLM_MAX_RETRIES=3
LLM_RETRY_BACKOFF_S=1.5
```

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["gunicorn", "main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

```bash
docker build -t hakiki-scanner-api .
docker run -p 8000:8000 --env-file .env hakiki-scanner-api
```

### Nginx reverse proxy

```nginx
upstream fastapi_app {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name yourdomain.com;

    client_max_body_size 10M;

    location / {
        proxy_pass http://fastapi_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        proxy_pass http://fastapi_app;
        access_log off;
    }
}
```

## Troubleshooting

### Gemini API key not set
```
RuntimeError: GEMINI_API_KEY is not set. Add it to backend/.env.
```
Set `GEMINI_API_KEY` in `.env`.

### Port already in use
```
OSError: [Errno 48] Address already in use
```
Change `API_PORT` in `.env`, or kill the existing process: `lsof -ti:8000 | xargs kill -9`.

### CORS errors
Update `CORS_ORIGINS` in `.env` to include your frontend URL.

## Frontend Integration

```javascript
// Upload an image or document
const validateFile = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://localhost:8000/validate", {
    method: "POST",
    body: formData,
  });
  return res.json();
};

// Validate text
const validateText = async (text) => {
  const res = await fetch("http://localhost:8000/validate-text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return res.json();
};
```

## Development

### Debug mode
```bash
DEBUG=True python main.py
```

### Code style
```bash
black main.py config.py pipelines/ validation/ verification/ llm/
```

## License

MIT — see LICENSE file.
