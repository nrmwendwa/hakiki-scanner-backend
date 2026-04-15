# Hakiki Scanner API - Production Backend

A production-ready FastAPI backend for face authenticity detection using a pre-trained EfficientNet model. Detects real vs AI-generated vs suspicious faces with high accuracy.

## Features

✅ **Fast Image Analysis** - GPU-accelerated inference with CUDA support  
✅ **RESTful API** - Clean FastAPI endpoints with automatic documentation  
✅ **Production Ready** - Comprehensive error handling and validation  
✅ **CORS Enabled** - Configured for frontend integration  
✅ **Auto API Docs** - Interactive Swagger UI at `/docs`  
✅ **Health Checks** - Built-in monitoring endpoints  
✅ **Detailed Logging** - Complete request/response logging  

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU inference)
- Model file: `models/efficientnet3class_full_model.pth`

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Verify model exists**
   ```bash
   ls -lh models/efficientnet3class_full_model.pth
   ```

4. **Start the API**
   ```bash
   python main.py
   ```

   The API will start at `http://localhost:8000`

5. **Test the API**
   - Interactive docs: http://localhost:8000/docs
   - Health check: `curl http://localhost:8000/health`
   - API info: `curl http://localhost:8000/info`

## API Endpoints

### Health & Info

- **GET `/health`** - Health check
  ```bash
  curl http://localhost:8000/health
  ```
  Response:
  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "version": "1.0.0"
  }
  ```

- **GET `/info`** - API and model information
  ```bash
  curl http://localhost:8000/info
  ```

### Prediction

- **POST `/predict`** - Analyze image for authenticity
  ```bash
  curl -X POST -F "image=@face.jpg" http://localhost:8000/predict
  ```
  
  Request: Multipart form with `image` file  
  Response:
  ```json
  {
    "verdict": "real",
    "confidence": 87.45,
    "scores": {
      "real": 87.45,
      "suspicious": 8.92,
      "fake": 3.63
    }
  }
  ```

  **Verdicts:**
  - `real` - Genuine/real face
  - `suspicious` - Potentially manipulated or unclear
  - `fake` - AI-generated or synthetic face

## Configuration

Environment variables in `.env`:

```bash
# API Configuration
API_HOST=0.0.0.0      # Bind address
API_PORT=8000         # Port number
DEBUG=False           # Debug mode (disable in production)
WORKERS=4             # Number of worker processes

# Model Configuration
MODEL_PATH=models/efficientnet3class_full_model.pth
DEVICE=cuda           # 'cuda' or 'cpu'

# File Upload
MAX_UPLOAD_SIZE_MB=10

# CORS
CORS_ORIGINS=http://localhost:8080,http://localhost:3000

# Logging
LOG_LEVEL=INFO        # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Production Deployment

### Using Gunicorn (Recommended)

1. **Install Gunicorn**
   ```bash
   pip install gunicorn
   ```

2. **Start with Gunicorn**
   ```bash
   gunicorn main:app \
     --workers 4 \
     --worker-class uvicorn.workers.UvicornWorker \
     --bind 0.0.0.0:8000 \
     --access-logfile - \
     --error-logfile -
   ```

### Using Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create model directory
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Start API
CMD ["gunicorn", "main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

Build and run:
```bash
docker build -t hakiki-scanner-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models hakiki-scanner-api
```

### Using Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/efficientnet3class_full_model.pth
      - DEVICE=cpu  # Use cpu for CPU-only systems
      - WORKERS=4
      - API_HOST=0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Start with:
```bash
docker-compose up
```

### Nginx Configuration

Example Nginx reverse proxy setup:

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

## Performance Tuning

### GPU Optimization
```bash
# Enable GPU with cuDNN
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDNN_ENABLED=1
```

### Model Optimization
- Use `DEVICE=cuda` for GPU inference (~10-50ms per image)
- Use `DEVICE=cpu` for CPU inference (~100-500ms per image)

### API Optimization
- Increase `WORKERS` for parallel requests (recommended: 2-4 per CPU core)
- Use Gunicorn's `--worker-connections` to handle more concurrent clients
- Enable HTTP/2 in reverse proxy for multiplexing

## Monitoring

### Logs
```bash
# Real-time logs
tail -f access.log

# View specific errors
grep ERROR logs/*.log
```

### Metrics to Monitor
- Request latency: Should be <100ms for GPU, <500ms for CPU
- Error rate: Should be <0.1%
- Model memory: Typically 100-200MB
- API memory: 300-500MB total

### Health Monitoring
```bash
# Check API health
watch -n 5 "curl -s http://localhost:8000/health | jq"

# Model info
curl http://localhost:8000/info | jq
```

## Troubleshooting

### Model Loading Fails
```
Error: Model file not found
```
- Check `MODEL_PATH` in `.env`
- Verify model file exists: `ls -lh models/efficientnet3class_full_model.pth`

### CUDA Not Available
```
GPU: False, using CPU
```
- Check CUDA installation: `nvidia-smi`
- Set `DEVICE=cpu` in `.env` if no GPU available

### Port Already in Use
```
OSError: [Errno 48] Address already in use
```
- Change `API_PORT` in `.env`
- Or kill existing process: `lsof -ti:8000 | xargs kill -9`

### CORS Errors
- Update `CORS_ORIGINS` in `.env` to include your frontend URL
- Example: `CORS_ORIGINS=http://localhost:8080,https://yourdomain.com`

## API Response Examples

### Successful Prediction (Real Face)
```json
{
  "verdict": "real",
  "confidence": 92.34,
  "scores": {
    "real": 92.34,
    "suspicious": 5.12,
    "fake": 2.54
  }
}
```

### Suspicious Image
```json
{
  "verdict": "suspicious",
  "confidence": 68.45,
  "scores": {
    "real": 25.30,
    "suspicious": 68.45,
    "fake": 6.25
  }
}
```

### AI-Generated Face
```json
{
  "verdict": "fake",
  "confidence": 88.92,
  "scores": {
    "real": 3.45,
    "suspicious": 7.63,
    "fake": 88.92
  }
}
```

### Error Response
```json
{
  "error": "Invalid file format. Allowed formats: jpeg, jpg, png, webp",
  "status_code": 400
}
```

## Frontend Integration

Update your frontend to point to the API:

```javascript
// React example
const analyzeImage = async (file) => {
  const formData = new FormData();
  formData.append("image", file);
  
  const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    body: formData,
  });
  
  const result = await response.json();
  console.log(result.verdict, result.confidence);
};
```

## Development

### Enable Debug Mode
```bash
DEBUG=True python main.py
```

### Run Tests (when available)
```bash
pytest tests/ -v
```

### Code Style
```bash
# Format code
black main.py config.py model_service.py

# Check linting
pylint main.py
```

## License

MIT - See LICENSE file

## Support

For issues and feature requests, please open an issue on the GitHub repository.
