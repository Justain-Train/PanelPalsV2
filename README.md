# PanelPals V2 - Backend

FastAPI backend for the PanelPals V2 Webtoon OCR → TTS pipeline.

## Architecture

This is the **backend-only** implementation. The frontend (Chrome extension, Puppeteer screenshot capture) is maintained separately.

**Pipeline Flow:**
```
External Client (Frontend) → FastAPI Backend → Google Vision API (OCR)
                                              ↓
                                         Text Bubble Grouping
                                              ↓
                                         ElevenLabs TTS (Parallel)
                                              ↓
                                         Audio Stitching
                                              ↓
                                         WAV File Output
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env` and configure:
- **Google Vision API**: Set `GOOGLE_APPLICATION_CREDENTIALS` to your service account JSON path
- **ElevenLabs API**: Set `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID`

### 3. Google Cloud Setup

1. Create a Google Cloud project
2. Enable the Vision API
3. Create a service account
4. Download the JSON credentials file
5. Update `GOOGLE_APPLICATION_CREDENTIALS` in `.env`

### 4. ElevenLabs Setup

1. Sign up at [ElevenLabs](https://elevenlabs.io/)
2. Get your API key from the dashboard
3. Choose a voice ID (or use the default narrator voice)
4. Update `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` in `.env`

## Running the Server

### Development Mode

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Or use Python directly:

```bash
python -m backend.main
```

### Production Mode

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs (dev mode only)
- **Health Check**: http://localhost:8000/health

## Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=backend --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# E2E tests only
pytest -m e2e
```

## Project Structure

```
backend/
├── __init__.py           # Package initialization
├── main.py               # FastAPI app entry point
├── config.py             # Configuration management
├── routers/              # API route handlers (to be added)
├── services/             # Business logic (OCR, TTS, etc.)
└── models/               # Pydantic models for requests/responses

tests/
├── conftest.py           # Pytest fixtures
├── test_main.py          # Main app tests
├── test_config.py        # Configuration tests
└── ...                   # Feature-specific tests
```

## Configuration Reference

See `.env.example` for all available configuration options. Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `BUBBLE_MAX_VERTICAL_GAP` | Max vertical gap (px) for same bubble | 35 |
| `BUBBLE_MAX_CENTER_SHIFT` | Max horizontal shift for same bubble | 40 |
| `AUDIO_PAUSE_DURATION_MS` | Pause between text bubbles (ms) | 500 |
| `MAX_IMAGES_PER_REQUEST` | Max images per batch | 50 |
| `RATE_LIMIT_PER_MINUTE` | API rate limit per client | 10 |

## Security Features

- **CORS**: Restrictive origin policy
- **Rate Limiting**: Per-client request throttling
- **Input Validation**: Max file sizes and batch limits
- **Security Headers**: XSS, clickjacking, MIME-sniffing protection

## Next Steps

The skeleton is ready. Next implementations:
1. OCR endpoint (`POST /ocr/process`)
2. Text bubble grouping algorithm
3. TTS integration
4. Audio stitching engine

---

**Instruction File**: See `.github/copilot-instructions.md` for complete project specification.
