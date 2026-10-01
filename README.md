# PanelPals V2 - Backend

FastAPI backend that turns a Webtoon chapter (an ordered list of panel images) into a single narrated MP3.

This is the **backend only**. The frontend (Chrome extension) is maintained separately.

## Pipeline

```
Panel images (one request per chapter)
   ↓  upload validation (count, size, format, pixel limit)
Google Vision OCR        - consecutive panels stitched into ≤8000px strips (~8x fewer billable calls),
   ↓                       up to 10 calls in parallel, non-Latin / junk tokens filtered
Text bubble grouping     - words → bubbles per panel
   ↓
Dialogue classifier      - trained model (random forest) with heuristic fallback;
   ↓                       sound effects, credits, signs etc. are dropped
Audio tags               - sound words (SOB, GIGGLE, SIGH...) become ElevenLabs tags ([crying], [chuckles]...)
   ↓
Page-break merging       - bubbles split across panels are joined
   ↓
TTS preprocessing        - all-caps → sentence case, ellipses and stammers kept
   ↓
ElevenLabs TTS           - parallel (plan's concurrency limit), retries with backoff
   ↓
Audio stitching          - one decode per clip, parallel, single-pass join
   ↓
MP3 response
```

A 166-panel chapter takes about 40 s end to end (OCR ~3-12 s depending on upload speed, TTS ~20 s, stitching ~4 s).

## Setup

### Docker

```bash
docker-compose up -d --build      # http://localhost:8000
make logs                         # other targets: build, up, down, shell, test, prod-build, prod-up
```

See [backend/docs/DOCKER_SETUP.md](backend/docs/DOCKER_SETUP.md) for details.

### Local

```bash
pip install -r requirements.txt
cp .env.example .env              # then fill in the values below
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Required in `.env`:
- `GOOGLE_APPLICATION_CREDENTIALS` - path to a Google Cloud service account JSON with the Vision API enabled
- `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`
- `ELEVENLABS_MAX_PARALLEL_REQUESTS` - your ElevenLabs plan's concurrency limit (too high causes dropped lines)
- `API_KEYS` - required when `DEBUG=false` (see [Security](#security))

`.env` is validated strictly: unknown variables stop the app from starting.

## API

### `POST /process/chapter`

Multipart form:

| Field | Description |
|---|---|
| `chapter_id` | 1-64 characters: letters, digits, `-`, `_` |
| `images` | Panel images in reading order (PNG, JPEG or WebP) |

Header `X-API-Key: <key>` is required unless the server runs with `DEBUG=true` and no keys configured.

```bash
curl -X POST http://localhost:8000/process/chapter \
  -H "X-API-Key: $KEY" -F chapter_id=ep1 \
  -F images=@panel_1.jpg -F images=@panel_2.jpg -o ep1.mp3
```

Returns `audio/mpeg` with headers:

| Header | Meaning |
|---|---|
| `X-Bubble-Count`, `X-Duration-MS` | Lines read and audio length |
| `X-Missing-Lines` | Line numbers TTS couldn't produce after retries (absent when complete) |
| `Server-Timing` | Per-stage timings (ocr, classify, tts, stitch, ...) |

Errors: `400` invalid input, `401` bad API key, `413` image too large, `429` rate limited (`Retry-After`), `500` processing failed, `503` service not configured.

### `GET /health`

Liveness check. Service configuration flags are only included when `DEBUG=true`. Swagger UI is at `/docs` in debug mode only.

## Configuration

All settings are in `backend/config.py` and documented in `.env.example`. The most important:

| Variable | Default | Description |
|---|---|---|
| `CLASSIFIER_MODE` | `heuristic` | `model` uses the trained classifier (falls back to the heuristic if missing) |
| `ML_MODEL_PATH` | `models/best_model.joblib` | `model_metadata.json` must sit next to it |
| `ML_THRESHOLD` | `0.4` | Min P(dialogue) to keep a line (below 0.5 favours keeping dialogue) |
| `OCR_MAX_PARALLEL_REQUESTS` | `10` | Concurrent Vision calls |
| `OCR_STITCH_MAX_HEIGHT` | `8000` | Panel strip height for OCR; `0` = one call per panel |
| `ELEVENLABS_MAX_PARALLEL_REQUESTS` | `5` | Concurrent TTS calls - match your plan |
| `TTS_MAX_RETRIES` | `3` | Retries for rate-limit / busy / 5xx / network errors |
| `TTS_SENTENCE_CASE` | `true` | Sentence case + ellipses for TTS (caps read as shouting) |
| `TTS_AUDIO_TAGS` | `true` | Sound words → ElevenLabs audio tags |
| `AUDIO_PAUSE_DURATION_MS` | `700` | Pause between lines |
| `MAX_IMAGES_PER_REQUEST` | `200` | Images per chapter request |
| `MAX_IMAGE_SIZE_MB` / `MAX_IMAGE_PIXELS` | `10` / `40000000` | Per-image limits |
| `RATE_LIMIT_PER_MINUTE` | `10` | Requests per client per minute (`0` disables) |
| `ML_COLLECT_DATA` | `false` | Save classified bubbles to `backend/ml/ml_data/raw` during requests |

## Security

- **API keys**: `X-API-Key` checked against `API_KEYS` (JSON list, e.g. `["key1","key2"]`) with a constant-time comparison. With no keys and `DEBUG=false` the endpoint returns 503 instead of running open. Generate keys with `openssl rand -hex 32`.
- **Rate limiting**: per API key (or IP), sliding one-minute window. In-memory per worker - use a shared store (Redis) when running multiple servers.
- **Input validation**: image count, file size, format and pixel dimensions are checked before any paid API call; `chapter_id` is restricted to safe characters.
- **Secrets**: `ELEVENLABS_API_KEY` and `API_KEYS` are `SecretStr` (masked in logs and errors).
- **Errors**: clients get generic messages; details go to the server log (and the response only in `DEBUG`).
- **Headers**: `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`; restrictive CORS (`ALLOWED_ORIGINS`).
- **Model file**: `joblib` model files can execute code when loaded - only load models you trained.

Not handled in the app (do at deployment): TLS, a total request-size cap at the reverse proxy, and dependency scanning (`pip-audit`).

## ML classifier

The dialogue/background classifier is trained on labelled bubbles from several series (`backend/ml/ml_data/raw/collected_<series>_ep<N>.csv`).

```bash
# 1. Download an episode's panels (personal testing only - don't redistribute)
python backend/utilities/download_episode.py "<webtoons.com episode viewer URL>"

# 2. Collect features (OCR + classification, no TTS)
python -m backend.ml.collect_ml_data screenshots/<series>_ep<N>

# 3. Label: fill the `label` column with dialogue/background (see rules below)

# 4. Combine (dedupes replayed panels across episodes), prepare, train
python -m backend.ml.combine_ml_data backend/ml/ml_data/raw/*.csv --out backend/ml/ml_data/combined/combined.csv
python -m backend.ml.prepare_dataset backend/ml/ml_data/combined/combined.csv
python -m backend.ml.train_model backend/ml/ml_data/prepared        # writes models/best_model.joblib + metadata
```

File names determine the series used for de-duplication, so keep the `collected_<series>_ep<N>.csv` pattern. `backend/ml/ml_data/` is gitignored because the CSVs contain copyrighted webtoon text - back it up separately.

**Labelling rules** - read aloud (`dialogue`): speech, thoughts, narration and time/location captions, game/system messages and informative interface text, character name cards, story-relevant letters/texts/screens, translator notes. Not read (`background`): sound effects and action/feeling words, credits and titles, content warnings and "to be continued", signs and posters, interface chrome, timer digits, gibberish and OCR junk.

Evaluate with every series held out in turn (train on the others, test on the unseen one) - a random split overstates accuracy because panels from the same episode look alike.

## Testing

```bash
pytest                       # coverage threshold is set in pytest.ini
pytest --no-cov -m unit      # fast unit tests only
```

`backend/tests/test_live_pipeline.py` and `test_multi_panel_pipeline.py` are scripts that call a running server (paid API calls); pytest skips them - run them directly.

## Project structure

```
backend/
├── main.py                 # FastAPI app, middleware, /health
├── config.py               # Settings (.env)
├── security.py             # API keys, rate limiting
├── routers/process.py      # POST /process/chapter
├── services/
│   ├── vision.py           # OCR, panel stitching, noise filter
│   ├── text_grouping.py    # words → bubbles
│   ├── text_box_classifier.py, language_features.py   # dialogue classifier + features
│   ├── audio_tags.py       # sound words → audio tags
│   ├── bubble_continuation.py                         # page-break merging
│   ├── text_preprocessing.py, tts.py, audio.py        # TTS text, ElevenLabs, stitching
│   └── timing.py           # per-stage timings
├── ml/                     # data collection, combine/prepare/train
│   └── ml_data/            # labelled training data (gitignored - contains webtoon text)
├── utilities/              # download_episode.py (panel downloader), start_ml_collection.py
└── tests/
models/                     # trained classifier + metadata
```

## Next steps

- Distance-based pauses and a per-line timing map, so audio can follow the reader's scrolling
- Background jobs, object storage and a per-chapter result cache for scaling
- A second-stage check (vision LLM or review) for the ~10% of lines the classifier is unsure about
