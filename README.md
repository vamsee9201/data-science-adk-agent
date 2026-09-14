# Fieldnote — conversational data-science agent

Fieldnote is a local-first web app for exploring CSV datasets through conversation. A Google ADK
coordinator handles focused questions and delegates broader exploratory analysis or baseline
modeling to an embedded analyst agent. All calculations run through curated Python tools; the
model cannot execute arbitrary code.

## Features

- Strict UTF-8 CSV uploads up to 25 MB and four offline scikit-learn samples
- Dataset profiling, missingness, duplicates, outliers, correlations, grouping, and filtering
- Statistical tests and server-rendered charts
- Preview-and-confirm cleaning operations with an immutable original dataset
- Guarded classification and regression baselines with model comparisons and diagnostics
- Temporary, isolated sessions with streamed analysis progress
- Framework-free responsive browser UI served by FastAPI

## Setup

Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev
mkdir -p .secrets
cp ../lean-rag/ai-lab-fasa.json .secrets/vertex-service-account.json
```

The credential directory is git-ignored. Never commit or expose the service-account JSON. The
project ID is read from that file automatically. Configuration can be overridden with the values
shown in `.env.example`.

Start the complete application with one command:

```bash
make dev
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765). Port 8765 is used because another
local ADK service commonly occupies port 8000.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_APPLICATION_CREDENTIALS` | `.secrets/vertex-service-account.json` | Vertex service account |
| `GOOGLE_CLOUD_LOCATION` | `global` | Vertex location |
| `GOOGLE_MODEL` | `gemini-3.8-flash` | Coordinator and analyst model |
| `MAX_UPLOAD_MB` | `25` | Maximum CSV size |
| `SESSION_TTL_MINUTES` | `60` | Idle session lifetime |
| `AGENT_TURN_TIMEOUT_SECONDS` | `180` | Maximum duration of one agent turn |

## Tests

```bash
make test
```

Tests do not call Vertex. To run the opt-in credential smoke test:

```bash
RUN_VERTEX_SMOKE=1 uv run pytest -m vertex
```

This is an exploratory analysis tool. Its statistics and model results should be reviewed before
being used for consequential decisions, and associations should not be interpreted as causation.
