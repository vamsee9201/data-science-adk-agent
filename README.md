# Data Science AI Agent

**A conversational data-science workspace that turns a CSV into an interactive analysis session.**

Upload a dataset—or choose an included sample—then ask questions in plain English. The application
can inspect data quality, calculate statistics, explore relationships, create charts, propose
cleaning steps, and compare baseline machine-learning models while keeping the work visible and
grounded in computed results.

Built as a portfolio project with **Google ADK**, **Gemini 3.8 Flash on Vertex AI**, **FastAPI**,
**pandas**, **SciPy**, and **scikit-learn**.

> The goal is not to replace a data scientist. It is to demonstrate how an AI agent can make a
> careful, repeatable first pass over a dataset while showing its work and respecting safety
> boundaries.

## Why I built this

Exploring an unfamiliar dataset usually means moving between notebooks, plotting libraries,
statistical functions, and modeling code. That is powerful, but it assumes the user already knows
which operations to run.

This project explores a different interface: **conversation as the control layer for data
science**. A user describes the outcome they want, and the agent decides which approved analysis
tools are needed. The calculations still happen in deterministic Python functions; the language
model coordinates the work and explains the results.

This creates a useful middle ground:

- More flexible than a fixed dashboard
- More approachable than starting from a blank notebook
- Safer than giving an agent unrestricted Python, SQL, or shell access
- More transparent than returning an answer without showing how it was produced

## What the application can do

| Capability | What the user experiences |
| --- | --- |
| Load data | Upload a UTF-8 CSV up to 25 MB or select Iris, Wine, Breast Cancer, or Diabetes |
| Understand a dataset | Preview rows, inspect shape and inferred types, detect identifiers, and summarize cardinality |
| Audit quality | Find missing values, duplicate rows, suspicious types, and numeric outliers |
| Explore patterns | Filter, sort, group, aggregate, inspect distributions, and calculate associations |
| Test relationships | Run supported statistical tests and report assumptions and limitations |
| Create visuals | Generate histograms, bar charts, line charts, scatter plots, box plots, and correlation heatmaps |
| Clean safely | Preview a transformation and its row impact before choosing **Apply** or **Discard** |
| Build baselines | Compare simple classification or regression candidates against a dummy baseline |
| Continue the conversation | Ask follow-up questions while retaining the active dataset and prior context |

## Product walkthrough

The screenshots below come from a real browser session using the included Iris dataset. When a
response required more than one screenshot, the images are kept together and presented in order.

### 1. Start with a sample or upload a CSV

The landing screen keeps the workflow focused: choose an offline sample or upload a CSV. The chat
remains unavailable until a dataset is active, which prevents analysis requests from running
without data.

![Landing workspace with CSV upload and sample dataset controls](docs/images/01-landing-page.png)

### 2. See the agent's work without exposing hidden reasoning

For a broad request—such as finding the most important relationships—the coordinator delegates to
the analyst. The interface streams a user-readable plan and live activity labels such as dataset
inspection, statistical testing, and chart generation.

It intentionally does **not** display internal prompts, chain-of-thought, raw tool payloads,
credentials, or stack traces.

![Workspace showing the active Iris dataset and live delegated analysis](docs/images/02-live-analysis-progress.png)

<details>
<summary>View the focused progress panel</summary>

![Focused progress panel showing delegation, planning, statistical testing, and visualization](docs/images/03-live-analysis-progress-detail.png)

</details>

### 3. Get evidence, not unsupported conclusions

The final response combines narrative findings with the values produced by the analysis tools. In
this example, the agent identifies the strongest Iris feature relationships, reports statistical
evidence, states that association is not causation, and creates a correlation heatmap.

![Relationship findings supported by correlations and a statistical test](docs/images/04-relationship-findings.png)

![Continuation with caveats, follow-up ideas, and the generated heatmap](docs/images/05-relationship-heatmap.png)

### 4. Ask follow-up questions in the same session

The active dataset and conversation context remain available across turns. Here, the user asks
which two measurements deserve attention first, and the agent answers using evidence from the
previous analysis rather than treating the question as a new, disconnected request.

![Grounded follow-up identifying the most useful measurements](docs/images/06-multiturn-follow-up.png)

### 5. Compare models against a meaningful baseline

Modeling requires an explicit target. For classification, the application compares a dummy
baseline with logistic regression and random forest. It reports holdout metrics, feature
importance, exclusions, caveats, and a confusion matrix—without making deployment or production
readiness claims.

![Classification model comparison and ranked feature importance](docs/images/07-model-comparison.png)

<details>
<summary>Continue through the interpretation and diagnostic chart</summary>

![Model interpretation, limitations, and suggested next steps](docs/images/08-model-explanation.png)

![Selected model metrics and confusion matrix](docs/images/09-model-metrics-chart.png)

</details>

### 6. Audit data before changing it

The quality workflow checks missingness, duplicates, potential identifiers, and outliers before
recommending action. It distinguishes a technically unusual value from a value that should
actually be removed.

![Data-quality audit covering missing values, duplicates, identifiers, and outliers](docs/images/10-data-quality-audit.png)

<details>
<summary>Continue through the outlier evidence and recommendation</summary>

![Numeric outlier evidence and recommended actions](docs/images/11-outlier-detail.png)

![Recommendation accompanied by a box plot](docs/images/12-data-quality-chart.png)

</details>

### 7. Require confirmation before modifying the dataset

A cleaning tool never silently changes the active dataframe. It creates a proposal containing the
operation, affected columns, row-count impact, previews, and warnings. The user must explicitly
apply or discard it. Reset restores the immutable original dataset.

![Duplicate-removal proposal waiting for user confirmation](docs/images/13-cleaning-proposal.png)

The application also serves generated charts as session-scoped PNG artifacts:

| Correlation heatmap | Classification confusion matrix |
| --- | --- |
| ![Server-rendered correlation heatmap](docs/images/correlation-heatmap.png) | ![Server-rendered confusion matrix](docs/images/baseline-confusion-matrix.png) |

## What makes it an agent?

This is more than a chat interface placed in front of pandas.

1. **The coordinator interprets the request.** It decides whether the question is focused enough
   for a direct tool call or broad enough to require delegated analysis.
2. **The analyst creates a plan first.** Broad exploratory and modeling requests must record a
   concise analysis plan before computation begins.
3. **The agent selects approved tools.** It can combine profiling, statistics, visualization,
   cleaning, and modeling tools based on the question and intermediate results.
4. **Python performs the computation.** The language model does not invent statistics; pandas,
   SciPy, and scikit-learn produce the values the answer relies on.
5. **Progress streams to the browser.** The user sees planning, tool activity, artifacts, warnings,
   and the final answer as newline-delimited events.
6. **Session state supports follow-ups.** The active dataset, conversation history, artifacts,
   pending transformation, and model results remain isolated within one temporary session.

Simple questions can be handled directly by the coordinator. Requests such as “analyze this
dataset,” “what relationships matter most?”, or “build a baseline model” are delegated to the
specialized analyst agent.

## Architecture, in plain English

| Layer | Responsibility | Implementation |
| --- | --- | --- |
| Browser interface | Dataset selection, upload, chat, progress, tables, charts, metrics, and confirmation controls | Framework-free HTML, CSS, and JavaScript |
| Web API | Validation, session endpoints, artifact authorization, cancellation, and NDJSON streaming | FastAPI |
| Agent orchestration | Routes focused questions, delegates broad analysis, and formats grounded responses | Google Agent Development Kit |
| Model | Understands requests, chooses tools, and explains computed results | Gemini 3.8 Flash through Vertex AI |
| Analysis engine | Performs profiling, statistics, charts, transformations, and baseline modeling | pandas, NumPy, SciPy, seaborn, matplotlib, scikit-learn |
| Temporary state | Keeps each session's dataset, history, proposal, models, and artifacts isolated | In-memory registry plus per-session temporary directory |
| Deployment controls | Limits scale, concurrency, requests, sessions, model turns, and output size | Cloud Run configuration and application guardrails |

### Coordinator agent

The coordinator handles conversational requests and focused operations such as previews,
missing-value checks, filtering, sorting, grouping, and individual charts. It delegates broader
investigations and modeling work to the analyst.

### Analyst agent

The analyst is designed for multi-step work. It must plan first, execute relevant tools, inspect
their outputs, and return a structured response with findings, evidence, warnings, assumptions,
artifacts, and suggested next steps.

### Curated tools instead of arbitrary execution

The agent does **not** have a SQL-writing tool, unrestricted Python execution, or shell access.
Every dataset operation goes through a function implemented and tested in this repository. This
keeps the agent useful while limiting what a prompt can cause it to execute.

## Analysis and modeling details

### Profiling and exploration

- Dataset preview, shape, inferred schema, cardinality, descriptive statistics, and identifiers
- Missing-value, duplicate-row, distribution, and outlier analysis
- Filtering, sorting, grouping, and aggregation
- Numeric correlations and categorical associations
- Supported statistical tests selected according to the referenced column types

### Visualizations

- Histogram
- Bar chart
- Line chart
- Scatter plot
- Box plot
- Correlation heatmap

Charts are rendered on the server as PNG files. Artifact URLs are scoped to the owning session and
stop working when that session expires or is deleted.

### Cleaning workflow

Supported proposals include missing-value handling, duplicate removal, type conversion, and
outlier handling. A proposal reports its expected effect before it can be applied. Rejection leaves
the active dataframe unchanged, and reset restores the original upload or sample.

### Baseline modeling

The user must name or unambiguously reference the target. The pipeline excludes the target,
detected identifiers, constants, and obvious leakage fields while reporting those exclusions.

| Classification | Regression |
| --- | --- |
| Dummy classifier | Dummy regressor |
| Logistic regression | Ridge regression |
| Random forest classifier | Random forest regressor |
| Accuracy, macro-F1, and ROC-AUC when valid | MAE, RMSE, and R² |
| Confusion matrix | Predicted-versus-actual chart |

Training uses a fixed seed of 42 and an 80/20 split, stratified for classification. Numeric
features receive median imputation; categorical features receive most-frequent imputation and
one-hot encoding. Linear models receive feature scaling. Training is capped at a deterministic
50,000-row sample.

These are exploratory baselines, not production models.

## Safety, privacy, and cost controls

The public interface can trigger paid model calls, so the project includes several layers of
protection.

| Control | Default | Why it exists |
| --- | --- | --- |
| Upload validation | UTF-8 CSV, 25 MB, 100,000 rows, 200 columns | Rejects unsupported or unexpectedly large inputs |
| Session lifetime | 60 minutes of inactivity | Removes temporary datasets and artifacts |
| Active sessions | 25 per instance | Bounds in-memory usage |
| Turns per session | 20 | Prevents one anonymous session from running indefinitely |
| Agent turns per hour | 30 per instance | Adds a rolling short-term model-usage brake |
| Agent turns per day | 100 per instance | Adds a rolling daily model-usage brake |
| Concurrent agent turns | 1 per instance | Limits simultaneous Vertex calls |
| Model output | 2,048 tokens | Bounds individual response size |
| Turn timeout | 180 seconds | Stops unusually long requests |
| Cloud Run scale | Maximum one instance in the recommended deployment | Places a ceiling on compute fan-out |

Additional safeguards:

- Uploaded data and generated artifacts are temporary and are deleted on expiry, explicit session
  deletion, or server shutdown.
- Cross-session artifact access is rejected.
- CSV errors are sanitized; stack traces and credentials are not returned to the browser.
- The agent is instructed not to fabricate values or describe an operation as complete without a
  confirming tool result.
- Statistical associations are not presented as causal conclusions.
- Cloud Run uses a dedicated runtime identity instead of shipping a service-account key inside the
  container.
- `.secrets/` is ignored by Git, Docker, and gcloud source uploads.

The in-memory hourly and daily limits reset when Cloud Run replaces the container. They are useful
application brakes, but they are not billing guarantees. A real public deployment should combine
them with a one-instance ceiling, Vertex quotas, billing budgets and alerts, and monitoring.

## Included datasets

Four scikit-learn datasets are available directly in the interface and work offline:

| Dataset | Typical task |
| --- | --- |
| Iris | Multiclass classification |
| Wine | Multiclass classification |
| Breast Cancer | Binary classification |
| Diabetes | Regression |

The repository also includes
[`Aurora_Customer_Churn.csv`](sample_data/Aurora_Customer_Churn.csv), a synthetic upload-ready
dataset with 242 rows and 14 columns. It contains mixed feature types, an identifier, a `churned`
target, and intentional missing values, duplicates, and outliers so the full workflow can be
demonstrated without using sensitive data.

Useful prompts for the Aurora dataset:

- “Perform a complete exploratory analysis of this dataset.”
- “Investigate missing values, duplicates, identifiers, and outliers. Propose cleaning without
  applying it.”
- “What factors are most strongly associated with churn? Include visualizations.”
- “Build a baseline classification model using `churned` as the target.”

## Run locally

### Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- A Google Cloud project with Vertex AI access
- A local service-account credential for development, or another supported Application Default
  Credentials setup

### Setup

```bash
git clone https://github.com/vamsee9201/data-science-adk-agent.git
cd data-science-adk-agent
uv sync --extra dev

mkdir -p .secrets
cp /path/to/service-account.json .secrets/vertex-service-account.json

make dev
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

The application reads the Google Cloud project from the credential file during local development.
The credential is never sent to the frontend and should never be committed.

## Configuration

Copy `.env.example` when overriding defaults.

<details>
<summary>Environment variables</summary>

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | Enables production-specific behavior when set to `production` |
| `GOOGLE_APPLICATION_CREDENTIALS` | `.secrets/vertex-service-account.json` | Local development credential path |
| `GOOGLE_CLOUD_LOCATION` | `global` | Vertex AI location |
| `GOOGLE_MODEL` | `gemini-3.8-flash` | Coordinator and analyst model |
| `MAX_UPLOAD_MB` | `25` | Maximum CSV size |
| `MAX_DATASET_ROWS` | `100000` | Maximum uploaded rows |
| `MAX_DATASET_COLUMNS` | `200` | Maximum uploaded columns |
| `SESSION_TTL_MINUTES` | `60` | Inactive-session lifetime |
| `MAX_ACTIVE_SESSIONS` | `25` | In-memory sessions per instance |
| `MAX_TURNS_PER_SESSION` | `20` | Model-backed turns per session |
| `MAX_AGENT_TURNS_PER_HOUR` | `30` | Rolling hourly turns per instance |
| `MAX_AGENT_TURNS_PER_DAY` | `100` | Rolling daily turns per instance |
| `MAX_CONCURRENT_AGENT_TURNS` | `1` | Simultaneous model-backed turns per instance |
| `MAX_MODEL_OUTPUT_TOKENS` | `2048` | Maximum output tokens per model response |
| `AGENT_TURN_TIMEOUT_SECONDS` | `180` | Deadline for one agent turn |
| `ALLOWED_HOSTS` | Local hosts and `*.run.app` | Accepted HTTP Host values |
| `EXPOSE_API_DOCS` | Local only | Controls `/docs` and `/openapi.json` outside production |

</details>

## Deploy to Cloud Run

The included [Dockerfile](Dockerfile) runs one Uvicorn worker as a non-root user. The following
example deploys an authenticated service that scales to zero, uses at most one instance, and runs
with a dedicated Vertex-only service identity.

```bash
gcloud iam service-accounts create ds-agent-runtime \
  --project PROJECT_ID \
  --display-name "Data Science Agent runtime"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member "serviceAccount:ds-agent-runtime@PROJECT_ID.iam.gserviceaccount.com" \
  --role roles/aiplatform.user

gcloud run deploy data-science-ai-agent \
  --source . \
  --project PROJECT_ID \
  --region us-central1 \
  --service-account ds-agent-runtime@PROJECT_ID.iam.gserviceaccount.com \
  --no-allow-unauthenticated \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 2 \
  --cpu 1 \
  --memory 1Gi \
  --timeout 300 \
  --no-cpu-boost \
  --cpu-throttling \
  --set-env-vars APP_ENV=production,GOOGLE_CLOUD_PROJECT=PROJECT_ID,GOOGLE_CLOUD_LOCATION=global,GOOGLE_MODEL=gemini-3.8-flash,MAX_UPLOAD_MB=25,MAX_DATASET_ROWS=100000,MAX_DATASET_COLUMNS=200,MAX_ACTIVE_SESSIONS=25,MAX_TURNS_PER_SESSION=20,MAX_AGENT_TURNS_PER_HOUR=30,MAX_AGENT_TURNS_PER_DAY=100,MAX_CONCURRENT_AGENT_TURNS=1,MAX_MODEL_OUTPUT_TOKENS=2048
```

Making the service public allows anonymous visitors to spend Vertex tokens. Do that only after
setting appropriate Cloud quotas, billing alerts, application limits, and monitoring.

## API surface

<details>
<summary>HTTP endpoints</summary>

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/sessions` | Create a temporary workspace |
| `DELETE` | `/api/sessions/{id}` | Delete a workspace and its artifacts |
| `GET` | `/api/samples` | List bundled datasets |
| `POST` | `/api/sessions/{id}/datasets/upload` | Upload a validated CSV |
| `POST` | `/api/sessions/{id}/datasets/sample` | Select a sample dataset |
| `GET` | `/api/sessions/{id}/dataset` | Read active dataset metadata |
| `POST` | `/api/sessions/{id}/chat` | Stream newline-delimited analysis events |
| `POST` | `/api/sessions/{id}/chat/cancel` | Request turn cancellation |
| `POST` | `/api/sessions/{id}/transformations/{proposal}/apply` | Apply a confirmed cleaning proposal |
| `DELETE` | `/api/sessions/{id}/transformations/{proposal}` | Discard a cleaning proposal |
| `POST` | `/api/sessions/{id}/dataset/reset` | Restore the immutable original dataset |

</details>

## Tests

```bash
make test
```

The standard suite mocks Vertex and does not make paid model calls. The current suite contains 23
passing tests. Two opt-in smoke tests are skipped by default:

```bash
RUN_VERTEX_SMOKE=1 uv run pytest -m vertex
```

The tests cover CSV validation, profiling, statistics, charts, transformations, modeling,
session isolation, artifacts, streamed event ordering, cancellation, guardrails, and agent
routing.

## Current scope

This is a focused, single-user portfolio MVP. It intentionally does not include durable chat
history, authentication UI, Excel upload, notebooks, a database, arbitrary code execution, or
multiple simultaneously active datasets.

Uploaded data and generated artifacts are temporary. Exploratory findings and baseline models
should not be used as the sole basis for consequential decisions, and statistical association
should not be interpreted as causation.
