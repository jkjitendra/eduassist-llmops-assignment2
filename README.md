# MLOps Assignment 2 — LLMOps

## API Cost, Latency, and Prompt Version Management

**Project:** EduAssist AI — Academic Question Answering Assistant

**Course:** MLOps

**LLM Provider:** Google Gemini API

**Model:** `gemini-3.5-flash-lite`

**MLflow Experiment:** `eduassist_llmops_assignment`

**Prompt Registry:** `eduassist_academic_qa`

**Production Prompt:** Version 3 (JSON Output)

## 1. Project Overview

This project demonstrates the operational management of an LLM-based academic question-answering application.

The assignment implements three tasks:

1. Measure Gemini API latency for five academic questions.
2. Extract token usage and calculate estimated API costs.
3. Register, evaluate, compare, and promote three prompt versions using MLflow.

All three prompt versions are evaluated using the same five academic questions.

**Difference Between MLOps and LLMOps**

MLOps focuses on managing the machine learning lifecycle, including data preparation, model training, deployment, monitoring, and retraining. LLMOps extends these practices to applications powered by large language models, where prompts, model APIs, inference costs, token consumption, response latency, and output quality must also be managed. Unlike traditional MLOps, LLMOps places particular emphasis on prompt versioning, structured output validation, and evaluating potentially non-deterministic responses. In this assignment, MLflow is used to register and compare prompt versions, track API performance and cost, and promote the most suitable version to production.

## 2. System Requirements

Tested configuration:

- Python: 3.13.5
- Operating system: macOS (Apple Silicon)
- RAM: 16 GB
- Google GenAI SDK: 2.34.0
- MLflow: 3.16.1
- Gemini Developer API version: v1beta (SDK default)

The project can also run on Linux (VM) using a compatible Python installation.

A GPU is not required because LLM inference is performed through the Gemini API.

Required libraries are listed in `requirements.txt`.

### Required Python Libraries

The project uses the following pinned dependencies:

- `mlflow==3.16.1`
- `google-genai==2.34.0`
- `pandas==3.1.0`
- `jupyterlab==4.6.5`
- `ipykernel==7.4.0`
- `python-dotenv==1.2.4`
- `nbconvert==7.17.2`

## 3. Project Structure

```text
eduassist-llmops-assignment2/
├── notebooks/
│   └── MLOps_Assignment_2.ipynb
├── outputs/
│   ├── MLOps_Assignment_2.html
│   ├── latency_tokens_cost.csv
│   ├── prompt_evaluation_results.csv
│   └── prompt_comparison.csv
├── screenshots/
│   ├── mlflow_experiment_runs.png
│   ├── mlflow_experiment_metrics.png
│   ├── mlflow_prompt_registry.png
│   ├── mlflow_prompt_comparison.png
│   └── mlflow_production_alias.png
├── report/
│   └── MLOps_Assignment_2_Report.pdf
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Generated working files such as `.venv`, `.env`, `mlflow.db`, and `mlartifacts/` are not required in the submission ZIP.

## 4. Environment Setup

Extract the submission ZIP and open a terminal in the extracted project directory.

### 4.1 macOS

Verify Python:

```bash
python3 --version
```

Create and activate the project environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 4.2 Linux (VM)

Install Python and its virtual-environment support if required.

For Ubuntu/Debian systems:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

Create and activate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For both operating systems, verify dependency compatibility:

```bash
python -m pip check
```

Python 3.13.5 was used for the submitted experiments. A compatible Python version is required to install the pinned dependencies.

## 5. Configure Environment Variables

Copy the example configuration:

```bash
cp .env.example .env
```

Edit `.env` and configure:

```dotenv
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-3.5-flash-lite
MLFLOW_TRACKING_URI=http://127.0.0.1:5050
```

Replace `YOUR_GEMINI_API_KEY` with a valid Gemini API key.

On macOS and Linux, restrict file permissions:

```bash
chmod 600 .env
```

Obtain an API key from:

https://aistudio.google.com/api-keys

The submitted experiment used a Gemini API Free Tier project.

Check free-tier eligibility and quotas before executing API requests. Do not enable paid billing unless explicitly intended.

Never commit or submit `.env` or actual API credentials.

## 6. Start the MLflow Tracking Server and UI

The following command starts the MLflow tracking server and web interface at http://127.0.0.1:5050.

Open Terminal 1 and navigate to the project root.

Activate the Python environment:

```bash
source .venv/bin/activate
```

Create the artifact directory:

```bash
mkdir -p mlartifacts
```

Start MLflow:

```bash
mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --artifacts-destination "$PWD/mlartifacts" \
  --host 127.0.0.1 \
  --port 5050
```

Open:

http://127.0.0.1:5050

The MLflow server uses SQLite to store experiment metadata, runs, metrics, and registered prompt versions.

Artifacts are stored in the `mlartifacts/` directory.

Keep this terminal running while executing the notebook.

## 7. Initialize the MLflow Experiment

Open Terminal 2.

Activate the Python environment:

```bash
source .venv/bin/activate
```

Run:

```bash
python - <<'PY'
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5050")

experiment_name = "eduassist_llmops_assignment"

experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

print("Experiment:", experiment_name)
print("Experiment ID:", experiment_id)
PY
```

This creates the required experiment if it does not already exist.

## 8. Execute the Jupyter Notebook

Register the Python environment as a Jupyter kernel:

```bash
python -m ipykernel install \
  --sys-prefix \
  --name eduassist-llmops \
  --display-name "Python (EduAssist LLMOps)"
```

Start JupyterLab from the project root:

```bash
python -m jupyter lab \
  --notebook-dir="$PWD" \
  --ip=127.0.0.1 \
  --port=8888 \
  --no-browser
```

Open the authenticated Jupyter URL printed by the terminal.

Open:

```text
notebooks/MLOps_Assignment_2.ipynb
```

Select the kernel:

```text
Python (EduAssist LLMOps)
```

Run the notebook cells sequentially from top to bottom.

Do not share the Jupyter authentication token.

## 9. Optional — Google Colab with macOS/Linux Runtime

Google Colab can be used as the notebook interface while Python executes on the Mac or Linux VM.

From the project directory, run:

```bash
python -m jupyter lab \
  --ServerApp.allow_origin='https://colab.research.google.com' \
  --ServerApp.allow_credentials=True \
  --ServerApp.port_retries=0 \
  --ServerApp.root_dir="$PWD" \
  --ip=127.0.0.1 \
  --port=8888 \
  --no-browser
```

Open:

https://colab.research.google.com/

Upload the assignment notebook and choose:

**Connect → Connect to local runtime**

Paste the authenticated Jupyter server URL in this format:

```text
http://127.0.0.1:8888/?token=YOUR_JUPYTER_TOKEN
```

Keep the MLflow and Jupyter servers running during execution.

The connected runtime can access files on the host computer. Only execute trusted notebooks.

When using Colab, download the final notebook and save it back into the project's `notebooks/` directory.

## 10. Experiment Reproduction

Execute notebook cells in their existing order.

### Task 1 — Latency Measurement

Five academic questions are sent using Prompt v1.

The implementation records:

- Question ID and prompt version
- Model name
- Request start and end timestamps
- Response latency
- Response status and finish reason
- Input, output, thinking, and total tokens

Responses are considered complete only when they contain nonempty text and finish with `STOP`.

The notebook calculates mean, median, minimum, maximum, and sample standard deviation of latency.

### Task 2 — Token and Cost Calculation

Token usage is extracted from the Task 1 responses.

The calculation uses the following standard paid-tier prices:

| Category      | Price per 1 Million Tokens |
| ------------- | -------------------------: |
| Input tokens  |                   USD 0.30 |
| Output tokens |                   USD 2.50 |

Pricing reference:

https://ai.google.dev/gemini-api/docs/pricing

Pricing reference date: November 2, 2026.

Currency: USD.

The notebook calculates the estimated cost of each request and projects the cost of 10,000 monthly requests.

The project used an eligible free-tier configuration. Standard paid-tier costs are estimates, not charges incurred.

### Task 3 — Prompt Registry and Evaluation

Three prompt versions are registered:

| Version | Purpose                                   |
| ------- | ----------------------------------------- |
| v1      | Basic academic answering                  |
| v2      | Clear, concise undergraduate explanations |
| v3      | Structured JSON responses                 |

The registered prompt name is:

```text
eduassist_academic_qa
```

All versions answer the same five questions.

The notebook logs these required MLflow metrics:

```text
avg_latency_seconds
avg_input_tokens
avg_output_tokens
avg_cost_per_request
valid_json
```

Additional metrics include `schema_valid_rate` and `avg_total_tokens`.

Prompt v3 responses are checked for valid JSON and the required fields:

```text
answer
confidence
key_points
```

The best-performing prompt is assigned the `production` alias.

The evaluation scripts reuse successfully completed responses from saved CSV files to avoid unnecessary Gemini API calls.

To perform an entirely fresh measurement, first archive or remove the existing raw response CSV files. This consumes additional Gemini API quota and produces new measurements that may differ from the submitted results.

## 11. Observed Results

The submitted evaluation produced the following measurements:

| Metric                   |         v1 |         v2 |         v3 |
| ------------------------ | ---------: | ---------: | ---------: |
| Avg latency (seconds)    |    2.43376 |    1.78156 |    1.27832 |
| Avg input tokens         |       23.8 |       47.8 |       88.8 |
| Avg output tokens        |      499.6 |      292.0 |      142.6 |
| Avg total tokens         |      523.4 |      339.8 |      231.4 |
| Avg paid-tier cost (USD) | 0.00125614 | 0.00074434 | 0.00038314 |
| Valid JSON rate          |         0% |         0% |       100% |
| Schema validity rate     |         0% |         0% |       100% |
| Recommended              |         No |         No |        Yes |

Prompt v3 achieved the lowest average latency and estimated paid-tier cost, while satisfying the required JSON schema for all five evaluated questions.

It was promoted to production through MLflow.

These are observed results, not guaranteed future performance.

## 12. Output Files

The main submission outputs are:

```text
outputs/latency_tokens_cost.csv
outputs/prompt_evaluation_results.csv
outputs/prompt_comparison.csv
outputs/MLOps_Assignment_2.html
```

The CSV files contain measured token usage, latency, estimated costs, and evaluation results.

The HTML export provides a readable copy of the completed notebook.

## 13. MLflow Screenshots

The `screenshots/` directory contains evidence of:

- Three experiment runs
- Logged evaluation metrics
- Registered prompt versions
- Prompt comparison
- Production alias

The production version can be retrieved using:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5050")

prompt = mlflow.genai.load_prompt(
    "prompts:/eduassist_academic_qa@production"
)

print(prompt.version)
```

Expected version after successful reproduction: 3.

## 14. Limitations

The experiment evaluates five academic questions per prompt version.

JSON and schema validity are measured automatically, but factual accuracy is not independently graded.

API latency may vary with network conditions and model processing.

The original v1 evaluation reused one naturally completed response generated with a 350-token ceiling; its other completed responses used 2048. The v2 and v3 evaluation requests used a 2048-token ceiling.

A larger evaluation dataset and standardized repeated measurements would provide stronger production evidence.

The Gemini API Free Tier has quota and eligibility restrictions that may change over time.

## 15. Security

Do not include the following in the submission:

```text
.env
.venv/
mlflow.db
mlartifacts/
```

Do not upload API keys, authentication tokens, personal information, or billing details.

## 16. References

- Gemini API documentation: https://ai.google.dev/gemini-api/docs
- Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing
- Gemini API versions: https://ai.google.dev/gemini-api/docs/api-versions
- Google GenAI SDK: https://googleapis.github.io/python-genai/
- MLflow documentation: https://mlflow.org/docs/latest/
- Google Colab runtime: https://research.google.com/colaboratory/local-runtimes.html
