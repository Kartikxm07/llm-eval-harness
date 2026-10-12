# LLM Evaluation Harness

A small project to explore how well different large language models (LLMs) turn natural-language questions into SQL queries.

The goal is to build a fair and repeatable way to test two models on the same questions, run their SQL against a sample SQLite database, and compare their results.

> **Status:** Early-stage project — implementation and evaluation are in progress.

## Why this project?

LLMs can generate SQL from everyday questions, but their answers may differ in correctness and reliability. This project aims to compare two models using the same dataset and evaluation process rather than judging them from a few examples.

## Initial Goals

- Connect two LLMs through a common interface.
- Create a small set of natural-language questions with reference SQL queries.
- Use a SQLite e-commerce database for testing.
- Execute generated SQL safely and compare its results with expected results.
- Record useful metrics such as accuracy, execution success, and response time.
- Produce a simple report comparing the models.

These are project goals; features will be marked as complete as the project develops.

## Planned Workflow

1. Load a question from the evaluation dataset.
2. Ask both models to generate a SQL query.
3. Execute each query against the test database.
4. Compare the returned results with the reference answer.
5. Save the results and summarize the differences between the models.

## Tech Stack

- **Python** — main implementation language
- **SQLite** — sample database
- **LLM API** — to access the models being evaluated
- **JSON** — dataset and result storage
- **unittest** — testing as the project grows

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your GitHub repository URL and folder name.

### 2. Create a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

### 4. Configure API access

Create a local `.env` file for your API key and model settings. Keep real credentials private and do not commit `.env` to GitHub.

Example configuration:

```env
GROQ_API_KEY=your_api_key_here
MODEL_A_NAME=your_first_model
MODEL_B_NAME=your_second_model
```

The exact variable names should match the configuration used by the code.

## Evaluation Ideas

The evaluation will focus on:

- **Result accuracy:** whether the generated SQL returns the expected result.
- **Execution success:** whether the SQL runs without an error.
- **Latency:** how long each model takes to respond.
- **Model comparison:** how often the models agree and where their results differ.

Statistical comparisons may be added once the benchmark and results are ready.

## Roadmap

- [ ] Set up the project structure and dependencies
- [ ] Prepare the SQLite database and evaluation examples
- [ ] Implement a shared interface for both models
- [ ] Generate and execute SQL queries
- [ ] Compare model outputs with reference results
- [ ] Add evaluation metrics and reports
- [ ] Test the harness and document findings

## Project Structure

The structure may evolve as development continues.

```text
llm-evaluation-harness/
├── data/           # Database and evaluation examples
├── src/            # Evaluation code
├── tests/          # Automated tests
├── results/        # Generated evaluation reports
├── .env.example    # Example environment configuration
├── requirements.txt
└── README.md
```

## Current Focus

The initial focus is to build a reliable baseline: run the same questions through two models, check their SQL results, and make the comparison easy to reproduce.
