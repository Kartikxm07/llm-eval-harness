# Fair Two-Model LLM SQL Evaluation Harness

An enterprise-grade, statistically sound benchmark framework comparing two LLMs (**Model A: 120B parameter** vs **Model B: 20B parameter**) on Natural-Language-to-SQL generation using an identical SQLite e-commerce database and a curated 50-example golden benchmark dataset.

---

## 1. System Architecture

```text
                               +-----------------------------+
                               |  Natural Language Question  |
                               +--------------+--------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
        +--------------------------+                      +--------------------------+
        |   Model A Adapter        |                      |   Model B Adapter        |
        |   (openai/gpt-oss-120b)  |                      |   (openai/gpt-oss-20b)   |
        +------------+-------------+                      +------------+-------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              | (Generated SQL Queries)
                                              v
                               +-------------------------------+
                               |       src/sql_executor        | <--- Read-Only SQLite
                               |  (mode=ro + C-API Authorizer) |      (data/ecommerce.db)
                               +--------------+----------------+
                                              | (Actual Result Rows)
                                              v
+-----------------------------+ +-------------------------------+
|   data/golden_dataset.json  | |          src/graders          |
|    (Reference SQL Query)    | |  (Multiset / Ordered Compare) |
+--------------+--------------+ +--------------+----------------+
               | (Expected Rows)               |
               +-------------------------------+
                               |
                               v
                 +---------------------------+
                 |    src/statistics.py      |
                 | - Latency Profiling       |
                 | - McNemar's Paired Test   |
                 | - Cohen's Kappa Agreement |
                 | - Wilson Score 95% CIs    |
                 +-------------+-------------+
                               |
                               v
                 +---------------------------+
                 |      src/report.py        |
                 | - model_a_results.json    |
                 | - model_a_summary.txt     |
                 | - model_b_results.json    |
                 | - model_b_summary.txt     |
                 | - comparison_results.json |
                 | - comparison_summary.txt  |
                 +---------------------------+
```

---

## 2. Project Directory Structure

```text
eval &harness/
├── .env                         # API credentials and dual-model configurations
├── .env.example                 # Environment configuration template
├── .gitignore                   # Ignores credentials, virtualenvs, cache files
├── README.md                    # System architecture & operation documentation
├── requirements.txt             # Python dependencies
├── data/
│   ├── ecommerce.db             # Read-only benchmark SQLite database
│   └── golden_dataset.json      # 50 curated benchmark cases (questions & SQL)
├── results/
│   ├── model_a_results.json     # Detailed per-case results for Model A (120B)
│   ├── model_a_summary.txt      # Human-readable metrics dashboard for Model A
│   ├── model_b_results.json     # Detailed per-case results for Model B (20B)
│   ├── model_b_summary.txt      # Human-readable metrics dashboard for Model B
│   ├── comparison_results.json  # Paired statistical comparison JSON
│   └── comparison_summary.txt   # Side-by-side terminal & audit comparison
├── src/
│   ├── __init__.py              # Package marker
│   ├── contamination.py         # Benchmark leakage & prompt contamination auditor
│   ├── database.py              # Schema definition and database seed script
│   ├── dataset.py               # Dataset loader, structural validator, order inference
│   ├── graders.py               # Multiset (Counter) & ordered semantic result comparator
│   ├── llm_judge.py             # Optional qualitative LLM judge for SQL elegance
│   ├── metrics.py               # Accuracy, execution rate, category & difficulty breakdowns
│   ├── model.py                 # Common model interface, rate limiting, Groq/OpenAI adapters
│   ├── rate_limiter.py          # Thread-safe request pacer avoiding 429 rate limits
│   ├── report.py                # Serializer for single and comparative benchmark reports
│   ├── retry.py                 # Exponential backoff retry utility for transient errors
│   ├── runner.py                # Central orchestrator for two-model comparative evaluation
│   ├── sql_executor.py          # Secure read-only SQLite execution engine with authorizer
│   └── statistics.py            # Latency distributions, McNemar test, Cohen's kappa, CIs
└── tests/
    ├── test_evaluation.py       # 36 unit, security, and regression tests
    └── test_two_model_comparison.py # 18 tests for model abstraction, stats, and reports
```

---

## 3. Configuration

Configure `.env` in the project root:

```env
GROQ_API_KEY=your_actual_groq_api_key

# Model A (120B parameter model)
MODEL_A_PROVIDER=groq
MODEL_A_NAME=openai/gpt-oss-120b

# Model B (20B parameter model)
MODEL_B_PROVIDER=groq
MODEL_B_NAME=openai/gpt-oss-20b
```

---

## 4. Execution Commands (PowerShell)

### A. Run Full Offline Test Suite (54 Tests)
Runs in `< 0.5s` without making any external API calls:
```powershell
python -m unittest discover -s tests -v
```

### B. Validate Database & Dataset Integrity Offline
```powershell
python -m src.dataset
```

### C. Run Quick 3-Case Sanity Check for Both Models
```powershell
python -m src.runner --limit 3
```

### D. Run Full 50-Case Comparative Benchmark
```powershell
python -m src.runner
```

### E. Recreate Comparison Reports Without Re-calling APIs
```powershell
python -m src.runner --re-report
```

---

## 5. Statistical & Evaluation Metrics

1. **Result Accuracy**: % of cases where generated SQL returned rows and columns matching the ground truth.
2. **SQL Execution Rate**: % of queries that executed without SQLite syntax or runtime errors.
3. **Wilson Score 95% Confidence Interval**: Realistic binomial accuracy interval given sample size $N=50$.
4. **Latency Profiling**: Wall-clock response latency (mean, median, p90, p95).
5. **McNemar's Paired Test**: Hypothesis test evaluating whether the performance difference between Model A and Model B is statistically significant or attributable to random variance on a 50-case sample.
6. **Cohen's Kappa ($\kappa$)**: Quantifies inter-model agreement beyond chance.