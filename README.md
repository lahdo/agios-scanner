# Agios PDF Scanner 🧪📄

Multiprocessing tool that scans every **page** of every **PDF** in a folder for
Agios-related drugs, trials, disease areas, mechanisms, and safety signals.

For every **page × term** the scanner runs three independent passes:

| # | method        | description                                                |
|---|---------------|------------------------------------------------------------|
| 1 | **Regex**     | fast literal / pattern match                               |
| 2 | **Embedding** | cosine-similarity between OpenAI embeddings                |
| 3 | **Chat LLM**  | asks *GPT-4o-mini* “Does this page mention TERM?” → yes/no |

Matches are logged with **PDF file**, **page number**, **term**, **category**,
**verbatim  context**, and **match_type** (`Regex`, `Embedding (…)`, `LLM (…)`).

Output:

* **`report.xlsx`** – sortable table of all hits (PDF → page → term)  
* **`summary.txt`** – one-page aggregate counts / flagged pages

---

## 1 · Quick start

```bash
git clone https://github.com/lahdo/agios-scanner.git
cd agios-pdf-scanner

# 1 · (optional) virtual-env
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\Activate.ps1

# 2 · dependencies
pip install -r requirements.txt

# 3 · OpenAI key
cp .env.example .env           # then edit .env → OPENAI_API_KEY="sk-..."

# 4 · run the scanner
python scanner.py /absolute/path/to/pdfs
```

## 2 · CLI cheatsheet

```bash
python scanner.py [pdf_dir] [options]

--no-embed           skip embedding similarity pass
--no-chat            skip chat LLM pass
--chat-model MODEL   override chat model (default: gpt-4o-mini)
--threshold 0.30     set cosine threshold for embedding pass
--ctx 350            characters of context before/after literal hit
--max-pages 5        only first 5 pages of each PDF (smoke-test)
--workers 8          number of CPU workers (default: 4)
--config config.ini  alternate INI file
--terms  my_terms.json  alternate term list

```

If you omit `pdf_dir`, the script falls back to `pdf_dir=` (or pdf_directory=)
in config.ini.

## 3 · Configuration (config.ini)

```bash
[DEFAULT]
pdf_dir        = /abs/path/to/pdfs
output_file    = report.xlsx
cache_file     = .cache.json
log_file       = scanner.log
workers        = 4
summary_file   = summary.txt

[SEARCH]
embedding_model     = text-embedding-3-small
embedding_threshold = 0.30
context_window      = 250
chat_model          = gpt-4o-mini
```

## 4 · Term list


`search_terms.json` holds the canonical lists for:
```bash

drug_names

disease_areas

clinical_trials

mechanisms

safety_signals


```
Swap with `--terms my_terms.json` or edit in place.


## 5 · Resumability

Each PDF is SHA-256 hashed ⇒ results cached in `.cache.json`.

Re-running the scanner skips unchanged files automatically.


## 6. PDFs

PDFs can be downloaded here: https://open.fda.gov/apis/other/approved_CRLs/