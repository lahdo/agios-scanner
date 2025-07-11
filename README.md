# Agios PDF Scanner 🧪📄

A multiprocessing Python tool that scans every **page** of every **PDF** in a
folder for Agios-related drugs, trials, disease areas, mechanisms, and safety
signals.  
It performs both **regex/literal** and **semantic (embedding)** matches and
produces:

* **`report.xlsx`** – sortable table of all hits (PDF → page → term)
* **`summary.txt`** – one-page aggregate of counts / flagged pages  
  (instantly skimmable)

---

## 1 · Quick start

```bash
git clone https://github.com/your-org/agios-pdf-scanner.git
cd agios-pdf-scanner


# 1.  Create and activate a virtual-env (optional but recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\\Scripts\\Activate.ps1

# 2.  Install dependencies
pip install -r requirements.txt   # PyMuPDF, openai, pandas, rich, scikit-learn …

# 3.  Add your OpenAI key
cp .env.example .env
# edit .env   →  OPENAI_API_KEY="sk-..."

# 4.  Point the scanner at a folder of PDFs
python scanner.py /absolute/path/to/pdfs

## 2 · Quick start
