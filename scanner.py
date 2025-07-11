#!/usr/bin/env python3
"""
Agios PDF Scanner – v3
======================

For every page in every PDF and every search term it now executes THREE
independent match strategies:

1. Regex / literal pattern
2. Embedding similarity  (cosine ≥ threshold)
3. Chat LLM judgement    (GPT-4o-mini by default)

Each successful hit is logged separately with its own `match_type`.

CLI flags let you turn off either semantic pass if you need speed:

    --no-embed         # skip embedding similarity
    --no-chat          # skip chat LLM

Everything else (cache, resumability, config.ini, etc.) is unchanged.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

# ---------------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    if not hasattr(fitz, "Document"):
        raise ImportError
except ImportError as exc:
    sys.exit(
        "PyMuPDF not installed (or name clash with another 'fitz').\n"
        "Fix with:\n"
        "  pip uninstall -y fitz frontend\n"
        "  pip install -U pymupdf\n"
        f"({exc})"
    )

import openai
import pandas as pd
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path)],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_cache(path: Path) -> Dict[str, list]:
    return json.loads(path.read_text()) if path.exists() else {}


def save_cache(cache: dict, path: Path) -> None:
    path.write_text(json.dumps(cache, indent=2))


def get_embedding(client, text: str, model: str) -> Sequence[float]:
    return client.embeddings.create(input=[text], model=model).data[0].embedding


# ---------------------------------------------------------------------------
# Multiprocessing initialiser
# ---------------------------------------------------------------------------

_worker_client = None


def init_worker(api_key: str) -> None:
    global _worker_client
    _worker_client = openai.OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Chat-LLM helper
# ---------------------------------------------------------------------------


def page_mentions_term(page_text: str, term: str, model: str) -> bool:
    """
    Ask the chat model to answer yes/no. Retries with exponential back-off.
    """
    prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert biomedical literature reviewer. "
                "Answer ONLY 'yes' or 'no'. Respond 'yes' if the page clearly "
                "mentions, describes, or implies the search term (directly or "
                "as a synonym). Otherwise respond 'no'."
            ),
        },
        {
            "role": "user",
            "content": f"Search term: {term}\n---\nPage text:\n{page_text[:8000]}",
        },
    ]
    for attempt in range(5):
        try:
            r = _worker_client.chat.completions.create(
                model=model, messages=prompt, max_tokens=1, temperature=0
            )
            return r.choices[0].message.content.strip().lower().startswith("y")
        except openai.RateLimitError:
            wait = 2 ** attempt
            logging.warning("Rate-limited (%s). Sleeping %s s …", term, wait)
            time.sleep(wait)
    return False


# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------


def process_pdf(
    pdf_path: Path,
    term_map: Dict[str, str],
    term_embeds: Dict[str, Sequence[float]],
    cfg: configparser.ConfigParser,
    use_embed: bool,
    use_chat: bool,
    chat_model: str,
    max_pages: int | None = None,
) -> List[dict]:
    """
    Outer loop: PDF
    Middle loop: page
    Inner loop: term
    """
    out: List[dict] = []

    ctx = int(cfg["SEARCH"]["context_window"])
    threshold = float(cfg["SEARCH"]["embedding_threshold"])
    emb_model = cfg["SEARCH"]["embedding_model"]

    patterns = {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in term_map}

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        logging.error("Cannot open %s (%s)", pdf_path.name, exc)
        return out

    last = len(doc) if max_pages is None else min(max_pages, len(doc))

    for page_idx in range(last):
        page = doc.load_page(page_idx)
        text = page.get_text()

        if not text.strip():
            out.append(
                {
                    "pdf_filename": pdf_path.name,
                    "page_number": page_idx + 1,
                    "term": "N/A",
                    "category": "N/A",
                    "quotation": "ERROR: No text extracted; manual review.",
                    "match_type": "Extraction Error",
                }
            )
            continue

        # one embedding per page (if embedding enabled)
        page_emb = (
            get_embedding(_worker_client, text, emb_model) if use_embed else None
        )

        for term, category in term_map.items():
            pat = patterns[term]

            # --- Regex ------------------------------------------------------
            for m in pat.finditer(text):
                s, e = m.span()
                out.append(
                    {
                        "pdf_filename": pdf_path.name,
                        "page_number": page_idx + 1,
                        "term": term,
                        "category": category,
                        "quotation": text[max(0, s - ctx) : min(len(text), e + ctx)],
                        "match_type": "Regex",
                    }
                )

            # --- Embedding --------------------------------------------------
            if use_embed:
                score = cosine_similarity([page_emb], [term_embeds[term]])[0][0]
                if score >= threshold:
                    out.append(
                        {
                            "pdf_filename": pdf_path.name,
                            "page_number": page_idx + 1,
                            "term": term,
                            "category": category,
                            "quotation": text[: ctx * 2],
                            "match_type": f"Embedding ({score:0.2f})",
                        }
                    )

            # --- Chat LLM ---------------------------------------------------
            if use_chat:
                if page_mentions_term(text, term, chat_model):
                    out.append(
                        {
                            "pdf_filename": pdf_path.name,
                            "page_number": page_idx + 1,
                            "term": term,
                            "category": category,
                            "quotation": text[: ctx * 2],
                            "match_type": f"LLM ({chat_model})",
                        }
                    )

    doc.close()
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def save_report(res: list[dict], path: Path) -> None:
    if not res:
        logging.info("No matches found.")
        return
    df = (
        pd.DataFrame(res)[
            [
                "pdf_filename",
                "page_number",
                "term",
                "category",
                "match_type",
                "quotation",
            ]
        ]
        .sort_values(["pdf_filename", "page_number", "term"])
    )
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    logging.info("Report → %s", path)


def save_summary(res: list[dict], pdf_cnt: int, path: Path) -> None:
    hits = [r for r in res if "Error" not in r["match_type"]]
    errs = [r for r in res if "Error" in r["match_type"]]
    by_cat: Dict[str, int] = {}
    for r in res:
        by_cat[r.get("category", "N/A")] = by_cat.get(r.get("category", "N/A"), 0) + 1

    lines = [
        f"PDFs scanned           : {pdf_cnt}",
        f"Total matches          : {len(hits)}",
        f"Pages needing attention: {len(errs)}",
        "Matches by category:",
    ]
    for cat, cnt in sorted(by_cat.items(), key=lambda t: t[1], reverse=True):
        lines.append(f"  {cat}: {cnt}")

    path.write_text("\n".join(lines))
    logging.info("Summary → %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agios PDF Scanner (regex + embed + chat)")

    p.add_argument("pdf_dir", nargs="?", help="Folder with PDFs (or set in config)")
    p.add_argument("--config", default="config.ini", help="INI config")
    p.add_argument("--terms", help="Alternate JSON term list")
    p.add_argument("--out", help="Override report path")
    p.add_argument("--cache", help="Override cache path")
    p.add_argument("--log", help="Override log path")
    p.add_argument("--workers", type=int, help="Worker count")
    p.add_argument("--no-embed", action="store_true", help="Skip embedding similarity")
    p.add_argument("--no-chat", action="store_true", help="Skip chat LLM pass")
    p.add_argument("--chat-model", help="Chat model (default: gpt-4o-mini)")
    p.add_argument("--threshold", type=float, help="Cosine sim threshold")
    p.add_argument("--ctx", type=int, help="Context chars")
    p.add_argument("--max-pages", type=int, help="Debug: scan first N pages only")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    a = args()
    here = Path(__file__).parent.resolve()

    # --- config ------------------------------------------------------------
    cfg_path = (
        Path(a.config).expanduser()
        if Path(a.config).is_absolute()
        else here / a.config
    )
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)

    # --- resolve paths -----------------------------------------------------
    pdf_dir_str = a.pdf_dir or cfg["DEFAULT"].get("pdf_dir") or cfg["DEFAULT"].get(
        "pdf_directory"
    )
    if not pdf_dir_str:
        sys.exit("Provide pdf_dir (positional) or [DEFAULT] pdf_dir= in config.")
    pdf_dir = Path(pdf_dir_str).expanduser().resolve()
    if not pdf_dir.exists():
        sys.exit(f"PDF directory not found: {pdf_dir}")

    out_path = Path(a.out or cfg["DEFAULT"]["output_file"]).expanduser().resolve()
    cache_path = Path(a.cache or cfg["DEFAULT"]["cache_file"]).expanduser().resolve()
    log_path = Path(a.log or cfg["DEFAULT"]["log_file"]).expanduser().resolve()
    summary_path = Path(cfg["DEFAULT"]["summary_file"]).expanduser().resolve()

    setup_logging(log_path)

    workers = a.workers or int(cfg["DEFAULT"]["workers"])
    use_embed = not a.no_embed
    use_chat = not a.no_chat
    chat_model = a.chat_model or cfg["SEARCH"].get("chat_model", "gpt-4o-mini")

    if not use_embed and not use_chat:
        sys.exit("Both semantic methods disabled (nothing beyond regex). Exiting.")

    if a.threshold is not None:
        cfg["SEARCH"]["embedding_threshold"] = str(a.threshold)
    if a.ctx is not None:
        cfg["SEARCH"]["context_window"] = str(a.ctx)

    # --- OpenAI ------------------------------------------------------------
    api_key = os.getenv("OPENAI_API_KEY") or cfg["DEFAULT"].get("openai_api_key")
    if not api_key:
        sys.exit("OPENAI_API_KEY missing (env or config).")
    main_client = openai.OpenAI(api_key=api_key)

    # --- terms -------------------------------------------------------------
    term_path = Path(a.terms or here / "search_terms.json").expanduser()
    if not term_path.exists():
        sys.exit(f"Term file not found: {term_path}")
    term_json = json.loads(term_path.read_text())
    term_map = {t.lower(): cat for cat, ts in term_json.items() for t in ts}
    logging.info("Loaded %d terms.", len(term_map))

    # --- embeddings (if needed) -------------------------------------------
    term_embeds = {}
    if use_embed:
        logging.info("Embedding terms …")
        emb_model = cfg["SEARCH"]["embedding_model"]
        term_embeds = {t: get_embedding(main_client, t, emb_model) for t in term_map}

    # --- discover PDFs -----------------------------------------------------
    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    logging.info("Found %d PDFs.", len(pdfs))

    # --- cache -------------------------------------------------------------
    cache = load_cache(cache_path)
    pending: List[Path] = []
    for pdf in pdfs:
        dig = sha256(pdf)
        if dig in cache:
            logging.info("✔  Cached: %s", pdf.name)
        else:
            pending.append(pdf)

    all_results = sum(cache.values(), [])

    # --- multiprocessing ---------------------------------------------------
    if pending:
        with Progress(
            SpinnerColumn(), "[progress.description]{task.description}", BarColumn(), TimeElapsedColumn()
        ) as pg:
            tk = pg.add_task("[cyan]Scanning …", total=len(pending))
            with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(api_key,)) as ex:
                futs = {
                    ex.submit(
                        process_pdf,
                        pdf,
                        term_map,
                        term_embeds,
                        cfg,
                        use_embed,
                        use_chat,
                        chat_model,
                        a.max_pages,
                    ): pdf
                    for pdf in pending
                }
                for fut in as_completed(futs):
                    pdf = futs[fut]
                    try:
                        res = fut.result()
                        all_results.extend(res)
                        cache[sha256(pdf)] = res
                    except Exception as exc:  # noqa: BLE001
                        logging.error("✖  %s failed (%s)", pdf.name, exc)
                    pg.update(tk, advance=1)

    # --- save --------------------------------------------------------------
    save_cache(cache, cache_path)
    save_report(all_results, out_path)
    save_summary(all_results, len(pdfs), summary_path)
    logging.info("Done.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
