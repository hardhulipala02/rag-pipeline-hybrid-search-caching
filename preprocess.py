#!/usr/bin/env python3
"""
preprocess.py — Convert SEC filings CSV rows into plain-text paragraphs.

Reads ./data/sec_filings.csv, takes the first MAX_DOCS rows, and writes
one .txt file per row into ./data/ so ingestion.py can embed them.
"""

import csv
from pathlib import Path

CSV_PATH = Path("./data/sec_filings.csv")
OUT_DIR = Path("./data")
MAX_DOCS = 30


FORM_TYPE_LABELS = {
    "8-K":    "a current report (Form 8-K)",
    "8-K/A":  "an amendment to a current report (Form 8-K/A)",
    "10-K":   "an annual report (Form 10-K)",
    "10-K/A": "an amendment to an annual report (Form 10-K/A)",
    "10KSB":  "an annual report for smaller reporting companies (Form 10-KSB)",
    "10-Q":   "a quarterly report (Form 10-Q)",
    "10-Q/A": "an amendment to a quarterly report (Form 10-Q/A)",
    "S-1":    "a registration statement (Form S-1)",
    "S-1/A":  "an amendment to a registration statement (Form S-1/A)",
    "SB-2/A": "an amendment to a small-business registration statement (Form SB-2/A)",
}


def format_date(raw: str) -> str:
    """Turn '2006-10-27' into 'October 27, 2006', fall back to raw string."""
    try:
        from datetime import datetime
        return datetime.strptime(raw.strip(), "%Y-%m-%d").strftime("%B %-d, %Y")
    except Exception:
        return raw.strip()


def row_to_paragraph(row: dict) -> str:
    accession  = row["Accession No"].strip()
    cik        = row["CIK"].strip()
    company    = row["Company Name"].strip()
    ticker     = row["Ticker"].strip()
    description = row["Description"].strip()
    form_type  = row["Form Type"].strip()
    filing_type = row["Filing Type"].strip()
    url        = row["Filing URL"].strip()
    filed_at   = format_date(row["Filed At"])

    form_label = FORM_TYPE_LABELS.get(form_type, f"a regulatory filing (Form {form_type})")

    # Company identification line
    ticker_clause = f" (ticker: {ticker})" if ticker else ""
    company_line = f"{company}{ticker_clause}, identified by CIK {cik},"

    # Filing action sentence
    para = (
        f"On {filed_at}, {company_line} submitted {form_label} "
        f"to the U.S. Securities and Exchange Commission. "
    )

    # Exhibit / filing type
    para += (
        f"The submission carries accession number {accession} and "
        f"includes a document classified as filing type {filing_type}. "
    )

    # Description (optional)
    if description:
        para += f"The exhibit or document is described as: {description}. "

    # URL
    if url:
        para += f"The full filing is publicly available at {url}."

    return para.strip()


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove any previously generated filing_*.txt files to avoid stale data
    removed = 0
    for old in OUT_DIR.glob("filing_*.txt"):
        old.unlink()
        removed += 1
    if removed:
        print(f"Removed {removed} existing filing_*.txt file(s).")

    written = 0
    with CSV_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if written >= MAX_DOCS:
                break

            paragraph = row_to_paragraph(row)
            out_path = OUT_DIR / f"filing_{written + 1:03d}.txt"
            out_path.write_text(paragraph, encoding="utf-8")
            written += 1
            print(f"  [{written:02d}/{MAX_DOCS}] {out_path.name}  —  {row['Company Name'][:50]}")

    print(f"\nDone. {written} text file(s) written to '{OUT_DIR}'.")


if __name__ == "__main__":
    main()
