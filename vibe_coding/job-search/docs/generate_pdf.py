#!/usr/bin/env python3
"""Convert cv-new.md to a styled PDF via HTML + Chrome headless."""

import subprocess
import sys
from pathlib import Path
import markdown

SCRIPT_DIR = Path(__file__).parent

MD_FILE = SCRIPT_DIR / "cv-new.md"
HTML_FILE = SCRIPT_DIR / "cv-new.html"
PDF_FILE = SCRIPT_DIR / "Ori_Mosenzon_CV.pdf"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #1a1a1a;
    max-width: 780px;
    margin: 0 auto;
    padding: 32px 40px;
}

h1 {
    font-size: 26pt;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 2px;
    letter-spacing: -0.5px;
}

h1 + p {
    font-size: 11pt;
    color: #475569;
    margin-bottom: 4px;
}

h1 + p + p {
    font-size: 9.5pt;
    color: #64748b;
    margin-bottom: 20px;
}

hr {
    border: none;
    border-top: 1.5px solid #e2e8f0;
    margin: 14px 0;
}

h2 {
    font-size: 11pt;
    font-weight: 700;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 18px;
    margin-bottom: 8px;
    padding-bottom: 3px;
    border-bottom: 1.5px solid #3b82f6;
    display: inline-block;
}

h3 {
    font-size: 10.5pt;
    font-weight: 600;
    color: #1e293b;
    margin-top: 10px;
    margin-bottom: 2px;
}

h3 + p strong, h3 + p {
    font-size: 9.5pt;
    color: #64748b;
    margin-bottom: 5px;
}

ul {
    margin: 4px 0 6px 18px;
}

li {
    margin-bottom: 2px;
    font-size: 10pt;
}

p {
    margin-bottom: 5px;
    font-size: 10.5pt;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9.5pt;
    margin: 8px 0;
}

th {
    background: #f1f5f9;
    font-weight: 600;
    text-align: left;
    padding: 5px 8px;
    border: 1px solid #e2e8f0;
    color: #334155;
}

td {
    padding: 4px 8px;
    border: 1px solid #e2e8f0;
    vertical-align: top;
}

tr:nth-child(even) td {
    background: #f8fafc;
}

strong {
    font-weight: 600;
    color: #0f172a;
}

em {
    color: #475569;
}

a {
    color: #3b82f6;
    text-decoration: none;
}

@media print {
    body { padding: 0; }
    h2 { page-break-after: avoid; }
    h3 { page-break-after: avoid; }
}
"""

def main():
    md_text = MD_FILE.read_text()
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    HTML_FILE.write_text(html)
    print(f"HTML written to {HTML_FILE}")

    result = subprocess.run([
        "google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf=" + str(PDF_FILE),
        "--no-pdf-header-footer",
        "--no-margins",
        str(HTML_FILE)
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"PDF created: {PDF_FILE}")
    else:
        print("Chrome error:", result.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
