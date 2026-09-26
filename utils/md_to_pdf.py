"""
utils/md_to_pdf.py
Converts Markdown documents to beautifully styled PDFs using markdown and weasyprint.
"""

import os
import re
import markdown
import weasyprint

CSS = """
@page {
    size: A4;
    margin: 18mm 16mm 20mm 16mm;
    @bottom-right {
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 8pt;
        color: #718096;
    }
    @bottom-left {
        content: "Amazon ML Challenge 2026 — Team Playbook";
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 8pt;
        color: #718096;
    }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 9.5pt;
    line-height: 1.5;
    color: #24292e;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 19pt;
    font-weight: 700;
    color: #1a202c;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 12px;
    page-break-after: avoid;
}

h2 {
    font-size: 13.5pt;
    font-weight: 600;
    color: #2b6cb0;
    border-bottom: 1px solid #edf2f7;
    padding-bottom: 4px;
    margin-top: 18px;
    margin-bottom: 10px;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    font-weight: 600;
    color: #2d3748;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

h4 {
    font-size: 10pt;
    font-weight: 600;
    color: #4a5568;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
}

p {
    margin-top: 0;
    margin-bottom: 8px;
}

a {
    color: #3182ce;
    text-decoration: none;
}

code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 8.5pt;
    background-color: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 3px;
    padding: 1px 4px;
    color: #c53030;
}

pre {
    background-color: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 10px 12px;
    margin: 10px 0;
    overflow-x: auto;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    border: none;
    padding: 0;
    font-size: 8pt;
    line-height: 1.4;
    color: #2d3748;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
}

table th {
    background-color: #edf2f7;
    color: #2d3748;
    font-weight: 600;
    border: 1px solid #cbd5e0;
    padding: 6px 10px;
    text-align: left;
}

table td {
    border: 1px solid #e2e8f0;
    padding: 5px 10px;
    vertical-align: top;
}

table tr:nth-child(even) td {
    background-color: #f7fafc;
}

blockquote {
    border-left: 4px solid #4299e1;
    background-color: #ebf8ff;
    color: #2b6cb0;
    padding: 8px 12px;
    margin: 10px 0;
    border-radius: 0 4px 4px 0;
    page-break-inside: avoid;
}

blockquote p {
    margin: 0;
}

hr {
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 14px 0;
}

ul, ol {
    margin-top: 0;
    margin-bottom: 8px;
    padding-left: 20px;
}

li {
    margin-bottom: 3px;
}

.alert-important {
    border-left: 4px solid #dd6b20;
    background-color: #fffaf0;
    color: #7b341e;
}

.alert-warning {
    border-left: 4px solid #e53e3e;
    background-color: #fff5f5;
    color: #9b2c2c;
}

.alert-note {
    border-left: 4px solid #3182ce;
    background-color: #ebf8ff;
    color: #2b6cb0;
}
"""


def process_alerts(md_text: str) -> str:
    """Transform GitHub alerts (> [!NOTE], > [!IMPORTANT], > [!WARNING]) to styled blockquotes"""
    def replace_alert(match):
        alert_type = match.group(1).lower()
        content = match.group(2).strip()
        css_class = f"alert-{alert_type}" if alert_type in ["important", "warning", "note", "caution"] else ""
        return f'<blockquote class="{css_class}"><strong>{alert_type.upper()}:</strong> {content}</blockquote>'

    pattern = r'>\s*\[!(NOTE|IMPORTANT|WARNING|CAUTION|TIP)\]\s*\n((?:>.*\n?)+)'
    
    def repl(m):
        atype = m.group(1).upper()
        lines = [line.lstrip('>').strip() for line in m.group(2).strip().splitlines()]
        body = " ".join(lines)
        css_cls = f"alert-{atype.lower()}"
        return f'<blockquote class="{css_cls}"><strong>{atype}:</strong> {body}</blockquote>\n'

    return re.sub(pattern, repl, md_text)


def convert_md_to_pdf(md_path: str, pdf_path: str):
    print(f"Converting: {md_path} -> {pdf_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    md_content = process_alerts(md_content)

    html_body = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"]
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{CSS}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    html = weasyprint.HTML(string=full_html, base_url=os.path.dirname(md_path))
    html.write_pdf(pdf_path)
    print(f"  ✅ Created {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")


if __name__ == "__main__":
    docs_dir = "/home/shubham/Projects/Amazon ML Hackathon/docs"
    targets = [
        ("Team_A_Data_Pipeline_Roadmap.md", "Team_A_Data_Pipeline_Roadmap.pdf"),
        ("Team_B_Modeling_Submission_Roadmap.md", "Team_B_Modeling_Submission_Roadmap.pdf"),
        ("Team_Setup_Colab_Git_S3_Guide.md", "Team_Setup_Colab_Git_S3_Guide.pdf"),
        ("Git_SageMaker_Setup_Guide.md", "Git_SageMaker_Setup_Guide.pdf"),
        ("ai_prompt_playbook.md", "ai_prompt_playbook.pdf"),
    ]

    for md_file, pdf_file in targets:
        md_full = os.path.join(docs_dir, md_file)
        pdf_full = os.path.join(docs_dir, pdf_file)
        if os.path.exists(md_full):
            convert_md_to_pdf(md_full, pdf_full)
        else:
            print(f"Warning: {md_full} not found")
