"""Rebuild the PDF handbook from versioned Markdown and architecture diagrams.

Needs `pandoc`, `dot` (Graphviz), `weasyprint` in the author's environment.
Application execution/tests do NOT need these documentation dependencies.
"""
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    base = Path(__file__).resolve().parents[1]
    docs = base / "docs"
    for cmd in ("pandoc", "dot"):
        if not shutil.which(cmd):
            print(f"Missing documentation build tool: {cmd}", file=sys.stderr)
            return 2
    try:
        from weasyprint import HTML
    except ImportError:
        print("Install documentation dependency: pip install weasyprint", file=sys.stderr)
        return 2
    for file in sorted((docs / "diagrams").glob("*.dot")):
        subprocess.run(["dot", "-Tsvg", str(file), "-o", str(file.with_suffix(".svg"))], check=True)
    output_html = docs / "_handbook_render.html"
    output_pdf = base / "AutoTune_AI_Handbook_v0.8.2.pdf"
    subprocess.run(["pandoc", "--standalone", "--from=markdown+link_attributes", "--to=html5",
                    "--metadata=pagetitle:AutoTune AI - Technical Handbook v0.8.2",
                    str(docs / "HANDBOOK_SOURCE.md"), "-o", str(output_html)], check=True)
    HTML(filename=str(output_html)).write_pdf(str(output_pdf), stylesheets=[str(docs / "handbook.css")])
    output_html.unlink(missing_ok=True)
    print(f"Created {output_pdf} ({output_pdf.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
