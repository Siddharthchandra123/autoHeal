"""Build the cumulative v0.13.0 handbook PDF from the Markdown source."""
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    base = Path(__file__).resolve().parents[1]
    docs = base / "docs"
    for command in ("pandoc", "dot"):
        if not shutil.which(command):
            print(f"Missing documentation build tool: {command}", file=sys.stderr)
            return 2
    try:
        from weasyprint import HTML
    except ImportError:
        print("Install documentation dependency: pip install weasyprint", file=sys.stderr)
        return 2
    for source in sorted((docs / "diagrams").glob("*.dot")):
        subprocess.run(["dot", "-Tsvg", str(source), "-o", str(source.with_suffix(".svg"))], check=True)
    html_path = docs / "_handbook_render_v013.html"
    pdf_path = base / "AutoTune_AI_Handbook_v0.13.0.pdf"
    subprocess.run([
        "pandoc", "--standalone", "--from=markdown+link_attributes", "--to=html5",
        "--metadata=pagetitle:AutoTune AI - Technical Handbook v0.13.0",
        str(docs / "HANDBOOK_SOURCE.md"), "-o", str(html_path)
    ], check=True)
    HTML(filename=str(html_path)).write_pdf(str(pdf_path), stylesheets=[str(docs / "handbook.css")])
    html_path.unlink(missing_ok=True)
    print(f"Created {pdf_path} ({pdf_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
