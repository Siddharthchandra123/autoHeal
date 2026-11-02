"""Append Phase 6B release notes to the cumulative v0.8.11 handbook."""
from pathlib import Path
import fitz
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'AutoTune_AI_Handbook_v0.8.11.pdf'
APPENDIX = ROOT / '_phase6b_appendix.pdf'
OUT = ROOT / 'AutoTune_AI_Handbook_v0.9.0.pdf'
C = {'ink':colors.HexColor('#193047'), 'blue':colors.HexColor('#1E5674'),
     'muted':colors.HexColor('#587086'), 'pale':colors.HexColor('#EEF5F8'),
     'aqua':colors.HexColor('#178C7D'), 'rule':colors.HexColor('#CEDFE8')}

styles={
 'title':ParagraphStyle('CoverTitle',fontName='Helvetica-Bold',fontSize=24,leading=28,textColor=C['ink'],spaceAfter=13),
 'h1':ParagraphStyle('SecTitle',fontName='Helvetica-Bold',fontSize=19,leading=23,textColor=C['ink'],spaceAfter=12),
 'h2':ParagraphStyle('H2',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=C['blue'],spaceBefore=12,spaceAfter=6),
 'body':ParagraphStyle('BodyX',fontName='Helvetica',fontSize=9.4,leading=14.2,textColor=C['ink'],spaceAfter=8),
 'muted':ParagraphStyle('MutedX',fontName='Helvetica',fontSize=8.5,leading=12.5,textColor=C['muted'],spaceAfter=8),
 'cell':ParagraphStyle('CellX',fontName='Helvetica',fontSize=8.2,leading=11.3,textColor=C['ink']),
 'cellhead':ParagraphStyle('CellHeadX',fontName='Helvetica-Bold',fontSize=8.3,leading=11.2,textColor=colors.white),
 'code':ParagraphStyle('CodeX',fontName='Courier',fontSize=7.8,leading=11.7,textColor=C['ink'],spaceAfter=6),
}

def p(text, sty='body'):
 return Paragraph(text,styles[sty])

def tbl(rows, widths):
 data=[[p(str(v),'cellhead' if i==0 else 'cell') for v in row] for i,row in enumerate(rows)]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),C['blue']),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,C['pale']]),
   ('GRID',(0,0),(-1,-1),.35,C['rule']),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9)]))
 return t

def bullet(text): return p('<font color="#178C7D"><b>&#8226;</b></font>  '+text)

def frame(canvas,doc):
 canvas.saveState()
 w,h=A4
 canvas.setFillColor(C['blue']);canvas.rect(0,h-13*mm,w,13*mm,fill=1,stroke=0)
 canvas.setFont('Helvetica-Bold',9);canvas.setFillColor(colors.white)
 canvas.drawString(21*mm,h-8.4*mm,'AUTOTUNE AI / TECHNICAL HANDBOOK / v0.9.0')
 canvas.setStrokeColor(C['rule']);canvas.line(21*mm,17*mm,w-21*mm,17*mm)
 canvas.setFillColor(C['muted']);canvas.setFont('Helvetica',8)
 canvas.drawString(21*mm,12.4*mm,'Phase 6B - Read-only MLflow Viewer | November 2, 2026')
 canvas.drawRightString(w-21*mm,12.4*mm,str(doc.page))
 canvas.restoreState()

story=[]
story += [Spacer(1,20*mm),p('Phase 6B | Experiment Intelligence Dashboard','title'),p('Cumulative handbook addendum - AutoTune AI v0.9.0','h2'),p('Architecture, implementation, security, test evidence, local deployment and next sprint.'),Spacer(1,8*mm),
 tbl([['Verification area','Current status'],['Windows CPU Hugging Face training','Previously user-verified'],['Windows CPU self-healing with injected failure','Previously user-verified; not natural overfitting'],['Windows localhost MLflow experiment logging','Previously user-verified: 3 run links'],['FastAPI read-only viewer','New code; 22 security/API tests passed'],['Complete Python suite','238 passing in build container'],['Next.js + TypeScript frontend','Source created; npm install/build NOT verified'],['Live Windows dashboard integration','Pending user acceptance']],[72*mm,96*mm]),Spacer(1,9*mm),
 p('What changed?','h2'),p('AutoTune AI now includes a dedicated experiment-viewer API and a dark-mode web dashboard source. This layer reads MLflow tracking records but does not modify experiments, source code, checkpoints, models, training jobs or deployment state.'),
 p('Why this architecture?','h2'),p('The existing training controller and its immutable local receipts remain the authority for training results. MLflow is a queryable observability view. The dashboard uses a narrow API boundary rather than granting a browser raw MLflow credentials or filesystem access.'),
 p('Evidence limitation:','h2'),p('The 238 tests were executed against the source snapshot, with a fake MLflow client for the viewer. The new UI was not installed or built in the package-generation environment. Real MLflow-to-dashboard acceptance still needs to run on Windows.')]
story.append(PageBreak())
story += [p('System Architecture','h1'),p('The data path is one-directional from the browser to existing experiment records. The previous autonomous training controller remains separate.'),
 Image(str(ROOT/'docs/diagrams/viewer_phase6b_v090.png'),width=170*mm,height=66*mm),Spacer(1,6*mm),
 p('Component connections','h2'),tbl([['Component','Responsibility'],['Hugging Face + PEFT trainer','Performs real fine-tuning on CPU and saves validated checkpoint files.'],['AutoTune recovery controller','Controls stop/retry logic, applies allowed hyperparameters and records local receipts.'],['MLflow + SQLite','Stores parent-child experiment lineage, metrics and metadata.'],['FastAPI viewer :8000','Fixed local MLflow SDK client; exposes constrained, read-only JSON.'],['Next.js viewer :3000','Server-rendered experiment summary, run details and simple loss charts.']],[61*mm,107*mm]),
 p('Read-only data boundary','h2'),p('The API intentionally has no POST/PUT/PATCH/DELETE training actions, arbitrary artifact retrieval, remote model import, filesystem path parameters, or commands that could execute models or Python code. The dashboard supports inspection only.')]
story.append(PageBreak())
story += [p('Data Contract and API Design','h1'),p('All API routes are confined to <b>GET</b> and use whitelisted, bounded inputs.'),
 tbl([['Route','Response / safeguard'],['/api/health','API status only; does not assert MLflow connectivity.'],['/api/experiments','Up to 100 experiment names and IDs.'],['/api/experiments/{id}/runs','1-100 latest runs; integer experiment ID.'],['/api/runs/{run_id}','Validated 32-character hex run identifier.'],['/api/runs/{run_id}/metrics/{metric}','Metric name allowlist, finite values, max 250 points.']],[72*mm,96*mm]),
 p('Response filtering','h2'),p('Run responses include only selected scalar metric values, training configuration fields, run names, parent IDs, attempt roles, diagnoses and fault-injection flags. Arbitrary MLflow metadata and artifact paths are omitted; this reduces accidental exposure but cannot substitute for an authenticated service.'),
 p('Lineage semantics','h2'),p('The MLflow parent is the AutoTune controller run; two child trials are associated by the <font face="Courier">mlflow.parentRunId</font> tag. The UI lists all runs, shows trial diagnoses, and compares recorded validation losses. Differences measured on the tiny test set do not constitute significant quality improvements.'),
 p('Failure behavior','h2'),p('Malformed IDs return HTTP 400, invalid limits return HTTP 422, unsupported write methods return HTTP 405, and internal MLflow exceptions return a sanitized HTTP 503. The underlying filesystem journal remains untouched.')]
story.append(PageBreak())
story += [p('Threat Boundaries and Security Controls','h1'),
 tbl([['Threat','Mitigation (implemented in viewer)'],['SSRF via arbitrary MLflow server','Startup rejects any target except http://127.0.0.1:port without query, path or credentials.'],['Credential/path leakage','Allowlisted run fields, truncated safe strings, generic 503 errors.'],['Unexpected API write operations','Read-only GET endpoints; 405 for POST/DELETE.'],['Malformed run or metric input','Strict ID regular expressions and metric-name allowlist.'],['Unexpected browser origins','Explicit localhost:3000 and 127.0.0.1:3000 CORS origins.'],['Host-header abuse','TrustedHost allowlist for localhost and testserver.'],['Unbounded charts','250-point maximum returned for metric histories.']],[57*mm,111*mm]),
 p('Not yet secure for multi-user deployments','h2'),p('The web UI, MLflow, and API do not implement authentication, fine-grained authorization, HTTPS termination, per-user scopes, request rate limiting or a formal vulnerability management gate. Keep them bound to loopback. Do not expose :3000, :5000 or :8000 to the internet or shared networks.'),
 p('Vulnerabilities vs training anomalies','h2'),p('Overfitting and unstable gradients are monitored as training-health conditions, not cybersecurity vulnerabilities. Dataset poisoning, malicious model provenance, credential compromise and dependency CVEs require separate scanning and incident policy. This phase does not claim those scanners exist.')]
story.append(PageBreak())
story += [p('Windows Deployment and Verification','h1'),p('Open three independent PowerShell terminals from the existing project directory. Preserve all previous outputs and MLflow SQLite data.'),
 p('Terminal A - MLflow','h2'),p('.\\.venv\\Scripts\\mlflow.exe server --backend-store-uri sqlite:///mlflow.db','code'),p('--host 127.0.0.1 --port 5000','code'),
 p('Terminal B - FastAPI viewer','h2'),p('.\\.venv\\Scripts\\python.exe -m pip install -r requirements-viewer.txt','code'),p('$env:AUTOTUNE_MLFLOW_URI = "http://127.0.0.1:5000"','code'),p('.\\.venv\\Scripts\\python.exe -m uvicorn autotune.viewer_api:app','code'),p('--host 127.0.0.1 --port 8000','code'),
 p('Terminal C - Next.js + TypeScript','h2'),p('cd .\\web','code'),p('npm install','code'),p('npm run typecheck','code'),p('npm run build','code'),p('npm run dev','code'),
 p('Acceptance conditions','h2'),bullet('GET http://127.0.0.1:8000/api/health responds with ready and mlflow: not probed.'),bullet('GET /api/experiments shows AutoTune-AI-CPU and its three prior run records.'),bullet('Next.js home, experiment and run detail pages render using the API, with validation-loss trend curves.'),bullet('Invalid paths and untrusted CORS origins are rejected; run details expose no secret fields.'),
 p('Troubleshooting','h2'),p('If the viewer shows unavailable, check that MLflow server is running and that the backend is bound to port 8000. If npm install fails, check network access and Node version; no frontend build verification is assumed. Do not disable Windows security controls to solve dashboard issues.')]
story.append(PageBreak())
story += [p('Tests, Next Milestones, Decision Record','h1'),
 tbl([['Test category','Outcome'],['Inherited ML/PEFT/recovery/MLflow Python tests','216 passing locally in build environment'],['New local API/SSRF/response-sanitization tests','22 passing'],['Total Python regression suite','238 passing'],['Node installation / Next.js build','Not completed in build environment (timeout)'],['Windows live MLflow -> FastAPI -> Next.js','Pending user-run acceptance']],[75*mm,93*mm]),
 p('Next execution order','h2'),bullet('<b>Finish Phase 6B acceptance:</b> Windows API + Next.js build and run-detail inspection using the real three-run MLflow experiment.'),
 bullet('<b>Security Sprint S0:</b> pip-audit/npm audit reporting, immutable Hugging Face model revisions, safe model serialization checks, dataset PII/security triage, deny/quarantine actions and test evidence.'),
 bullet('<b>Phase 6C:</b> richer metric streaming/refresh, comparative run views and authenticated controls only after explicit authorization design.'),
 bullet('<b>Phase 7-8:</b> Kubernetes/Kubeflow pipelines on CPU cluster and Katib HPO. Unsloth remains optional for future compatible GPU resources.'),
 p('Preserved project truth','h2'),p('The user has already verified a real Tiny GPT-2 CPU LoRA run, fresh adapter reload, checkpoint continuation, a fault-injected two-attempt self-healing loop, and successful local MLflow tracking. The first stop was injected, and the observed validation difference was negligible. No deployment approval or security-scanner completion is claimed.'),
 p('References','h2'),p('MLflow Tracking Client API: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html','muted'),
 p('Next.js App Router Data Fetching: https://nextjs.org/docs/app/getting-started/fetching-data','muted'),
 p('FastAPI CORS: https://fastapi.tiangolo.com/tutorial/cors/','muted')]

doc=SimpleDocTemplate(str(APPENDIX),pagesize=A4,leftMargin=21*mm,rightMargin=21*mm,topMargin=22*mm,bottomMargin=22*mm, title='AutoTune AI - Phase 6B v0.9.0')
doc.build(story,onFirstPage=frame,onLaterPages=frame)
base=fitz.open(str(BASE));addon=fitz.open(str(APPENDIX));base.insert_pdf(addon)
base.set_metadata({**base.metadata,'title':'AutoTune AI Technical Handbook v0.9.0','subject':'Cumulative architecture and Phase 6B read-only dashboard'})
base.save(str(OUT),garbage=4,deflate=True)
print('BASE',len(fitz.open(BASE)),'ADDED',len(addon),'TOTAL',len(fitz.open(OUT)))
