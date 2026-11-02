import Link from 'next/link';
import RunTable from '../../../components/RunTable';
import { getRuns } from '../../../lib/api';
export const dynamic='force-dynamic';
export default async function ExperimentDetail({params}:{params:Promise<{id:string}>}) {
  const {id}=await params;
  let runs;
  try {runs=await getRuns(id)}catch {return <><h1>Experiment {id}</h1><p className="subtitle">Local MLflow query failed or experiment ID is invalid.</p></>}
  const best=runs.filter(r=>r.tags['autotune.role']==='trial' && typeof r.metrics.best_val_loss==='number').sort((a,b)=>a.metrics.best_val_loss-b.metrics.best_val_loss)[0];
  return <><div className="eyebrow"><Link href="/">EXPERIMENTS</Link> / {id}</div><h1>Experiment {id}</h1><p className="subtitle">Controller hierarchy, trial metadata and recorded metrics.</p>
  <div className="stats"><div className="stat"><strong>{runs.length}</strong><span>VISIBLE RUNS</span></div><div className="stat"><strong>{runs.filter(r=>r.tags['autotune.role']==='trial').length}</strong><span>TRAINING TRIALS</span></div><div className="stat"><strong>{best ? best.metrics.best_val_loss.toFixed(5) : '—'}</strong><span>LOWEST VALIDATION LOSS</span></div></div>
  <section className="panel"><div className="section-heading"><h2>Experiment runs</h2><span>Maximum 50 latest</span></div><RunTable runs={runs}/></section><p className="disclaimer">Validation scores on a tiny test dataset are not a production quality or safety assessment.</p></>;
}
