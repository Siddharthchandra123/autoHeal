import Link from 'next/link';
import { getExperiments } from '../lib/api';
export const dynamic = 'force-dynamic';
export default async function Home() {
  let experiments;
  try { experiments=await getExperiments(); }
  catch {return <><h1>Experiment Overview</h1><p className="subtitle">Unable to reach the local viewer API. Start MLflow on port 5000 and FastAPI on port 8000.</p><div className="panel empty">No data is fabricated in offline mode.</div></>}
  return <><div className="eyebrow">MISSION CONTROL / EXPERIMENTS</div><h1>Model training, under control.</h1><p className="subtitle">Inspect traced experiments, trials and diagnoses from your local MLflow server.</p>
    <div className="stats"><div className="stat"><strong>{experiments.length}</strong><span>EXPERIMENTS</span></div><div className="stat"><strong>Local</strong><span>TRACKING SOURCE</span></div><div className="stat"><strong>0</strong><span>DEPLOY ACTIONS</span></div></div>
    <section className="panel"><div className="section-heading"><h2>Tracked experiments</h2><span>Refresh page for latest data</span></div>
    {experiments.length ? experiments.map(e=><Link className="experiment" key={e.experiment_id} href={`/experiments/${e.experiment_id}`}><div><b>{e.name}</b><p>Experiment ID {e.experiment_id}</p></div><span className="status ok">{e.lifecycle_stage}</span><span className="chevron">→</span></Link>):<div className="empty">No experiments recorded yet.</div>}
    </section><p className="disclaimer">Security note: do not expose this local viewer or unauthenticated MLflow server to a shared network.</p></>;
}
