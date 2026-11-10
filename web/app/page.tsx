import Link from 'next/link';
import { getExperiments } from '../lib/api';
export const dynamic = 'force-dynamic';

export default async function Home() {
  let experiments;
  try {
    experiments = await getExperiments();
  } catch {
    return <>
      <div className="eyebrow">CONTROL PLANE / SYSTEM STATUS</div>
      <h1>Autonomous ML.<br /><span style={{ color: '#f4df35' }}>Under your control.</span></h1>
      <p className="subtitle">Your local mission control for experiment intelligence, knowledge retrieval and self-healing model workflows.</p>
      <div className="stats">
        <div className="stat"><strong>OFFLINE</strong><span>MLFLOW CONNECTION</span></div>
        <div className="stat"><strong>LOCAL</strong><span>EXECUTION MODE</span></div>
        <div className="stat"><strong>—</strong><span>EXPERIMENTS LOADED</span></div>
      </div>
      <section className="panel"><div className="section-heading"><h2>Viewer connection</h2><span>WAITING FOR SERVICES</span></div><div className="empty">Unable to reach the local viewer API. Start MLflow on port 5000 and FastAPI on port 8000, then refresh this page. No metrics are fabricated in offline mode.</div></section>
      <p className="disclaimer">Security note: keep the local viewer and MLflow server bound to localhost unless authentication and network controls are configured.</p>
    </>;
  }

  return <>
    <div className="eyebrow">MISSION CONTROL / EXPERIMENT INTELLIGENCE</div>
    <h1>Autonomous ML.<br /><span style={{ color: '#f4df35' }}>Under your control.</span></h1>
    <p className="subtitle">A unified control plane for experiment tracking, knowledge intelligence and agentic model operations. Observe what your models do — and build toward systems that decide what happens next.</p>
    <div className="stats">
      <div className="stat"><strong>{experiments.length.toString().padStart(2, '0')}</strong><span>TRACKED EXPERIMENTS</span></div>
      <div className="stat"><strong>CONNECTED</strong><span>MLFLOW DATA SOURCE</span></div>
      <div className="stat"><strong>PHASE 08</strong><span>PLATFORM MILESTONE</span></div>
    </div>
    <section className="panel">
      <div className="section-heading"><h2>Experiment registry</h2><span>LIVE FROM LOCAL MLFLOW</span></div>
      {experiments.length ? experiments.map(e => <Link className="experiment" key={e.experiment_id} href={`/experiments/${e.experiment_id}`}><div><b>{e.name}</b><p>EXPERIMENT ID / {e.experiment_id}</p></div><span className="status ok">{e.lifecycle_stage}</span><span className="chevron">↗</span></Link>) : <div className="empty">No experiments recorded yet. Run a tracked training job to populate this registry.</div>}
    </section>
    <section className="panel">
      <div className="section-heading"><h2>Knowledge intelligence</h2><span>PHASE 07</span></div>
      <p className="subtitle" style={{ marginBottom: 16 }}>Explore ingested documents, grounded retrieval and generated training datasets.</p>
      <Link className="experiment" href="/knowledge"><div><b>Open Knowledge Hub</b><p>DOCUMENTS / SEARCH / DATASETS / GENERATION JOBS</p></div><span className="chevron">↗</span></Link>
    </section>
    <p className="disclaimer">Security note: keep local services bound to localhost unless authentication and network controls are configured.</p>
  </>;
}
