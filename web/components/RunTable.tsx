import Link from 'next/link';
import type { Run } from '../lib/api';

export default function RunTable({ runs }: { runs: Run[] }) {
  if (!runs.length) return <div className="empty">No runs found in this experiment.</div>;
  return <div className="tableWrap"><table><thead><tr>
    <th>Run</th><th>Type</th><th>Status</th><th>Diagnosis</th><th>Best val. loss</th><th>Fault injected</th>
  </tr></thead><tbody>{runs.map(run=><tr key={run.run_id}>
    <td><Link href={`/runs/${run.run_id}`}>{run.tags['mlflow.runName'] || run.run_id.slice(0,12)}</Link><small>{run.run_id.slice(0,12)}</small></td>
    <td>{run.tags['autotune.role'] || 'Uncategorized'}</td>
    <td><span className={`status ${run.status==='FINISHED'?'ok':''}`}>{run.status}</span></td>
    <td>{run.tags['autotune.diagnosis']||'—'}</td>
    <td className="tabular">{run.metrics.best_val_loss?.toFixed(6) ?? '—'}</td>
    <td>{run.tags['autotune.fault_injected'] || run.tags['autotune.fault_injection'] || 'false'}</td>
  </tr>)}</tbody></table></div>;
}
