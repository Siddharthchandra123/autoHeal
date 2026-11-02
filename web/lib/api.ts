export type Experiment = { experiment_id: string; name: string; lifecycle_stage: string };
export type Run = {
  run_id: string;
  experiment_id: string;
  status: string;
  start_time: number | null;
  end_time: number | null;
  metrics: Record<string, number>;
  params: Record<string, string>;
  tags: Record<string, string>;
};
export type Point = { step: number; value: number };

function baseUrl(): string {
  const base = process.env.AUTOTUNE_VIEWER_API || 'http://127.0.0.1:8000';
  if (!/^http:\/\/127\.0\.0\.1:\d{2,5}$/.test(base)) {
    throw new Error('Viewer API must use numeric 127.0.0.1 loopback');
  }
  return base;
}

export async function readApi<T>(path: string): Promise<T> {
  if (!path.startsWith('/api/') || path.includes('..') || path.includes('?')) {
    throw new Error('Invalid API path');
  }
  const res = await fetch(baseUrl() + path, { cache: 'no-store', signal: AbortSignal.timeout(5000) });
  if (!res.ok) throw new Error(`Local API returned ${res.status}`);
  return res.json() as Promise<T>;
}

export async function getExperiments() {
  return (await readApi<{ experiments: Experiment[] }>('/api/experiments')).experiments;
}
export async function getRuns(id: string) {
  if (!/^\d{1,12}$/.test(id)) throw new Error('Invalid experiment ID');
  return (await readApi<{ runs: Run[] }>(`/api/experiments/${id}/runs`)).runs;
}
export async function getRun(id: string) {
  if (!/^[a-f0-9]{32}$/i.test(id)) throw new Error('Invalid run ID');
  return readApi<Run>(`/api/runs/${id}`);
}
export async function getHistory(id: string, metric: string) {
  if (!/^[a-f0-9]{32}$/i.test(id) || !['train_loss','val_loss','eval_loss','loss'].includes(metric)) {
    throw new Error('Invalid metric request');
  }
  return (await readApi<{ points: Point[] }>(`/api/runs/${id}/metrics/${metric}`)).points;
}
