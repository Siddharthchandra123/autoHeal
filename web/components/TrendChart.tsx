import type { Point } from '../lib/api';

export default function TrendChart({ title, points }: { title: string; points: Point[] }) {
  if (!points.length) return <div className="empty">No {title.toLowerCase()} history logged for this run.</div>;
  const w=700, h=180, pad=20;
  const minX=Math.min(...points.map(p=>p.step));
  const maxX=Math.max(...points.map(p=>p.step));
  const minY=Math.min(...points.map(p=>p.value));
  const maxY=Math.max(...points.map(p=>p.value));
  const X=(x:number)=>pad+(x-minX)/(maxX-minX||1)*(w-2*pad);
  const Y=(y:number)=>h-pad-(y-minY)/(maxY-minY||1)*(h-2*pad);
  const polyline=points.map(p=>`${X(p.step)},${Y(p.value)}`).join(' ');
  return <section className="panel"><div className="section-heading"><h3>{title}</h3><span>{points.length} points</span></div>
    <svg className="trend" role="img" aria-label={`${title} training metric curve`} viewBox={`0 0 ${w} ${h}`}>
      <line x1={pad} y1={h-pad} x2={w-pad} y2={h-pad} stroke="#384353" />
      <polyline points={polyline} stroke="#65d1bd" fill="none" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      {points.length===1 && <circle cx={X(points[0].step)} cy={Y(points[0].value)} r="4" fill="#65d1bd" />}
      <text x={pad} y={14} fill="#a3afc5" fontSize="11">{maxY.toFixed(4)}</text>
      <text x={pad} y={h-3} fill="#a3afc5" fontSize="11">step {minX}</text>
      <text x={w-pad} y={h-3} textAnchor="end" fill="#a3afc5" fontSize="11">step {maxX}</text>
    </svg>
  </section>;
}
