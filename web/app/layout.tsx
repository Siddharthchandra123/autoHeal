import Link from 'next/link';
import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = { title: 'AutoTune AI | Experiment Viewer', description: 'Local read-only MLOps experiments' };
export default function RootLayout({children}: Readonly<{children: React.ReactNode}>) {
  return <html lang="en"><body><div className="app">
    <aside className="sidebar"><div className="logo"><span className="logomark">◈</span><div>AutoTune <span>AI</span><small>MISSION CONTROL</small></div></div>
      <nav><Link href="/">◫ &nbsp; Overview</Link><div className="navMuted">EXPERIMENT TRACKING</div>
      <div className="navItem">◉ &nbsp; MLflow Viewer</div></nav>
      <div className="sidebarFoot">LOCAL MODE <span className="liveDot" /> <small>Read-only · CPU-first</small></div>
    </aside><main className="main"><header className="topbar"><span>Training Intelligence / Experiment Analytics</span><span className="pill">READ ONLY</span></header>
    <div className="content">{children}</div></main></div></body></html>;
}
