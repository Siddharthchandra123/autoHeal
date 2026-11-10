import Link from 'next/link';
import type { Metadata } from 'next';
import StartupSequence from '../components/StartupSequence';
import './globals.css';

export const metadata: Metadata = {
  title: 'AutoTune AI | Autonomous MLOps Control Plane',
  description: 'Agentic MLOps mission control for experiment tracking, knowledge, tuning and self-healing.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>
    <StartupSequence />
    <div className="app">
      <aside className="sidebar">
        <Link href="/" className="logo"><span className="logomark">◈</span><span className="logoWords">AUTOTUNE <b>AI</b><small>AGENTIC MLOPS</small></span></Link>
        <div className="navCaption">CONTROL PLANE</div>
        <nav>
          <Link href="/" className="navItem"><span>⌂</span> Mission Control</Link>
          <Link href="/knowledge" className="navItem"><span>⌘</span> Knowledge Hub</Link>
          <div className="navCaption navSection">MODEL LIFECYCLE</div>
          <Link href="/" className="navItem"><span>◉</span> Experiments</Link>
          <span className="navItem navDisabled"><span>⌁</span> Hyperparameter Tuning <small>SOON</small></span>
          <span className="navItem navDisabled"><span>⟳</span> Self-Healing <small>SOON</small></span>
          <div className="navCaption navSection">INFRASTRUCTURE</div>
          <span className="navItem navDisabled"><span>▦</span> Model Registry <small>SOON</small></span>
          <span className="navItem navDisabled"><span>⌗</span> Observability <small>SOON</small></span>
          <span className="navItem navDisabled"><span>⌘</span> Cluster & Workers <small>SOON</small></span>
        </nav>
        <div className="sidebarBottom"><div className="systemBadge"><i /> LOCAL SYSTEM ONLINE</div><small>CPU-FIRST · PRIVATE WORKSPACE</small><small className="buildLabel">AUTOTUNE AI / PHASE 08</small></div>
      </aside>
      <main className="main">
        <header className="topbar"><div className="breadcrumb">AUTOTUNE AI <span>/</span> <b>MISSION CONTROL</b></div><div className="topbarRight"><span className="topStatus"><i /> API CONNECTIVITY</span><span className="topVersion">BUILD 0.14.0</span><span className="avatar">A</span></div></header>
        <div className="content">{children}</div>
      </main>
    </div>
  </body></html>;
}
