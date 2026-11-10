'use client';

import { useEffect, useState } from 'react';

const agents = [
  { name: 'OPTIMIZER', color: '#f4df35', path: 'M 8 20 C 24 6, 40 7, 52 25 S 75 46, 93 25' },
  { name: 'MONITOR', color: '#67d8ff', path: 'M 12 80 C 30 58, 44 56, 58 73 S 77 96, 94 75' },
  { name: 'RECOVERY', color: '#ff9b72', path: 'M 10 135 C 30 110, 44 112, 58 128 S 77 150, 94 130' },
  { name: 'KNOWLEDGE', color: '#c39aff', path: 'M 160 18 C 140 38, 140 54, 160 72 S 180 100, 160 122' },
  { name: 'DEPLOYMENT', color: '#9be7a4', path: 'M 300 20 C 276 36, 278 56, 260 74 S 242 106, 260 132' },
];

export default function StartupSequence() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    try {
      if (sessionStorage.getItem('autotune-intro-seen') === '1') return;
      setVisible(true);
      const timer = window.setTimeout(() => {
        setVisible(false);
        sessionStorage.setItem('autotune-intro-seen', '1');
      }, 3900);
      return () => window.clearTimeout(timer);
    } catch {
      setVisible(false);
    }
  }, []);

  if (!visible) return null;

  return (
    <div className="startupOverlay" role="status" aria-label="Initializing AutoTune AI">
      <div className="startupGrid" />
      <div className="startupTop"><span className="brandGlyph">◈</span><span>AUTOTUNE <b>AI</b></span><span className="startupVersion">MISSION CONTROL / 0.14</span></div>
      <div className="startupCenter">
        <div className="startupEyebrow"><i /> AUTONOMOUS SYSTEM INITIALIZATION</div>
        <h1>INTELLIGENCE IN<br /><em>MOTION.</em></h1>
        <p>Five agents. One control plane. Continuous optimization.</p>
        <div className="agentStage">
          <svg viewBox="0 0 320 160" role="img" aria-label="Five animated agent pointers coordinating">
            <defs><radialGradient id="coreGlow"><stop offset="0" stopColor="#f4df35" stopOpacity=".35"/><stop offset="1" stopColor="#f4df35" stopOpacity="0"/></radialGradient></defs>
            <circle cx="160" cy="80" r="48" fill="url(#coreGlow)" />
            <circle cx="160" cy="80" r="22" fill="none" stroke="#f4df35" strokeOpacity=".45" strokeDasharray="2 5" />
            <circle cx="160" cy="80" r="5" fill="#f4df35" />
            {agents.map((agent, index) => <g key={agent.name}>
              <path d={agent.path} fill="none" stroke={agent.color} strokeOpacity=".22" strokeWidth="1" strokeDasharray="3 5" />
              <g>
                <animateMotion dur={`${2.8 + index * 0.23}s`} begin={`${index * -0.42}s`} repeatCount="indefinite" path={agent.path} />
                <path d="M0 0 L0 15 L4.5 10.5 L8.5 19 L12 17.5 L8 9 L15 9 Z" fill={agent.color} stroke="#080909" strokeWidth="1" />
              </g>
            </g>)}
          </svg>
          <div className="agentLegend">{agents.map(a => <span key={a.name}><i style={{ background: a.color }} />{a.name}</span>)}</div>
        </div>
        <div className="startupProgress"><span /></div>
        <div className="startupStatus"><span>SYNCING AGENT NETWORK</span><span>PLEASE WAIT</span></div>
      </div>
      <button className="skipIntro" onClick={() => { setVisible(false); try { sessionStorage.setItem('autotune-intro-seen', '1'); } catch {} }}>SKIP INTRO ↗</button>
      <div className="startupFooter"><span>SELF-HEALING / HYPERPARAMETER OPTIMIZATION / MODEL INTELLIGENCE</span><span>SYS. READY SOON</span></div>
    </div>
  );
}
