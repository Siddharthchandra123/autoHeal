"""Basic run comparison dashboard; pip install streamlit pandas."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title='AutoTune AI Mission Control',layout='wide')
st.title('AutoTune AI — Training Mission Control')
experiments = sorted([x for x in Path('runs').glob('*') if (x/'summary.json').exists()], reverse=True)
if not experiments:
    st.info('Run a CPU simulation first: python -m autotune.orchestrator --mode mock')
    st.stop()
selected = st.selectbox('Experiment',experiments,format_func=lambda x:x.name)
summary = json.loads((selected/'summary.json').read_text())
a,b,c = st.columns(3)
a.metric('Attempts',len(summary['attempts']))
b.metric('Best validation loss',f"{summary['best']['best_val_loss']:.4f}")
c.metric('Winning trial',summary['best']['trial'])
curves = []
for folder in sorted(selected.glob('trial-*')):
    path = folder/'metrics.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            obj=json.loads(line)
            curves.append({'trial':folder.name,**obj})
if curves:
    df=pd.DataFrame(curves)
    st.subheader('Validation loss per attempt')
    st.line_chart(df.pivot_table(index='step',columns='trial',values='val_loss'))
    st.subheader('Training loss per attempt')
    st.line_chart(df.pivot_table(index='step',columns='trial',values='train_loss'))
st.subheader('Detected conditions and decisions')
st.dataframe(pd.DataFrame(summary['attempts']), use_container_width=True)
st.subheader('Audit events')
events = [json.loads(line) for line in (selected/'events.jsonl').read_text().splitlines()]
st.dataframe(pd.DataFrame(events),use_container_width=True)
