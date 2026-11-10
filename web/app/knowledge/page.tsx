'use client';

import { useCallback, useEffect, useState } from 'react';

type DocumentRow = {id:string; filename:string; file_type:string; bytes:number; chars:number; processed_chars:number; chunk_count:number; status:string; error?:string|null; uploaded_at:string};
type SearchRow = {chunk_id:string; source_file:string; chunk_index:number; char_start?:number; char_end?:number; text:string; score:number; method:string};
type DatasetRow = {path:string; counts:Record<string,number>; source_disjoint:boolean; source_unit_kind?:'document'|'record'; created_at_utc:string};
type GenerationJob = {job_id:string; status:string; stage?:string; progress?:Record<string,any>; result?:any; error?:{type:string;message:string}};
type Job = {job_id:string; mode:string; status:string; result?:any; error?:{type:string;message:string}};
const API = process.env.NEXT_PUBLIC_KNOWLEDGE_API || 'http://127.0.0.1:8010';
const MAX_UPLOAD_BYTES = 512 * 1024 * 1024;

export default function KnowledgeEnginePage() {
  const [documents,setDocuments] = useState<DocumentRow[]>([]);
  const [datasets,setDatasets] = useState<DatasetRow[]>([]);
  const [stats,setStats] = useState<any|null>(null);
  const [query,setQuery] = useState('');
  const [results,setResults] = useState<SearchRow[]>([]);
  const [groundedAnswer,setGroundedAnswer] = useState<any|null>(null);
  const [file,setFile] = useState<File|null>(null);
  const [ollamaModel,setOllamaModel] = useState('qwen2.5:0.5b');
  const [job,setJob] = useState<Job|null>(null);
  const [generationJob,setGenerationJob] = useState<GenerationJob|null>(null);
  const [preview,setPreview] = useState<any|null>(null);
  const [busy,setBusy] = useState('');
  const [notice,setNotice] = useState('');
  const [error,setError] = useState('');
  const [searchMode,setSearchMode] = useState('auto');

  const refresh = useCallback(async () => {
    const [docs,dsets,summary] = await Promise.all([
      fetch(`${API}/api/knowledge/documents`).then(r=>r.json()),
      fetch(`${API}/api/knowledge/datasets`).then(r=>r.json()),
      fetch(`${API}/api/knowledge/stats`).then(r=>r.ok?r.json():null)
    ]);
    setDocuments(docs.documents || []); setDatasets(dsets.datasets || []); setStats(summary);
  },[]);
  useEffect(()=>{ refresh().catch(e=>setError(String(e))); },[refresh]);
  const hasPendingIngestion = documents.some(d=>['queued','indexing'].includes(d.status));
  useEffect(()=>{
    if (!hasPendingIngestion) return;
    const timer = window.setInterval(()=>{ refresh().catch(e=>setError(String(e))); },2000);
    return ()=>window.clearInterval(timer);
  },[hasPendingIngestion,refresh]);
  useEffect(()=>{
    if (!job || !['queued','running'].includes(job.status)) return;
    const timer = window.setInterval(async()=>{
      try { const r=await fetch(`${API}/api/knowledge/training/jobs/${job.job_id}`); const j=await r.json(); setJob(j); if(!['queued','running'].includes(j.status)){window.clearInterval(timer);setNotice(`Job ${j.status}.`);} }
      catch(e){setError(String(e));}
    },3000);
    return ()=>window.clearInterval(timer);
  },[job?.job_id,job?.status]);
  useEffect(()=>{
    if (!generationJob || !['queued','running'].includes(generationJob.status)) return;
    let stopped = false;
    const poll = async () => {
      try {
        const response = await fetch(`${API}/api/knowledge/datasets/generation-jobs/${generationJob.job_id}`);
        const body = await response.json().catch(()=>({detail:'Invalid job status response'}));
        if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
        if (stopped) return;
        setGenerationJob(body);
        if (body.status === 'completed') {
          setNotice(`Dataset ready: ${JSON.stringify(body.result?.counts || {})}. Split unit: ${body.result?.source_unit_kind || 'document'}.`);
          setError('');
          refresh().catch(e=>setError(String(e)));
        } else if (body.status === 'failed' || body.status === 'interrupted') {
          setError(`${body.error?.type || body.status}: ${body.error?.message || 'Dataset generation did not complete.'}`);
          setNotice('');
        }
      } catch(e) {
        if (!stopped) setError(String(e));
      }
    };
    const timer = window.setInterval(poll, 1500);
    poll();
    return ()=>{stopped=true;window.clearInterval(timer);};
  },[generationJob?.job_id,generationJob?.status,refresh]);

  async function request(path:string, init?:RequestInit) {
    const response=await fetch(`${API}${path}`,init);
    const body=await response.json().catch(()=>({detail:'Invalid API response'}));
    if(!response.ok) throw new Error(body.detail || body.error || `HTTP ${response.status}`);
    return body;
  }
  async function upload() {
    if(!file) return;
    const uploadLimit = Number(stats?.limits?.max_file_bytes ?? MAX_UPLOAD_BYTES);
    if(file.size > uploadLimit){setError(`File exceeds the ${(uploadLimit/1024/1024).toFixed(0)} MB upload limit.`);return;}
    setBusy('upload');setError('');setNotice('');
    try { const form=new FormData();form.append('file',file);const r=await request('/api/knowledge/documents',{method:'POST',body:form});
      setNotice(r.duplicate?`${r.filename} is already in the knowledge base.`:`Accepted ${r.filename}. Extraction and indexing are running in the background.`);setFile(null);await refresh(); }
    catch(e){setError(String(e));} finally{setBusy('');}
  }
  async function retryDocument(id:string) {
    setBusy(`retry-${id}`);setError('');
    try {await request(`/api/knowledge/documents/${id}/retry`,{method:'POST'});setNotice('Re-index job queued.');await refresh();}
    catch(e){setError(String(e));} finally{setBusy('');}
  }
  async function search() {
    if(!query.trim()) return;setBusy('search');setError('');
    try {const r=await request('/api/knowledge/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,top_k:8,mode:searchMode})});setResults(r.results||[]);setNotice(`Search mode: ${r.mode}. ${r.results?.length||0} matches.`);}
    catch(e){setError(String(e));} finally{setBusy('');}
  }
  async function askGrounded() {
    if(!query.trim()) return;setBusy('ask');setError('');setGroundedAnswer(null);
    try {const r=await request('/api/knowledge/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:query,model:ollamaModel,top_k:6,mode:searchMode})});setGroundedAnswer(r);setNotice(r.insufficient_evidence?'The engine could not verify a grounded answer.': 'Answer generated with verifiable source quotes.');}
    catch(e){setError(String(e));} finally{setBusy('');}
  }
  async function indexEmbeddings() {
    setBusy('index');setError('');
    try {const r=await request('/api/knowledge/index',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({batch_size:16})});setNotice(`Semantic index complete: ${r.indexed} new embeddings using ${r.model}.`);}
    catch(e){setError(String(e));} finally{setBusy('');}
  }
  async function generateDataset() {
    setBusy('dataset');setError('');setNotice('Starting dataset build. The job will continue in the background and report progress here.');setGenerationJob(null);
    try {
      const r=await request('/api/knowledge/datasets/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:ollamaModel,max_chunks:100,min_pairs:200,seed:42})});
      setGenerationJob(r);
      setNotice(`Dataset generation job ${r.job_id} ${r.status}.`);
    } catch(e){setError(String(e));setNotice('');} finally{setBusy('');}
  }
  async function previewDataset(dataset:DatasetRow, split:'train'|'validation'|'test') {
    setBusy('preview');setError('');
    try {const datasetId=dataset.path.split(/[\\/]/).pop();const r=await request(`/api/knowledge/datasets/${datasetId}/preview?split=${split}&limit=8`);setPreview(r);}
    catch(e){setError(String(e));} finally{setBusy('');}
  }
  async function runJob(dataset:DatasetRow,mode:'preflight'|'execute') {
    if(mode==='execute' && !window.confirm('This will download model weights and fine-tune up to three Hugging Face models sequentially on CPU. It may use substantial RAM, CPU and disk. Continue?')) return;
    setBusy(mode);setError('');setNotice('');
    try {const datasetId=dataset.path.split(/[\\/]/).pop();const j=await request('/api/knowledge/training/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_id:datasetId,mode,allow_cpu:mode==='execute',allow_small_data:false,max_candidates:3})});setJob(j);setNotice(`Training job ${j.job_id} ${j.status}.`);}
    catch(e){setError(String(e));} finally{setBusy('');}
  }

  return <>
    <div className="eyebrow">KNOWLEDGE INTELLIGENCE / PHASE 7</div>
    <h1>Knowledge Engine</h1>
    <p className="subtitle">Upload your domain knowledge, retrieve grounded context, create a leakage-safe SFT dataset, and launch sequential model trials.</p>
    <div className="knowledgeStats">
      <div className="stat"><strong>{stats?.documents ?? documents.length}</strong><span>DOCUMENTS</span></div>
      <div className="stat"><strong>{(stats?.chunks ?? 0).toLocaleString()}</strong><span>INDEXED CHUNKS</span></div>
      <div className="stat"><strong>{((stats?.extracted_characters ?? 0)/1_000_000).toFixed(2)}M</strong><span>EXTRACTED CHARACTERS</span></div>
      <div className="stat"><strong>{datasets.length}</strong><span>DATASETS BUILT</span></div>
    </div>
    {notice&&<div className="notice">{notice}</div>}{error&&<div className="errorBox">{error}</div>}

    <section className="panel knowledgePanel">
      <div className="section-heading"><h2>1. Upload knowledge</h2><span>PDF · DOCX · TXT · MD · CSV · JSON · JSONL · {(Number(stats?.limits?.max_file_bytes ?? MAX_UPLOAD_BYTES)/1024/1024).toFixed(0)} MB/file · {(Number(stats?.limits?.max_extracted_characters ?? 100_000_000)/1_000_000).toFixed(0)}M chars</span></div>
      <div className="uploadRow"><input type="file" accept=".pdf,.docx,.txt,.md,.csv,.json,.jsonl" onChange={e=>setFile(e.target.files?.[0]||null)}/><button disabled={!file||!!busy} onClick={upload}>{busy==='upload'?'Uploading…':'Upload & index'}</button></div>
      <p className="muted small">Uploads are streamed to disk; extraction and SQLite/FTS indexing run in bounded batches. Limits: 512 MB per file and 100 million extracted characters per document. Duplicate uploads are detected by SHA-256. Scanned PDFs need OCR before indexing.</p>
      {file&&<p className="muted small">Selected: {file.name} · {(file.size/1024/1024).toFixed(1)} MB</p>}
      <div className="docList">{documents.slice(0,12).map(d=><div className="docRow" key={d.id}><div className="docDetails"><b>{d.filename}</b><small>{d.file_type} · {(d.bytes/1024/1024).toFixed(1)} MB · {d.chars.toLocaleString()} extracted chars · {(d.chunk_count||0).toLocaleString()} chunks</small>{['queued','indexing'].includes(d.status)&&<small>Processed {Number(d.processed_chars||0).toLocaleString()} characters…</small>}{d.error&&<small className="docError">{d.error}</small>}</div><div className="docStatus"><span className={`status ${d.status==='indexed'?'ok':d.status==='failed'?'bad':''}`}>{d.status}</span>{d.status==='failed'&&<button className="secondary retryButton" disabled={!!busy} onClick={()=>retryDocument(d.id)}>Retry</button>}</div></div>)}{!documents.length&&<div className="empty">No documents uploaded yet.</div>}</div>
    </section>

    <section className="panel knowledgePanel">
      <div className="section-heading"><h2>2. Search your knowledge</h2><span>RAG retrieval</span></div>
      <div className="searchRow"><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Ask a question about your uploaded material…" onKeyDown={e=>e.key==='Enter'&&search()}/><select value={searchMode} onChange={e=>setSearchMode(e.target.value)}><option value="auto">Auto fallback</option><option value="lexical">Keyword search</option><option value="hybrid">Hybrid search</option><option value="semantic">Semantic only</option></select><button className="secondary" disabled={!query.trim()||!!busy} onClick={search}>{busy==='search'?'Searching…':'Search snippets'}</button><button disabled={!query.trim()||!ollamaModel.trim()||!!busy} onClick={askGrounded}>{busy==='ask'?'Answering…':'Ask grounded AI'}</button></div>
      <div className="rowActions"><button className="secondary" disabled={!!busy||!documents.length} onClick={indexEmbeddings}>{busy==='index'?'Indexing…':'Build semantic index (CPU)'}</button><span className="muted small">Semantic mode requires the optional sentence-transformers package and an embedding model download. Auto mode falls back to FTS5.</span></div>
      {groundedAnswer&&<div className="answerCard"><div className="resultMeta"><b>Grounded answer</b><span>{groundedAnswer.insufficient_evidence?'Insufficient evidence':'Verified quote citations'}</span></div><p>{groundedAnswer.answer}</p>{(groundedAnswer.citations||[]).map((c:any)=><blockquote key={c.chunk_id}><b>{c.source_file}</b><br/>Chunk {c.chunk_index??'-'} · characters {c.char_start??0}-{c.char_end??0}<br/>{c.quote}</blockquote>)}</div>}
      <div className="searchResults">{results.map(r=><article className="resultCard" key={r.chunk_id}><div className="resultMeta"><b>{r.source_file}</b><span>{r.method} · score {Number(r.score).toFixed(3)} · chars {r.char_start??0}-{r.char_end??0}</span></div><p>{r.text}</p></article>)}{!results.length&&<div className="empty">Search results will appear here with their source file.</div>}</div>
    </section>

    <section className="panel knowledgePanel">
      <div className="section-heading"><h2>3. Build a fine-tuning dataset</h2><span>Direct SFT import or grounded generation</span></div>
      <div className="searchRow"><input value={ollamaModel} onChange={e=>setOllamaModel(e.target.value)} placeholder="Local Ollama model tag"/><button disabled={!documents.length||!ollamaModel.trim()||!!busy||!!(generationJob&&['queued','running'].includes(generationJob.status))} onClick={generateDataset}>{busy==='dataset'||(generationJob&&['queued','running'].includes(generationJob.status))?'Generating…':'Build fine-tuning dataset'}</button></div>
      <p className="muted small">Structured instruction/response JSONL (including Dolly) is imported directly without Ollama. Other uploaded documents use local Ollama to generate answers with exact source quotes. Splits are disjoint by source record for imported datasets or by source document for generated Q&A.</p>
      <div className="datasetList">{datasets.map((d,i)=><div className="datasetRow" key={d.path}><div><b>{d.path.split(/[\\/]/).pop()}</b><small>Train {d.counts?.train??0} · Validation {d.counts?.validation??0} · Test {d.counts?.test??0} · {d.source_unit_kind==='record'?'source-record':'source-document'} disjoint {String(d.source_disjoint)}</small></div><div className="datasetButtons"><button className="secondary" disabled={!!busy} onClick={()=>previewDataset(d,'train')}>Preview train</button><button className="secondary" disabled={!!busy} onClick={()=>previewDataset(d,'validation')}>Preview validation</button><button className="secondary" disabled={!!busy} onClick={()=>previewDataset(d,'test')}>Preview test</button><button className="secondary" disabled={!!busy} onClick={()=>runJob(d,'preflight')}>Preflight</button><button disabled={!!busy||!d.source_disjoint} onClick={()=>runJob(d,'execute')}>Fine-tune models</button></div></div>)}{!datasets.length&&!(generationJob&&['queued','running'].includes(generationJob.status))&&<div className="empty">No datasets yet. Upload a supervised JSONL dataset such as Dolly for direct import, or upload at least four distinct knowledge documents for Ollama-grounded Q&A generation.</div>}</div>
      {generationJob&&<div className="answerCard generationJobCard"><div className="resultMeta"><b>Dataset build · {generationJob.stage||generationJob.status}</b><span className={`status ${generationJob.status==='completed'?'ok':generationJob.status==='failed'?'bad':''}`}>{generationJob.status}</span></div>
        <div className="generationProgress"><div className="progressTrack"><div className={`progressFill ${generationJob.progress?.total_chunks||generationJob.progress?.total_records?'':'indeterminate'}`} style={{width:(generationJob.progress?.total_chunks||generationJob.progress?.total_records)?`${Math.min(100,Math.round(100*(generationJob.progress?.processed_chunks??generationJob.progress?.processed_records??0)/(generationJob.progress?.total_chunks??generationJob.progress?.total_records??1)))}%`:'35%'}}/></div>
        <div className="progressStats"><span>Chunks {generationJob.progress?.processed_chunks??0}{generationJob.progress?.total_chunks?` / ${generationJob.progress.total_chunks}`:''}</span><span>Records {Number(generationJob.progress?.processed_records??0).toLocaleString()}</span><span>Accepted pairs {Number(generationJob.progress?.accepted_pairs??0).toLocaleString()}</span><span>Rejected {Number(generationJob.progress?.rejected_pairs??0).toLocaleString()}</span></div></div>
        {generationJob.progress?.current_source&&<p className="muted small">Current source: {generationJob.progress.current_source}</p>}
        {generationJob.error&&<div className="errorBox">{generationJob.error.type}: {generationJob.error.message}</div>}
        {generationJob.result&&<p className="muted small">Created {generationJob.result.output_dir} · mode {generationJob.result.generation_mode||'grounded generation'} · source unit {generationJob.result.source_unit_kind||'document'}</p>}
      </div>}
      {preview&&<div className="answerCard"><div className="resultMeta"><b>Dataset preview · {preview.split}</b><span>{preview.examples?.length||0} examples · {preview.source_unit_kind||'document'}-disjoint {String(preview.source_disjoint)}</span></div><pre className="jsonResult">{JSON.stringify(preview.examples,null,2)}</pre></div>}
    </section>

    {job&&<section className="panel knowledgePanel"><div className="section-heading"><h2>4. Orchestration job</h2><span className={`status ${job.status==='completed'?'ok':job.status==='failed'?'bad':''}`}>{job.status}</span></div>
      <p className="muted small">Job ID {job.job_id} · mode {job.mode}</p>
      {job.error&&<div className="errorBox">{job.error.type}: {job.error.message}</div>}
      {job.result&&<pre className="jsonResult">{JSON.stringify(job.result,null,2)}</pre>}
    </section>}
    <p className="disclaimer">Model training is an explicit user action. No candidate is automatically promoted. Held-out token-F1 is an initial objective filter, not a safety or production certification.</p>
  </>;
}
