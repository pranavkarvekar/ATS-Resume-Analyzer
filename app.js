/* ─────────────────────────────────────────────────────────
   ATS Resume Analyzer — app.js
   Frontend logic: PDF extraction, API calls, UI control
───────────────────────────────────────────────────────── */

// ── Config ──────────────────────────────────────────────
// ⚠️ After deploying backend on Render, replace the URL below
const API_BASE =
  window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : 'https://ats-resume-analyzer-api.onrender.com'; // ← Update after Render deploy

// Configure PDF.js worker
if (typeof pdfjsLib !== 'undefined') {
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js';
}

// ── State ────────────────────────────────────────────────
const state = {
  resumeText: '',
  fileName: '',
  activeTab: 'review',
  isLoading: false,
  ragAvailable: false,
};

// ── DOM Refs ─────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// ── Init ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initFileUpload();
  initSidebarFeatureLinks();
  checkHealth();
});

// ── Health Check ─────────────────────────────────────────
async function checkHealth() {
  const ragEl = $('rag-value');
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error('Backend not responding');
    const data = await res.json();
    state.ragAvailable = data.rag_available;
    ragEl.textContent   = data.rag_available ? 'Active ✓' : 'Index not found';
    ragEl.className     = `rag-value ${data.rag_available ? 'active' : 'inactive'}`;
  } catch {
    ragEl.textContent = 'Backend offline';
    ragEl.className   = 'rag-value inactive';
  }
}

// ── Tab System ───────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
}

function switchTab(tabId) {
  // Update buttons
  document.querySelectorAll('.tab-btn').forEach((b) => {
    const active = b.dataset.tab === tabId;
    b.classList.toggle('active', active);
    b.setAttribute('aria-selected', active);
  });
  // Update panels
  document.querySelectorAll('.tab-panel').forEach((p) => {
    p.classList.toggle('active', p.id === `panel-${tabId}`);
  });
  // Update sidebar feature items
  document.querySelectorAll('.feature-item').forEach((f) => {
    f.classList.toggle('active', f.dataset.tab === tabId);
  });
  state.activeTab = tabId;
  // Hide results when switching tabs
  hideResults();
}

function initSidebarFeatureLinks() {
  document.querySelectorAll('.feature-item').forEach((f) => {
    f.addEventListener('click', () => switchTab(f.dataset.tab));
  });
}

// ── File Upload ───────────────────────────────────────────
function initFileUpload() {
  const zone  = $('upload-zone');
  const input = $('file-input');
  const removeBtn = $('remove-file');

  // Click to open file browser
  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });

  // Drag & Drop
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('dragging');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragging');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  });

  // Input change
  input.addEventListener('change', () => {
    if (input.files[0]) handleFileSelect(input.files[0]);
  });

  // Remove file
  removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
  });
}

async function handleFileSelect(file) {
  if (!file.name.endsWith('.pdf') && file.type !== 'application/pdf') {
    showToast('⚠️ Please upload a PDF file only', 'warning');
    return;
  }

  $('upload-zone').style.display    = 'none';
  $('upload-success').style.display = 'flex';
  $('file-name').textContent        = file.name;
  state.fileName = file.name;

  showToast('📄 Extracting text from PDF…', 'success');

  try {
    const text = await extractTextFromPDF(file);
    if (!text || text.trim().length < 50) {
      throw new Error('Could not extract enough text. Is this a scanned/image-based PDF?');
    }
    state.resumeText = text;
    $('empty-state').style.display = 'none';
    showToast(`✅ Resume loaded (${text.length.toLocaleString()} characters)`, 'success');
  } catch (err) {
    showToast(`❌ ${err.message}`, 'error');
    clearFile();
  }
}

function clearFile() {
  state.resumeText  = '';
  state.fileName    = '';
  $('file-input').value              = '';
  $('upload-zone').style.display     = 'flex';
  $('upload-success').style.display  = 'none';
  $('empty-state').style.display     = 'block';
  hideResults();
}

// ── PDF Text Extraction (PDF.js) ──────────────────────────
async function extractTextFromPDF(file) {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  let fullText = '';

  for (let i = 1; i <= pdf.numPages; i++) {
    const page    = await pdf.getPage(i);
    const content = await page.getTextContent();
    const strings = content.items.map((item) => item.str);
    fullText += strings.join(' ') + '\n';
  }
  return fullText.trim();
}

// ── Analyze Resume (LLM Modes) ────────────────────────────
async function analyzeResume(mode) {
  if (!state.resumeText) {
    showToast('⚠️ Please upload a PDF resume first', 'warning');
    return;
  }

  const jobDesc = ($('job-desc').value || '').trim();
  const requiresJD = ['review', 'optimize', 'score', 'fit'].includes(mode);
  if (requiresJD && !jobDesc) {
    showToast('⚠️ Please paste a job description first', 'warning');
    return;
  }

  const language = $('lang-select') ? $('lang-select').value : 'French';

  // Disable button
  const btn = $(`btn-${mode}`);
  if (btn) btn.disabled = true;

  showLoading(getLoadingText(mode));

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode,
        resume_text: state.resumeText,
        job_desc:    jobDesc,
        language,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || 'Analysis failed');
    }

    const data = await res.json();
    showTextResult(data.result, mode === 'score' && data.rag_used);
  } catch (err) {
    showError(err.message);
  } finally {
    hideLoading();
    if (btn) btn.disabled = false;
  }
}

// ── Analyze Keywords ──────────────────────────────────────
async function analyzeKeywords() {
  if (!state.resumeText) {
    showToast('⚠️ Please upload a PDF resume first', 'warning');
    return;
  }

  const jobDesc = ($('job-desc').value || '').trim();
  if (!jobDesc) {
    showToast('⚠️ Please paste a job description first', 'warning');
    return;
  }

  const btn = $('btn-keywords');
  if (btn) btn.disabled = true;

  showLoading('Analyzing keyword coverage…');

  try {
    const res = await fetch(`${API_BASE}/api/keywords`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_desc: jobDesc, resume_text: state.resumeText }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || 'Keyword analysis failed');
    }

    const data = await res.json();
    showKeywordsResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    hideLoading();
    if (btn) btn.disabled = false;
  }
}

// ── Result Renderers ──────────────────────────────────────
function showTextResult(text, ragUsed = false) {
  const panel = $('results-panel');
  const box   = $('result-box');
  const badge = $('rag-badge');
  const kwRes = $('keywords-result');

  kwRes.style.display = 'none';
  badge.style.display = ragUsed ? 'block' : 'none';
  box.textContent     = text;
  box.style.display   = 'block';
  panel.style.display = 'block';

  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showKeywordsResult(data) {
  const panel   = $('results-panel');
  const box     = $('result-box');
  const kwRes   = $('keywords-result');
  const metrics = $('metrics-row');
  const badge   = $('rag-badge');

  box.style.display   = 'none';
  badge.style.display = 'none';

  // Metrics cards
  const pctColor = data.coverage_pct >= 60 ? '#4ade80' : data.coverage_pct >= 40 ? '#fbbf24' : '#f87171';
  metrics.innerHTML = `
    <div class="metric-card">
      <div class="metric-value" style="color:#4ade80;">${data.matched.length}</div>
      <div class="metric-label">Matched</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="color:#f87171;">${data.missing.length}</div>
      <div class="metric-label">Missing</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="color:${pctColor};">${data.coverage_pct}%</div>
      <div class="metric-label">Coverage</div>
    </div>`;

  // Keyword chips
  $('matched-chips').innerHTML = data.matched
    .map((kw, i) => `<span class="chip chip-matched" style="animation-delay:${i * 15}ms;">${kw}</span>`)
    .join('');
  $('missing-chips').innerHTML = data.missing
    .map((kw, i) => `<span class="chip chip-missing" style="animation-delay:${i * 15}ms;">${kw}</span>`)
    .join('');

  kwRes.style.display   = 'block';
  panel.style.display   = 'block';
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showError(message) {
  hideLoading();
  const panel = $('results-panel');
  const box   = $('result-box');
  box.textContent   = `❌ Error: ${message}`;
  box.style.display = 'block';
  panel.style.display = 'block';
  showToast(`❌ ${message}`, 'error');
}

// ── Loading States ────────────────────────────────────────
function showLoading(text = 'Analyzing your resume…') {
  state.isLoading = true;
  $('results-panel').style.display  = 'block';
  $('loading-state').style.display  = 'flex';
  $('result-box').style.display     = 'none';
  $('keywords-result').style.display = 'none';
  $('rag-badge').style.display      = 'none';
  $('loading-text').textContent     = text;
}

function hideLoading() {
  state.isLoading = false;
  $('loading-state').style.display = 'none';
}

function hideResults() {
  $('results-panel').style.display  = 'none';
  $('result-box').style.display     = 'none';
  $('keywords-result').style.display = 'none';
  $('rag-badge').style.display      = 'none';
  $('loading-state').style.display  = 'none';
}

// ── Loading Text per Mode ─────────────────────────────────
function getLoadingText(mode) {
  const map = {
    review:    'Analyzing your resume from an HR perspective…',
    optimize:  'Crafting improvement strategies…',
    score:     'Scoring your resume with RAG pipeline…',
    fit:       'Evaluating job alignment…',
    design:    'Evaluating design and structure…',
    translate: 'Translating your resume…',
  };
  return map[mode] || 'Analyzing your resume…';
}

// ── Toast Notification ────────────────────────────────────
let toastTimer;
function showToast(message, type = 'success') {
  const toast = $('toast');
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.className   = `toast show ${type}`;
  toastTimer = setTimeout(() => { toast.className = 'toast'; }, 3500);
}
