// --- Elements ---
const textInput = document.getElementById('text-input');
const micBtn = document.getElementById('mic-btn');
const clearBtn = document.getElementById('clear-btn');
const processBtn = document.getElementById('process-btn');
const resultsContainer = document.getElementById('results-container');
const metricsContainer = document.getElementById('metrics-container');
const targetLangSelect = document.getElementById('target-language-select');
const tiltCard = document.getElementById('tilt-card');

const API_URL = 'http://127.0.0.1:5000';


// Init
document.addEventListener('DOMContentLoaded', async () => {
  await loadLanguages();
  setup3DTilt();
  clearBtn.classList.add('hidden');
});

textInput.addEventListener('input', () => {
  clearBtn.classList.toggle('hidden', !textInput.value);
});
clearBtn.addEventListener('click', () => {
  textInput.value = '';
  clearBtn.classList.add('hidden');
  textInput.focus();
});
processBtn.addEventListener('click', handleProcess);

// --- API ---
async function loadLanguages() {
  // Hard-coded since only 3 focus languages
  const langs = {
    en: "🌐 English",
    hi: "🇮🇳 Hindi (हिंदी)",
    te: "🇮🇳 Telugu (తెలుగు)"
  };
  targetLangSelect.innerHTML = `<option value="" disabled selected>Choose language...</option>` +
    Object.entries(langs)
      .map(([code, name]) => `<option value="${code}">${name}</option>`)
      .join('');
}

async function handleProcess() {
  const text = textInput.value.trim();
  if (!text) {
    textInput.focus();
    return;
  }

  resultsContainer.innerHTML = `
    <div class="flex justify-center p-8">
      <div class="loader"></div>
      <p class="ml-4 text-slate-600">Analyzing pipeline...</p>
    </div>`;
  metricsContainer.innerHTML = `<p class="text-slate-500">Calculating...</p>`;

  try {
    const payload = { text, target_lang: targetLangSelect.value };
    const res = await fetch(`${API_URL}/api/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderResults(data);
    renderMetrics(data.metrics);
  } catch (e) {
    console.error(e);
    resultsContainer.innerHTML = `<div class="bg-red-50 border border-red-200 p-4 rounded-lg"><p class="text-red-700">❌ Backend error. Is Flask running?</p></div>`;
    metricsContainer.innerHTML = `<p class="text-red-500">Failed</p>`;
  }
}

// --- Render ---
function renderResults(d) {
  const norm = d.normalized_text ? `
    <div class="mt-3 p-3 bg-sky-50 border-l-4 border-sky-300 rounded-r-lg animate-fadeIn">
      <h4 class="font-semibold text-slate-600">Normalized Input</h4>
      <p class="mt-1 text-slate-800">${d.normalized_text}</p>
    </div>` : '';

  const ragList = (d.retrieved_items || [])
    .map((r, i) => `
      <div class="p-3 rounded-md border bg-amber-50 mb-2 result-item animate-slideUp">
        <div class="text-xs text-amber-700 mb-1">#${i+1} • ${r.domain} • ${r.lang} • score ${r.score.toFixed(3)}</div>
        <div class="text-slate-800">${r.text}</div>
      </div>
    `).join('') || `<div class="text-slate-500">No relevant items found.</div>`;

  resultsContainer.innerHTML = `
    <div class="bg-gray-50 rounded-lg p-6 shadow-sm space-y-4 result-card animate-fadeIn">
      <div>
        <h3 class="font-semibold text-slate-500">Detected Language</h3>
        <span class="inline-block bg-indigo-100 text-indigo-700 text-sm font-bold px-3 py-1 rounded-full">${d.detected_language_name}</span>
        <p class="mt-2 text-lg text-slate-900">${d.original_text}</p>
      </div>
      ${norm}
      <div>
        <h3 class="font-semibold text-slate-500">Retrieved Context (Top-K)</h3>
        ${ragList}
      </div>
      <div>
        <h3 class="font-semibold text-slate-500">Final Output</h3>
        <p id="translated-text" class="mt-1 text-xl font-semibold text-indigo-700 animate-popIn">${d.translated_text}</p>
      </div>
    </div>
  `;

  // stagger-in animation for retrieved docs
  document.querySelectorAll('.result-item').forEach((el, i) => {
    el.style.animationDelay = `${i * 120}ms`;
  });
}

function renderMetrics(m) {
  metricsContainer.innerHTML = `
    <div class="grid grid-cols-2 sm:grid-cols-3 gap-4 animate-fadeIn">
      <div>
        <p class="text-sm text-slate-500">BLEU</p>
        <p class="text-lg font-bold text-slate-800">${m.bleu}</p>
      </div>
      <div>
        <p class="text-sm text-slate-500">COMET</p>
        <p class="text-lg font-bold text-slate-800">${m.comet}</p>
      </div>
      <div>
        <p class="text-sm text-slate-500">Latency</p>
        <p class="text-lg font-bold text-slate-800">${m.latency_ms} ms</p>
      </div>
      <div>
        <p class="text-sm text-slate-500">NE Preservation</p>
        <p class="text-lg font-bold text-slate-800">${(m.named_entity_preservation*100).toFixed(2)}%</p>
      </div>
      <div>
        <p class="text-sm text-slate-500">Toxicity Leakage</p>
        <p class="text-lg font-bold text-slate-800">${(m.toxicity_leakage*100).toFixed(2)}%</p>
      </div>
    </div>
  `;
}

// --- 3D tilt (light) ---
function setup3DTilt() {
  if (!tiltCard) return;
  tiltCard.addEventListener('mousemove', (e) => {
    const rect = tiltCard.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    tiltCard.style.transform = `rotateX(${(-y / 30)}deg) rotateY(${(x / 30)}deg)`;
  });
  tiltCard.addEventListener('mouseleave', () => {
    tiltCard.style.transform = 'rotateX(0) rotateY(0)';
  });
}
