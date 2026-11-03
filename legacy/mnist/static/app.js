'use strict';

(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const sourceNames = { canvas: '手写画板', upload: '图片上传', sample: '测试样例' };
  const pageNames = { workbench: '识别工作台', history: '识别记录', evaluation: '数据评估', about: '实验改编' };
  const numberFormat = new Intl.NumberFormat('zh-CN');
  const state = { page: 'workbench', mode: 'canvas', busy: false, drawn: false, strokes: [], uploadData: null, info: null, metrics: null, historyPage: 1, historyPages: 1, historyTotal: 0, historyFilters: {}, historyRequest: 0, statsRequest: 0, detailId: null };
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const count = (value) => numberFormat.format(Number(value) || 0);
  const percent = (value, digits = 1) => Number.isFinite(Number(value)) ? `${(Math.min(1, Math.max(0, Number(value))) * 100).toFixed(digits)}%` : '—';
  const safeImage = (value) => typeof value === 'string' && (/^data:image\/(png|jpeg|webp);base64,/i.test(value) || /^\/(?!\/)/.test(value)) ? value : '';
  const dateParts = (value) => {
    if (!value) return { date: '—', time: '' };
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return { date: String(value).slice(0, 10), time: String(value).slice(11, 19) };
    return { date: date.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' }), time: date.toLocaleTimeString('zh-CN', { hour12: false }) };
  };
  const dateText = (value) => { const parts = dateParts(value); return `${parts.date} ${parts.time}`.trim(); };

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 90000);
    try {
      const response = await fetch(path, { credentials: 'same-origin', ...options, signal: options.signal || controller.signal, headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...(options.headers || {}) } });
      let body;
      try { body = await response.json(); } catch { throw new Error(`服务返回了无法读取的响应（HTTP ${response.status}），请检查后端是否正常运行。`); }
      if (!response.ok || body.success === false) throw new Error(body.error || `请求失败（HTTP ${response.status}）`);
      return body;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('请求超时，请确认本机服务正常，再重试。');
      if (error instanceof TypeError) throw new Error('无法连接本机服务，请确认项目已启动。');
      throw error;
    } finally { clearTimeout(timeout); }
  }

  function toast(message, type = '') {
    const element = document.createElement('div');
    element.className = `toast ${type}`;
    element.textContent = message;
    $('#toast-container').append(element);
    setTimeout(() => element.remove(), 5200);
  }
  function inputError(message = '') { $('#input-error').textContent = message; $('#input-error').hidden = !message; }
  function navigate(page) {
    if (!(page in pageNames)) page = 'workbench';
    state.page = page;
    $$('.page').forEach((element) => { element.hidden = element.id !== `page-${page}`; });
    $$('.nav-item').forEach((element) => { const selected = element.dataset.page === page; element.classList.toggle('active', selected); if (selected) element.setAttribute('aria-current', 'page'); else element.removeAttribute('aria-current'); });
    $('#breadcrumb-current').textContent = pageNames[page];
    document.title = `${pageNames[page]} · 墨迹 INKLAB`;
    if (location.hash !== `#${page}`) history.replaceState(null, '', `#${page}`);
    if (page === 'history') loadHistory();
    if (page === 'evaluation') { loadStats(); if (!state.metrics) loadMetrics(); }
  }
  $$('[data-page]').forEach((button) => button.addEventListener('click', () => { navigate(button.dataset.page); window.scrollTo({ top: 0, behavior: 'smooth' }); }));
  $$('[data-go]').forEach((button) => button.addEventListener('click', () => { navigate(button.dataset.go); window.scrollTo({ top: 0, behavior: 'smooth' }); }));
  window.addEventListener('hashchange', () => navigate(location.hash.slice(1)));

  const canvas = $('#digit-canvas');
  const context = canvas.getContext('2d', { willReadFrequently: true });
  let pointerId = null;
  let previousPoint = null;
  function resetCanvas() {
    context.fillStyle = '#ffffff'; context.fillRect(0, 0, canvas.width, canvas.height);
    state.drawn = false; state.strokes = []; previousPoint = null; pointerId = null;
    $('#canvas-hint').hidden = false; $('#undo-button').disabled = true;
    $$('.sample-button.selected').forEach((button) => button.classList.remove('selected'));
  }
  function position(event) { const rect = canvas.getBoundingClientRect(); return { x: (event.clientX - rect.left) * canvas.width / rect.width, y: (event.clientY - rect.top) * canvas.height / rect.height }; }
  function penSettings() { context.strokeStyle = '#172339'; context.fillStyle = '#172339'; context.lineWidth = Number($('#stroke-width').value); context.lineCap = 'round'; context.lineJoin = 'round'; }
  canvas.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 && event.pointerType !== 'touch') return;
    if (state.busy || pointerId !== null) return;
    event.preventDefault();
    state.strokes.push({ image: context.getImageData(0, 0, canvas.width, canvas.height), drawn: state.drawn });
    if (state.strokes.length > 35) state.strokes.shift();
    pointerId = event.pointerId; canvas.setPointerCapture(event.pointerId); previousPoint = position(event); penSettings();
    context.beginPath(); context.arc(previousPoint.x, previousPoint.y, context.lineWidth / 2, 0, Math.PI * 2); context.fill();
    state.drawn = true; $('#canvas-hint').hidden = true; $('#undo-button').disabled = false; inputError();
    $$('.sample-button.selected').forEach((button) => button.classList.remove('selected'));
  });
  canvas.addEventListener('pointermove', (event) => {
    if (pointerId !== event.pointerId || !previousPoint) return;
    event.preventDefault(); penSettings();
    const events = typeof event.getCoalescedEvents === 'function' ? event.getCoalescedEvents() : [event];
    for (const nextEvent of events.length ? events : [event]) { const point = position(nextEvent); context.beginPath(); context.moveTo(previousPoint.x, previousPoint.y); context.lineTo(point.x, point.y); context.stroke(); previousPoint = point; }
  });
  const endStroke = (event) => { if (event.pointerId === pointerId) { if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId); pointerId = null; previousPoint = null; } };
  canvas.addEventListener('pointerup', endStroke); canvas.addEventListener('pointercancel', endStroke);
  $('#clear-button').addEventListener('click', () => { if (!state.busy) { resetCanvas(); inputError(); } });
  $('#undo-button').addEventListener('click', () => { if (state.busy) return; const stroke = state.strokes.pop(); if (stroke) { context.putImageData(stroke.image, 0, 0); state.drawn = stroke.drawn; $('#canvas-hint').hidden = state.drawn; } $('#undo-button').disabled = !state.strokes.length; inputError(); });
  $('#stroke-width').addEventListener('input', (event) => { $('#stroke-width-value').textContent = event.target.value; });
  function setMode(mode) {
    state.mode = mode; inputError();
    $('#canvas-input').hidden = mode !== 'canvas'; $('#upload-input').hidden = mode !== 'upload';
    for (const value of ['canvas', 'upload']) { $(`#mode-${value}`).classList.toggle('selected', mode === value); $(`#mode-${value}`).setAttribute('aria-selected', String(mode === value)); }
  }
  $('#mode-canvas').addEventListener('click', () => { if (!state.busy) setMode('canvas'); });
  $('#mode-upload').addEventListener('click', () => { if (!state.busy) setMode('upload'); });
  function readFile(file) {
    return new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = () => reject(new Error('无法读取这张图片，请重新选择。')); reader.readAsDataURL(file); });
  }
  async function acceptFile(file) {
    if (!file || state.busy) return;
    inputError();
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) return inputError('请选择 PNG、JPEG 或 WEBP 图片。');
    const limit = state.info?.limits?.max_image_bytes || 2 * 1024 * 1024;
    if (file.size > limit) return inputError('图片超过 2 MB，请压缩后重新选择。');
    try {
      const data = await readFile(file);
      const image = new Image();
      await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error('图片无法解析，可能已经损坏。')); image.src = data; });
      const dimension = state.info?.limits?.max_dimension || 4096;
      if (image.width > dimension || image.height > dimension) throw new Error(`图片宽高不能超过 ${dimension} 像素。`);
      state.uploadData = data; $('#upload-preview').src = data; $('#upload-name').textContent = file.name;
      $('#upload-size').textContent = `${image.width} × ${image.height} px · ${(file.size / 1024).toFixed(1)} KB`;
      $('#drop-zone').hidden = true; $('#upload-preview-wrap').hidden = false; setMode('upload');
      $$('.sample-button.selected').forEach((button) => button.classList.remove('selected'));
    } catch (error) { inputError(error.message); }
  }
  $('#image-upload').addEventListener('change', (event) => { acceptFile(event.target.files[0]); event.target.value = ''; });
  $('#replace-upload').addEventListener('click', () => $('#image-upload').click());
  const dropZone = $('#drop-zone');
  ['dragenter', 'dragover'].forEach((type) => dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach((type) => dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); }));
  dropZone.addEventListener('drop', (event) => { const files = event.dataTransfer.files; if (files.length !== 1) return inputError('每次请选择一张图片。'); acceptFile(files[0]); });

  function setBusy(busy) {
    state.busy = busy; $('#predict-button').disabled = busy; $('#predict-button').classList.toggle('busy', busy);
    $('#predict-button > span').textContent = busy ? '模型正在识别…' : '开始识别';
    $('#mode-canvas').disabled = busy; $('#mode-upload').disabled = busy; $('#clear-button').disabled = busy;
    $('#image-upload').disabled = busy; $('#stroke-width').disabled = busy; $('#undo-button').disabled = busy || !state.strokes.length;
    $$('.sample-button').forEach((button) => { button.disabled = busy; });
    if (busy) { $('#result-status').textContent = '推理中'; $('#result-status').className = 'result-status loading'; }
  }
  function probabilitiesOf(value) {
    if (!Array.isArray(value)) return Array(10).fill(0);
    return Array.from({ length: 10 }, (_, index) => { const item = value[index]; if (typeof item === 'number') return Math.max(0, Math.min(1, item)); if (item && typeof item === 'object') return Math.max(0, Math.min(1, Number(item.probability ?? item.confidence) || 0)); return 0; });
  }
  function renderPrediction(result) {
    $('#result-empty').hidden = true; $('#result-content').hidden = false;
    $('#result-digit').textContent = result.digit; $('#result-digit-label').textContent = result.digit; $('#result-confidence').textContent = percent(result.confidence, 2);
    $('#result-elapsed').textContent = Number(result.elapsed_ms || 0).toFixed(1); $('#result-source').textContent = sourceNames[result.source] || sourceNames[state.mode] || '输入图片';
    $('#result-saved').textContent = result.record_id ? `已保存 #${result.record_id}` : '识别完成';
    $('#processed-image').src = safeImage(result.preview || result.thumbnail);
    const probabilities = probabilitiesOf(result.probabilities);
    const top3 = Array.isArray(result.top3) && result.top3.length ? result.top3.slice(0, 3) : probabilities.map((probability, digit) => ({ digit, probability })).sort((a, b) => b.probability - a.probability).slice(0, 3);
    $('#top3-list').innerHTML = top3.map((item) => `<div class="probability-row"><span class="probability-digit">${esc(item.digit)}</span><div class="probability-track"><div class="probability-fill" style="width:${Math.max(0, Math.min(100, (Number(item.probability) || 0) * 100))}%"></div></div><span class="probability-value">${percent(item.probability, 2)}</span></div>`).join('');
    const warnings = Array.isArray(result.warnings) ? result.warnings : [];
    $('#result-warnings').hidden = !warnings.length; $('#result-warnings').innerHTML = warnings.map((warning) => `<p>${esc(warning)}</p>`).join('');
    $('#result-status').textContent = '识别完成'; $('#result-status').className = 'result-status done';
  }
  async function predict(sample) {
    if (state.busy) return;
    inputError(); let payload;
    if (Number.isInteger(sample)) payload = { sample };
    else if (state.mode === 'canvas') { if (!state.drawn) return inputError('请先在画板中写下一个数字，或选择下方样例。'); payload = { image: canvas.toDataURL('image/png'), source: 'canvas' }; }
    else { if (!state.uploadData) return inputError('请先选择一张数字图片。'); payload = { image: state.uploadData, source: 'upload' }; }
    setBusy(true);
    try {
      const result = await api('/api/predict', { method: 'POST', body: JSON.stringify(payload) });
      renderPrediction(result); await Promise.allSettled([loadRecent(), loadStats()]);
    } catch (error) { inputError(error.message); $('#result-status').textContent = '本次识别失败'; $('#result-status').className = 'result-status'; if (!$('#result-content').hidden) toast('本次识别失败，右侧保留上一次成功结果。', 'error'); }
    finally { setBusy(false); }
  }
  $('#predict-button').addEventListener('click', () => predict());
  function renderSamples(samples) {
    $('#sample-grid').innerHTML = samples.map((sample) => `<button class="sample-button" data-digit="${Number(sample.digit)}" aria-label="识别数字 ${Number(sample.digit)} 的真实测试样例" title="数字 ${Number(sample.digit)} · 真实测试样例"><img src="${esc(safeImage(sample.url))}" alt="数字 ${Number(sample.digit)} 的样例" loading="lazy"><span>${Number(sample.digit)}</span></button>`).join('') || '<span class="muted">暂无可用样例</span>';
    $$('.sample-button').forEach((button) => button.addEventListener('click', () => { if (state.busy) return; $$('.sample-button').forEach((item) => item.classList.toggle('selected', item === button)); predict(Number(button.dataset.digit)); }));
  }
  async function loadInfo() {
    try {
      const { info } = await api('/api/info'); state.info = info;
      const ready = Boolean(info.ready); $('#service-status').textContent = ready ? '模型服务已就绪' : '模型暂不可用'; $('#service-dot').className = `status-dot ${ready ? 'ready' : 'error'}`;
      const samples = Array.isArray(info.samples) ? info.samples.filter((sample) => Number.isInteger(Number(sample.digit)) && Number(sample.digit) >= 0 && Number(sample.digit) <= 9) : [];
      renderSamples(samples);
    } catch (error) { $('#service-status').textContent = '服务连接失败'; $('#service-dot').className = 'status-dot error'; $('#sample-grid').innerHTML = '<span class="muted">样例暂不可用</span>'; inputError(error.message); }
  }

  async function loadRecent() {
    try {
      const data = await api('/api/history?page=1&per_page=4');
      $('#nav-record-count').textContent = Number(data.total) > 999 ? '999+' : count(data.total); $('#overview-total').textContent = count(data.total);
      $('#recent-records').innerHTML = data.records.length ? data.records.map((record) => `<button class="recent-card" data-detail-id="${Number(record.id)}" aria-label="查看记录 ${Number(record.id)}，识别数字 ${esc(record.digit)}"><img src="${esc(safeImage(record.thumbnail))}" alt="记录输入缩略图"><div><strong>数字 ${esc(record.digit)}<span>${esc(sourceNames[record.source] || '其他')}</span></strong><p>${esc(dateText(record.created_at))}</p></div><small>${percent(record.confidence)}</small></button>`).join('') : '<p class="empty-inline">还没有识别记录，试着写下第一个数字吧。</p>';
    } catch { $('#recent-records').innerHTML = '<p class="empty-inline">暂时无法读取记录，完成识别后将自动重试。</p>'; }
  }
  function filterParams(filters = state.historyFilters, includePage = false) {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(filters)) if (value !== '' && value != null) params.set(key, value);
    if (includePage) { params.set('page', state.historyPage); params.set('per_page', $('#per-page').value); }
    return params.toString();
  }
  function selectedFilters() { return { digit: $('#filter-digit').value, source: $('#filter-source').value, date_from: $('#filter-from').value, date_to: $('#filter-to').value }; }
  async function loadHistory() {
    const request = ++state.historyRequest;
    $('#history-error').hidden = true; $('#history-body').setAttribute('aria-busy', 'true');
    $('#previous-page').disabled = true; $('#next-page').disabled = true;
    try {
      const data = await api(`/api/history?${filterParams(state.historyFilters, true)}`);
      if (request !== state.historyRequest) return;
      state.historyTotal = Number(data.total) || 0; state.historyPages = Math.max(1, Number(data.pages) || 1);
      if (state.historyPage > state.historyPages) { state.historyPage = state.historyPages; return loadHistory(); }
      $('#history-total').textContent = count(data.total);
      $('#history-body').innerHTML = data.records.length ? data.records.map((record) => {
        const date = dateParts(record.created_at);
        return `<tr><td><img class="record-thumbnail" src="${esc(safeImage(record.thumbnail))}" alt="记录 ${Number(record.id)} 的图像"></td><td><span class="record-digit">${esc(record.digit)}</span></td><td><span class="table-confidence">${percent(record.confidence, 2)}</span></td><td><span class="source-tag ${esc(record.source)}">${esc(sourceNames[record.source] || record.source)}</span></td><td class="record-time">${esc(date.date)}<small>${esc(date.time)}</small></td><td>${record.corrected_digit != null ? `<span class="feedback-tag">校对为 ${esc(record.corrected_digit)}</span>` : record.note ? '<span class="feedback-tag">已备注</span>' : '<span class="muted">—</span>'}</td><td><div class="table-actions"><button class="text-button" data-detail-id="${Number(record.id)}">详情</button><button class="text-button delete-record" data-delete-id="${Number(record.id)}">删除</button></div></td></tr>`;
      }).join('') : '<tr><td colspan="7" class="table-empty"><strong>这里还没有记录</strong>尝试调整筛选条件，或回到工作台完成一次识别。</td></tr>';
      $('#pagination-label').textContent = `共 ${count(data.total)} 条 · 第 ${state.historyPage} / ${state.historyPages} 页`;
      $('#previous-page').disabled = state.historyPage <= 1; $('#next-page').disabled = state.historyPage >= state.historyPages;
      const filtered = Object.values(state.historyFilters).some(Boolean); $('#history-summary').textContent = filtered ? '当前显示筛选结果，导出 CSV 使用相同条件' : '所有成功识别都会保存在本机数据库';
    } catch (error) { if (request === state.historyRequest) { $('#history-error').textContent = error.message; $('#history-error').hidden = false; $('#history-body').innerHTML = '<tr><td colspan="7" class="table-empty">读取失败，请重新筛选或切换页面重试。</td></tr>'; } }
    finally { if (request === state.historyRequest) $('#history-body').removeAttribute('aria-busy'); }
  }
  $('#history-filters').addEventListener('submit', (event) => {
    event.preventDefault(); const filters = selectedFilters();
    if (filters.date_from && filters.date_to && filters.date_from > filters.date_to) return toast('开始日期不能晚于结束日期。', 'error');
    state.historyFilters = filters; state.historyPage = 1; loadHistory();
  });
  $('#history-filters').addEventListener('reset', () => { state.historyFilters = {}; state.historyPage = 1; setTimeout(loadHistory, 0); });
  $('#per-page').addEventListener('change', () => { state.historyPage = 1; loadHistory(); });
  $('#previous-page').addEventListener('click', () => { if (state.historyPage > 1) { state.historyPage--; loadHistory(); } });
  $('#next-page').addEventListener('click', () => { if (state.historyPage < state.historyPages) { state.historyPage++; loadHistory(); } });
  $('#export-button').addEventListener('click', async () => {
    const button = $('#export-button'); button.disabled = true;
    try {
      const response = await fetch(`/api/export/csv?${filterParams()}`, { credentials: 'same-origin' });
      if (!response.ok) { let message = '导出失败，请重试。'; try { message = (await response.json()).error || message; } catch {} throw new Error(message); }
      const blob = await response.blob(); const url = URL.createObjectURL(blob); const link = document.createElement('a');
      const today = new Date();
      const localDate = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
      link.href = url; link.download = `INKLAB_识别记录_${localDate}.csv`; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 30000); toast('CSV 已导出，使用当前已应用的筛选条件。', 'success');
    } catch (error) { toast(error.message || '导出失败，请确认本机服务正常。', 'error'); } finally { button.disabled = false; }
  });

  const confirmDialog = $('#confirm-dialog'); let confirmResolve = null;
  function finishConfirm(value) { confirmDialog.close(); const resolve = confirmResolve; confirmResolve = null; resolve?.(value); }
  function confirmAction(title, message, label = '确认删除') { return new Promise((resolve) => { $('#confirm-title').textContent = title; $('#confirm-message').textContent = message; $('#confirm-proceed').textContent = label; confirmResolve = resolve; confirmDialog.showModal(); $('#confirm-cancel').focus(); }); }
  $('#confirm-cancel').addEventListener('click', () => finishConfirm(false)); $('#confirm-proceed').addEventListener('click', () => finishConfirm(true));
  confirmDialog.addEventListener('cancel', (event) => { event.preventDefault(); finishConfirm(false); });
  confirmDialog.addEventListener('click', (event) => { if (event.target === confirmDialog) { const rect = confirmDialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) finishConfirm(false); } });
  async function refreshRecords() { await Promise.allSettled([loadRecent(), loadStats(), state.page === 'history' ? loadHistory() : Promise.resolve()]); }
  async function deleteRecord(id) {
    const accepted = await confirmAction('删除这条识别记录？', `记录 #${id} 的预测、缩略图和人工备注将一并删除，操作无法撤销。`);
    if (!accepted) return;
    try { await api(`/api/history/${id}`, { method: 'DELETE' }); if (state.detailId === id) $('#record-dialog').close(); toast('记录已删除。', 'success'); await refreshRecords(); }
    catch (error) { toast(error.message, 'error'); }
  }
  $('#clear-history-button').addEventListener('click', async () => {
    const accepted = await confirmAction('清空工作空间的全部记录？', '将删除数据库中的所有识别记录，包括当前筛选条件之外的记录。模型和训练指标会保留。此操作无法撤销。', '清空全部记录');
    if (!accepted) return;
    try { await api('/api/history', { method: 'DELETE', body: JSON.stringify({ confirm: 'DELETE_ALL' }) }); state.historyPage = 1; toast('全部识别记录已清空。', 'success'); await refreshRecords(); } catch (error) { toast(error.message, 'error'); }
  });
  document.addEventListener('click', (event) => {
    const detail = event.target.closest('[data-detail-id]'); if (detail) openDetail(Number(detail.dataset.detailId));
    const deletion = event.target.closest('[data-delete-id]'); if (deletion) deleteRecord(Number(deletion.dataset.deleteId));
  });
  $('#record-dialog').addEventListener('close', () => { state.detailId = null; });
  async function openDetail(id) {
    state.detailId = id; const dialog = $('#record-dialog'); $('#record-detail-body').innerHTML = '<p class="empty-inline">正在读取记录详情…</p>'; if (!dialog.open) dialog.showModal();
    try {
      const { record } = await api(`/api/history/${id}`); if (state.detailId !== id || !dialog.open) return;
      const probabilities = probabilitiesOf(record.probabilities);
      $('#record-detail-body').innerHTML = `<div class="detail-hero"><img src="${esc(safeImage(record.thumbnail))}" alt="记录 ${id} 的输入预览"><div><small>模型预测 · 记录 #${id}</small><div class="detail-digit">${esc(record.digit)}</div><small>置信度 ${percent(record.confidence, 2)}</small></div></div><dl class="detail-meta"><div><dt>识别时间</dt><dd>${esc(dateText(record.created_at))}</dd></div><div><dt>输入来源</dt><dd>${esc(sourceNames[record.source] || record.source)}</dd></div><div><dt>推理耗时</dt><dd>${Number(record.elapsed_ms || 0).toFixed(1)} ms</dd></div><div><dt>原始预测</dt><dd>数字 ${esc(record.digit)}（人工校对不会修改此项）</dd></div></dl><div class="detail-probabilities" aria-label="全部数字的预测概率">${probabilities.map((probability, digit) => `<div class="detail-probability ${Number(record.digit) === digit ? 'winner' : ''}"><strong>${digit}</strong><span>${percent(probability, 2)}</span></div>`).join('')}</div><form class="feedback-form" id="feedback-form"><h3>人工校对与备注</h3><p>可记录你认为正确的数字，用于复核；此操作不会修改模型，也不会自动重训。</p><label for="corrected-digit">校对数字<select id="corrected-digit"><option value="">未校对 / 清除校对</option>${Array.from({ length: 10 }, (_, digit) => `<option value="${digit}" ${record.corrected_digit != null && Number(record.corrected_digit) === digit ? 'selected' : ''}>数字 ${digit}</option>`).join('')}</select></label><label for="record-note">记录备注<textarea id="record-note" maxlength="300" placeholder="例如：这是数字 7，横线写得较短。">${esc(record.note || '')}</textarea></label><div class="textarea-counter"><span id="note-count">${String(record.note || '').length}</span> / 300</div><p class="dialog-error" id="feedback-error" hidden></p><div class="feedback-actions"><button class="danger-text-button" type="button" data-delete-id="${id}"><svg><use href="#i-trash"/></svg>删除此记录</button><button class="primary-button" id="save-feedback" type="submit">保存校对与备注</button></div></form>`;
      $('#record-note').addEventListener('input', (event) => { $('#note-count').textContent = event.target.value.length; });
      $('#feedback-form').addEventListener('submit', async (event) => {
        event.preventDefault(); const value = $('#corrected-digit').value; const note = $('#record-note').value; const button = $('#save-feedback'); $('#feedback-error').hidden = true; button.disabled = true; button.textContent = '正在保存…';
        try { await api(`/api/history/${id}`, { method: 'PATCH', body: JSON.stringify({ corrected_digit: value === '' ? null : Number(value), note }) }); toast('人工校对与备注已保存。', 'success'); await refreshRecords(); }
        catch (error) { $('#feedback-error').textContent = error.message; $('#feedback-error').hidden = false; }
        finally { button.disabled = false; button.textContent = '保存校对与备注'; }
      });
    } catch (error) { if (state.detailId === id) $('#record-detail-body').innerHTML = `<p class="dialog-error">${esc(error.message)}</p>`; }
  }

  async function loadStats() {
    const request = ++state.statsRequest;
    try {
      const { stats } = await api('/api/stats'); if (request !== state.statsRequest) return;
      $('#stat-total').textContent = count(stats.total); $('#overview-total').textContent = count(stats.total); $('#nav-record-count').textContent = Number(stats.total) > 999 ? '999+' : count(stats.total);
      $('#stat-confidence').textContent = Number(stats.total) ? percent(stats.mean_confidence) : '—';
      $('#stat-elapsed').innerHTML = Number(stats.total) ? `${Number(stats.mean_elapsed_ms || 0).toFixed(1)}<small>ms</small>` : '—'; $('#stat-feedback').textContent = count(stats.feedback_count);
      const distribution = Array.from({ length: 10 }, (_, digit) => ({ digit, count: Number((stats.by_digit || []).find((item) => Number(item.digit) === digit)?.count) || 0 }));
      const max = Math.max(1, ...distribution.map((item) => item.count));
      $('#digit-distribution').innerHTML = distribution.map((item) => `<div class="distribution-bar" title="预测数字 ${item.digit}：${item.count} 条"><span>${item.count}</span><div class="bar-track"><div class="bar-fill ${item.count === max ? 'largest' : ''}" style="height:${item.count / max * 100}%"></div></div><span>${item.digit}</span></div>`).join('');
      $('#source-distribution').innerHTML = ['canvas', 'upload', 'sample'].map((source) => { const value = Number((stats.by_source || []).find((item) => item.source === source)?.count) || 0; return `<div class="source-row"><span>${sourceNames[source]}</span><div class="probability-track"><div class="probability-fill" style="width:${stats.total ? value / stats.total * 100 : 0}%"></div></div><strong>${count(value)} 条</strong></div>`; }).join('');
    } catch (error) { if (state.page === 'evaluation') toast(`使用统计读取失败：${error.message}`, 'error'); }
  }
  $('#refresh-stats').addEventListener('click', async () => { const button = $('#refresh-stats'); button.disabled = true; await loadStats(); button.disabled = false; });
  function drawTrainingChart(history) {
    if (!history.length) { $('#training-chart').innerHTML = '<p class="empty-inline">没有可用的训练历史。</p>'; return; }
    const width = 500, height = 285, left = 49, right = 51, top = 24, bottom = 40;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const values = history.map((item) => ({ epoch: Number(item.epoch), accuracy: Number(item.test_accuracy) > 1 ? Number(item.test_accuracy) / 100 : Number(item.test_accuracy), loss: Number(item.train_loss) }));
    const accuracies = values.map((item) => item.accuracy).filter(Number.isFinite);
    const minAccuracy = Math.max(0, Math.floor((Math.min(...accuracies, 0.98) - 0.005) * 100) / 100);
    const maxAccuracy = 1;
    const maxLoss = Math.max(0.1, ...values.map((item) => Number.isFinite(item.loss) ? item.loss : 0)) * 1.15;
    const x = (index) => left + (history.length === 1 ? plotWidth / 2 : index / (history.length - 1) * plotWidth);
    const yAccuracy = (value) => top + (maxAccuracy - value) / (maxAccuracy - minAccuracy) * plotHeight;
    const yLoss = (value) => top + (1 - value / maxLoss) * plotHeight;
    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="真实训练过程：测试准确率和训练损失的逐轮变化"><text x="${left}" y="10" font-size="9" fill="#9eacc1">准确率 %</text><text x="${width - right}" y="10" text-anchor="end" font-size="9" fill="#bba68f">训练损失</text>`;
    for (let index = 0; index <= 4; index++) {
      const y = top + index / 4 * plotHeight;
      svg += `<line x1="${left}" y1="${y}" x2="${width - right}" y2="${y}" stroke="#edf1f7" stroke-dasharray="3 4"/><text x="${left - 9}" y="${y + 3}" text-anchor="end" font-size="9" fill="#9aa8bd">${((maxAccuracy - index / 4 * (maxAccuracy - minAccuracy)) * 100).toFixed(1)}</text><text x="${width - right + 9}" y="${y + 3}" font-size="9" fill="#b8a68f">${(maxLoss * (1 - index / 4)).toFixed(2)}</text>`;
    }
    const accuracyPoints = values.map((item, index) => `${x(index)},${yAccuracy(item.accuracy)}`).join(' ');
    const lossPoints = values.map((item, index) => `${x(index)},${yLoss(item.loss)}`).join(' ');
    svg += `<polyline points="${accuracyPoints}" fill="none" stroke="#365dec" stroke-width="2.5" stroke-linejoin="round"/><polyline points="${lossPoints}" fill="none" stroke="#ee9760" stroke-width="2" stroke-linejoin="round"/>`;
    values.forEach((item, index) => {
      svg += `<circle cx="${x(index)}" cy="${yAccuracy(item.accuracy)}" r="4" fill="white" stroke="#365dec" stroke-width="2"><title>第 ${item.epoch} 轮，测试准确率 ${percent(item.accuracy, 2)}</title></circle><circle cx="${x(index)}" cy="${yLoss(item.loss)}" r="3.5" fill="white" stroke="#ee9760" stroke-width="2"><title>第 ${item.epoch} 轮，训练损失 ${item.loss.toFixed(5)}</title></circle><text x="${x(index)}" y="${height - 15}" text-anchor="middle" font-size="10" fill="#97a6bd">第 ${item.epoch} 轮</text>`;
    });
    $('#training-chart').innerHTML = `${svg}</svg>`;
  }
  function drawConfusionMatrix(matrix) {
    if (!Array.isArray(matrix) || matrix.length !== 10 || matrix.some((row) => !Array.isArray(row) || row.length !== 10)) { $('#confusion-matrix').innerHTML = '<p class="empty-inline" style="grid-column:1/-1">没有可用的混淆矩阵。</p>'; return; }
    const max = Math.max(1, ...matrix.flat().map(Number)); let html = '<span></span>';
    for (let column = 0; column < 10; column++) html += `<span class="matrix-label">${column}</span>`;
    for (let row = 0; row < 10; row++) {
      html += `<span class="matrix-label">${row}</span>`;
      for (let column = 0; column < 10; column++) {
        const value = Number(matrix[row][column]) || 0; const intensity = Math.sqrt(value / max); const alpha = 0.035 + intensity * 0.93;
        html += `<button type="button" class="matrix-cell" style="background:rgba(54,93,236,${alpha.toFixed(3)});color:${intensity > 0.55 ? '#fff' : value ? '#7a90c6' : '#bfc9dc'}" title="真实数字 ${row} → 预测数字 ${column}：${value} 张" aria-label="真实数字 ${row}，预测数字 ${column}，${value} 张">${value}</button>`;
      }
    }
    $('#confusion-matrix').innerHTML = html;
    $$('.matrix-cell').forEach((button) => button.addEventListener('click', () => toast(button.title)));
  }
  async function loadMetrics() {
    try {
      const data = await api('/api/metrics'); const metrics = data.metrics; state.metrics = metrics; $('#metrics-error').hidden = true;
      const accuracy = Number(metrics.accuracy) > 1 ? Number(metrics.accuracy) / 100 : Number(metrics.accuracy);
      $('#overview-accuracy').innerHTML = `${(accuracy * 100).toFixed(2)}<span>%</span>`;
      $('#overview-samples').textContent = `${count(metrics.train_samples)} / ${count(metrics.total_samples)}`;
      $('#metric-accuracy').textContent = percent(accuracy, 2); $('#metric-correct').innerHTML = `${count(metrics.correct)}<small>/ ${count(metrics.total_samples)}</small>`;
      $('#metric-training').textContent = count(metrics.train_samples);
      const history = Array.isArray(metrics.history) ? metrics.history : [];
      $('#metric-epochs').innerHTML = `${history.length}<small>轮 · CPU</small>`; $('#metric-device').textContent = `实际设备：${metrics.device || 'CPU'}${Number.isFinite(Number(metrics.elapsed_seconds)) ? ` · 训练 ${Number(metrics.elapsed_seconds).toFixed(1)} s` : ''}`;
      drawTrainingChart(history); drawConfusionMatrix(metrics.confusion_matrix);
    } catch (error) { $('#metrics-error').textContent = `模型评估暂不可用：${error.message}`; $('#metrics-error').hidden = false; $('#training-chart').innerHTML = '<p class="empty-inline">训练数据读取失败</p>'; }
  }

  resetCanvas(); navigate(location.hash.slice(1) || 'workbench');
  Promise.allSettled([loadInfo(), loadMetrics(), loadStats(), loadRecent()]);
})();
