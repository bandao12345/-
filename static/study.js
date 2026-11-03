/* Learning materials and assistant. Untrusted content is always rendered as text. */
(() => {
  'use strict';
  const $ = selector => document.querySelector(selector);
  const $$ = selector => [...document.querySelectorAll(selector)];
  const core = window.inklabCore;
  const state = {
    documents: [], documentsRequest: 0, status: null, file: null, importing: false,
    searching: false, chatting: false, draftBusy: false, draftRecordId: null,
    previewId: null, previewTitle: '', previewRequest: 0, draftRequest: 0,
    history: [], latestOCR: null, corrected: new Map(), limits: { maxBytes: 2097152, maxText: 200000, maxPdfPages: 40 },
  };
  const sourceNames = {manual: '文本笔记', text: '文本笔记', upload: '导入文件', file: '导入文件', ocr: 'OCR 校对笔记', txt: 'TXT 资料', md: 'Markdown 资料', pdf: 'PDF 资料'};
  function node(tag, className, text) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = String(text);
    return item;
  }
  function icon(id) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', '#i-' + id);
    svg.setAttribute('aria-hidden', 'true');
    svg.append(use);
    return svg;
  }
  const number = value => Math.max(0, Number(value) || 0).toLocaleString('zh-CN');
  const positiveId = value => Number.isSafeInteger(Number(value)) && Number(value) > 0 ? Number(value) : null;
  const documentId = value => typeof value === 'string' && /^[a-f0-9]{32}$/.test(value) ? value : null;
  const date = value => { const parsed = new Date(value); return Number.isNaN(parsed.getTime()) ? '时间未记录' : parsed.toLocaleString('zh-CN', { hour12: false }); };
  function errorAt(selector, message = '') { const target = $(selector); target.textContent = message; target.hidden = !message; }
  function notify(message, type = '') { core?.toast(message, type); }
  async function api(path, options = {}, timeoutMs = 60000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const headers = new Headers(options.headers || {});
    if (typeof options.body === 'string') headers.set('Content-Type', 'application/json');
    try {
      const response = await fetch(path, { credentials: 'same-origin', ...options, headers, signal: controller.signal });
      let result;
      try { result = await response.json(); } catch (_) { throw new Error('服务未返回有效数据，请确认本机应用已更新并正常运行。'); }
      if (!response.ok || result.success === false) throw new Error(result.error || '请求失败，请稍后重试。');
      return result;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('请求等待超时，请稍后重试；提交资料后可先刷新列表，确认是否已保存。');
      throw error;
    } finally { clearTimeout(timer); }
  }
  function go(page) { core?.navigate(page); }

  function updateAssistantMode() {
    const enabled = $('#assistant-enable-llm').checked && Boolean(state.status?.llm_configured);
    const local = state.status?.llm_scope === 'local';
    const badge = $('#assistant-mode-badge');
    badge.classList.toggle('online-selected', enabled);
    badge.replaceChildren(node('i'), document.createTextNode(enabled ? local ? '已选择本机模型生成' : '已选择云模型生成' : '离线检索摘录'));
    $('#assistant-llm-label').textContent = local ? '使用本机模型生成' : '允许发送到已配置的云模型';
    $('#assistant-llm-privacy').textContent = local ? '主动开启后，问题、对话和检索片段交给本机模型处理，无需外部账号。' : '主动开启后，问题、对话和检索片段会发送到已配置的云模型。';
    if (!state.status?.llm_configured) {
      $('#assistant-llm-label').textContent = '启用模型生成（需先配置）';
      $('#assistant-llm-privacy').textContent = '可接入本机模型或云模型；本页默认只做本地检索摘录。';
    }
    $('#assistant-send-hint').textContent = enabled ? local ? '本机模型将使用问题、对话与检索片段生成回答' : '将把问题、对话与资料片段发送到已配置的云模型' : '本地检索与摘录，不调用大语言模型';
    $('#assistant-send span').textContent = state.chatting ? '正在处理…' : enabled ? '检索并请求模型' : '检索并回答';
  }
  async function loadStatus() {
    const button = $('#refresh-assistant-status');
    button.disabled = true;
    try {
      const data = await api('/api/assistant/status', {}, 15000);
      const status = data.status || {};
      state.status = status;
      const limits = status.limits || {};
      state.limits.maxBytes = Number(limits.max_upload_bytes) || state.limits.maxBytes;
      state.limits.maxText = Number(limits.max_text_characters) || state.limits.maxText;
      state.limits.maxPdfPages = Number(limits.max_pdf_pages) || state.limits.maxPdfPages;
      $('#study-draft-text').maxLength = state.draftRecordId ? Math.min(2000, state.limits.maxText) : state.limits.maxText;
      $('#knowledge-file-limit').textContent = `TXT / Markdown / 文本型 PDF · 最大 ${number(state.limits.maxBytes / 1048576)} MB · PDF ≤ ${number(state.limits.maxPdfPages)} 页`;
      $('#assistant-enable-llm').disabled = state.chatting || !status.llm_configured;
      if (!status.llm_configured) $('#assistant-enable-llm').checked = false;
      $('#assistant-connection-state').textContent = status.llm_configured
        ? `${status.llm_scope === 'local' ? '本机模型' : '云模型'}参数已配置${status.llm_model ? '：' + status.llm_model : ''}。${status.llm_scope === 'local' ? '生成在本机运行。' : ''}实际调用是否成功以本次回答为准。`
        : '本机尚未配置模型服务。离线检索摘录可以直接使用；配置后刷新，再主动开启。';
      $('#assistant-retrieval-method').textContent = status.retrieval_method || '本地词法检索';
      $('#assistant-document-count').textContent = `${number(status.document_count)} 份 / ${number(status.chunk_count)} 个片段`;
      $('#assistant-embedding-status').textContent = status.embedding ? '后端报告已启用' : '未使用（词法检索）';
      $('#knowledge-document-total').textContent = number(status.document_count);
      $('#knowledge-chunk-total').textContent = number(status.chunk_count);
      updateAssistantMode();
    } catch (error) {
      state.status = null;
      $('#assistant-enable-llm').checked = false;
      $('#assistant-enable-llm').disabled = true;
      $('#assistant-connection-state').textContent = '配置状态读取失败：' + error.message + ' 当前仅允许尝试离线检索。';
      updateAssistantMode();
    } finally { button.disabled = state.chatting; }
  }

  function documentKind(document) {
    if (document.source_record_id || document.source_type === 'ocr') return 'OCR 校对笔记';
    const extension = String(document.filename || '').split('.').pop().toLowerCase();
    if (['txt', 'md', 'pdf'].includes(extension)) return extension.toUpperCase() + ' 资料';
    return sourceNames[document.source_type] || '学习资料';
  }
  function renderDocuments() {
    const list = $('#knowledge-document-list');
    list.replaceChildren();
    $('#knowledge-list-count').textContent = number(state.documents.length);
    $('#nav-document-count').textContent = number(state.documents.length);
    $('#knowledge-document-total').textContent = number(state.documents.length);
    $('#knowledge-chunk-total').textContent = number(state.documents.reduce((total, item) => total + (Number(item.chunk_count) || 0), 0));
    if (!state.documents.length) {
      const empty = node('div', 'study-library-empty');
      empty.append(icon('library'), node('h3', '', '第一份笔记，就是知识库的起点'), node('p', '', '上传 TXT、Markdown 或文本型 PDF，也可以把核对后的手写识别结果整理进来。'));
      const button = node('button', 'secondary-button', '添加一段学习笔记');
      button.type = 'button';
      button.addEventListener('click', () => openDraft());
      empty.append(button);
      list.append(empty);
      return;
    }
    for (const document of state.documents) {
      const card = node('article', 'study-document-card');
      const mark = node('span', 'study-document-mark');
      mark.append(icon(document.source_record_id ? 'pen' : 'book'));
      const info = node('div', 'study-document-info');
      const title = node('button', 'study-document-title', document.title || document.filename || '未命名资料');
      title.type = 'button';
      title.addEventListener('click', () => previewDocument(document.id));
      info.append(title, node('p', '', `${documentKind(document)} · ${number(document.characters)} 字符 · ${number(document.chunk_count)} 个片段`), node('small', '', date(document.created_at)));
      const actions = node('div', 'study-document-actions');
      const preview = node('button', 'text-button', '查看原文');
      preview.type = 'button';
      preview.addEventListener('click', () => previewDocument(document.id));
      const remove = node('button', 'study-delete-document');
      remove.type = 'button';
      remove.title = '删除资料';
      remove.setAttribute('aria-label', `删除资料：${document.title || document.filename || '未命名资料'}`);
      remove.append(icon('trash'));
      remove.addEventListener('click', () => deleteDocument(document, remove));
      actions.append(preview, remove);
      card.append(mark, info, actions);
      list.append(card);
    }
  }
  async function loadDocuments() {
    const request = ++state.documentsRequest;
    $('#refresh-knowledge').disabled = true;
    errorAt('#knowledge-list-error');
    try {
      const data = await api('/api/knowledge/documents', {}, 20000);
      if (request !== state.documentsRequest) return;
      state.documents = Array.isArray(data.documents) ? data.documents : [];
      renderDocuments();
    } catch (error) {
      if (request !== state.documentsRequest) return;
      errorAt('#knowledge-list-error', error.message);
      if (!state.documents.length) $('#knowledge-document-list').replaceChildren(node('p', 'empty-inline', '资料列表读取失败，可点击“刷新资料”重试。'));
    } finally { if (request === state.documentsRequest) $('#refresh-knowledge').disabled = false; }
  }
  async function refreshLibrary() { await Promise.allSettled([loadDocuments(), loadStatus()]); }
  async function deleteDocument(document, button) {
    const id = documentId(document.id);
    if (!id) return notify('资料编号无效。', 'error');
    const approved = core?.confirmAction ? await core.confirmAction('删除这份学习资料？', `“${document.title || document.filename || '学习资料'}”及其检索片段将从本机知识库删除。原始 OCR 记录不受影响。`, '删除资料') : window.confirm('确定删除这份学习资料及其检索片段？');
    if (!approved) return;
    button.disabled = true;
    try {
      await api(`/api/knowledge/documents/${id}`, { method: 'DELETE' });
      if (state.previewId === id) $('#study-document-dialog').close();
      notify('资料与检索索引已删除。', 'success');
      await refreshLibrary();
      if ($('#knowledge-search-query').value.trim()) $('#knowledge-search-results').replaceChildren(node('p', 'study-muted', '资料已更新，请重新检索。此前助手回答中的引用可能来自已删除资料。'));
    } catch (error) { notify(error.message, 'error'); }
    finally { if (button.isConnected) button.disabled = false; }
  }
  async function previewDocument(value) {
    const id = documentId(value);
    if (!id) return notify('资料编号无效。', 'error');
    const request = ++state.previewRequest;
    state.previewId = id;
    $('#study-document-heading').textContent = '正在读取资料…';
    $('#study-document-meta').textContent = '';
    $('#study-document-text').textContent = '';
    $('#study-document-truncated').hidden = true;
    $('#study-document-ask').disabled = true;
    if (!$('#study-document-dialog').open) $('#study-document-dialog').showModal();
    try {
      const data = await api(`/api/knowledge/documents/${id}`, {}, 20000);
      if (request !== state.previewRequest || !$('#study-document-dialog').open) return;
      const document = data.document || {};
      state.previewTitle = document.title || document.filename || '学习资料';
      $('#study-document-heading').textContent = state.previewTitle;
      $('#study-document-meta').textContent = `${documentKind(document)} · ${number(document.characters)} 字符 · ${number(document.chunk_count)} 个片段 · ${date(document.created_at)}`;
      $('#study-document-text').textContent = document.text || '暂无可显示的正文。';
      $('#study-document-truncated').hidden = !document.truncated;
      $('#study-document-ask').disabled = false;
    } catch (error) { if (request === state.previewRequest) { $('#study-document-heading').textContent = '资料读取失败'; $('#study-document-text').textContent = error.message; } }
  }

  function acceptKnowledgeFile(file) {
    if (!file || state.importing) return;
    errorAt('#knowledge-import-error');
    if (!/\.(txt|md|pdf)$/i.test(file.name)) return errorAt('#knowledge-import-error', '请选择 TXT、Markdown（.md）或文本型 PDF 文件。');
    if (file.size > state.limits.maxBytes) return errorAt('#knowledge-import-error', `资料不能超过 ${number(state.limits.maxBytes / 1048576)} MB。`);
    if (!file.size) return errorAt('#knowledge-import-error', '这个文件是空的，请选择包含学习内容的资料。');
    state.file = file;
    $('#knowledge-file-name').textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
    $('#knowledge-file-title').value = file.name.replace(/\.[^.]+$/, '').slice(0, 120);
    $('#import-file-button').disabled = false;
  }
  function setImporting(value) {
    state.importing = value;
    for (const selector of ['#knowledge-file', '#knowledge-file-title', '#import-file-button']) $(selector).disabled = value || (selector === '#import-file-button' && !state.file);
    $('#import-file-button').classList.toggle('busy', value);
    $('#knowledge-drop-zone').setAttribute('aria-busy', String(value));
  }
  async function importFile(event) {
    event.preventDefault();
    if (state.importing) return;
    if (!state.file) return errorAt('#knowledge-import-error', '请先选择一份学习资料。');
    errorAt('#knowledge-import-error');
    const body = new FormData();
    body.append('file', state.file);
    if ($('#knowledge-file-title').value.trim()) body.append('title', $('#knowledge-file-title').value.trim());
    setImporting(true);
    try {
      const data = await api('/api/knowledge/documents', { method: 'POST', body });
      notify(data.duplicate ? '这份资料已存在，已保留原有索引。' : '学习资料已导入并建立检索索引。', 'success');
      state.file = null;
      $('#knowledge-file').value = '';
      $('#knowledge-file-title').value = '';
      $('#knowledge-file-name').textContent = '拖入资料，或点击选择文件';
      await refreshLibrary();
    } catch (error) { errorAt('#knowledge-import-error', error.message); }
    finally { setImporting(false); }
  }

  function draftLimit() { return state.draftRecordId ? Math.min(2000, state.limits.maxText) : state.limits.maxText; }
  function updateDraftCount() { $('#study-draft-count').textContent = `${number($('#study-draft-text').value.length)} / ${number(draftLimit())} 字符`; }
  function openDraft(details = {}) {
    if (state.draftBusy) return notify('笔记正在保存，请稍候。');
    state.draftRequest++;
    state.draftRecordId = positiveId(details.record_id);
    $('#study-draft-text').maxLength = draftLimit();
    let text = String(details.text || '');
    if (details.source === 'ocr' && state.corrected.has(state.draftRecordId)) text = state.corrected.get(state.draftRecordId);
    $('#study-draft-heading').textContent = state.draftRecordId ? '先核对，再整理手写笔记' : '整理一份学习笔记';
    $('#study-draft-title').value = String(details.title || '我的学习笔记').slice(0, 120);
    $('#study-draft-text').value = text;
    $('#study-draft-reviewed').checked = false;
    $('#study-draft-source').textContent = state.draftRecordId
      ? `来自识别记录 #${state.draftRecordId}。请核对错别字与结构；确认后先保存这份文字为人工校对，再加入资料库。`
      : '编辑学习内容，核对后加入本机资料库。资料会按纯文本分段建立索引。';
    $('#save-study-draft').textContent = state.draftRecordId ? '保存校对并加入资料库' : '确认加入资料库';
    errorAt('#study-draft-error');
    updateDraftCount();
    if (!$('#study-draft-dialog').open) $('#study-draft-dialog').showModal();
    $('#study-draft-text').focus();
  }
  function setDraftBusy(value) {
    state.draftBusy = value;
    for (const selector of ['#study-draft-title', '#study-draft-text', '#study-draft-reviewed', '#save-study-draft', '#close-study-draft', '#cancel-study-draft']) $(selector).disabled = value;
    $('#save-study-draft').classList.toggle('busy', value);
  }
  async function saveDraft(event) {
    event.preventDefault();
    if (state.draftBusy) return;
    const text = $('#study-draft-text').value.trim(), title = $('#study-draft-title').value.trim();
    errorAt('#study-draft-error');
    if (!title || !text) return errorAt('#study-draft-error', '请填写标题和有效的学习内容。');
    if (!$('#study-draft-reviewed').checked) return errorAt('#study-draft-error', '请先核对文字，再勾选确认。');
    if (text.length > draftLimit()) return errorAt('#study-draft-error', `正文超过 ${number(draftLimit())} 字符，请分成多份资料。`);
    const recordId = state.draftRecordId;
    let correctionSaved = false;
    setDraftBusy(true);
    try {
      let result;
      if (recordId) {
        await api(`/api/history/${recordId}`, { method: 'PATCH', body: JSON.stringify({ corrected_text: text }) });
        state.corrected.set(recordId, text);
        correctionSaved = true;
        result = await api('/api/knowledge/import-ocr', { method: 'POST', body: JSON.stringify({ record_id: recordId, confirmed: true, title, text }) });
        const correctedInput = $('#corrected-text');
        if (correctedInput && $('#record-dialog').open) { correctedInput.value = text; correctedInput.dispatchEvent(new Event('input', { bubbles: true })); }
      } else {
        result = await api('/api/knowledge/documents', { method: 'POST', body: JSON.stringify({ title, text }) });
      }
      $('#study-draft-dialog').close();
      notify(result.duplicate ? '该内容已在资料库中，无需重复导入。' : '已核对的学习笔记已加入资料库。', 'success');
      await refreshLibrary();
    } catch (error) { errorAt('#study-draft-error', (correctionSaved ? '人工校对已保存，资料入库尚未完成：' : '') + error.message); }
    finally { setDraftBusy(false); }
  }

  function sourceCard(source, index) {
    const details = node('details', 'study-citation');
    const summary = node('summary');
    summary.append(node('span', 'study-citation-index', `[${index + 1}]`), node('strong', '', source.title || source.filename || '学习资料'), node('span', 'study-citation-chevron', '＋'));
    details.append(summary);
    const content = node('div', 'study-citation-content');
    content.append(node('p', 'study-citation-location', `片段 ${number((Number(source.position) || 0) + 1)}${source.filename ? ' · ' + source.filename : ''}`), node('blockquote', '', source.excerpt || '暂无摘录内容。'));
    const actions = node('div', 'study-citation-actions');
    const id = documentId(source.document_id);
    if (id) {
      const button = node('button', 'text-button', '查看资料原文');
      button.type = 'button';
      button.addEventListener('click', () => previewDocument(id));
      actions.append(button);
    }
    const recordId = positiveId(source.source_record_id);
    if (recordId) {
      const button = node('button', 'text-button', '回看手写识别记录');
      button.type = 'button';
      button.dataset.detailId = String(recordId);
      actions.append(button);
    }
    content.append(actions);
    details.append(content);
    return details;
  }
  async function searchKnowledge(event) {
    event.preventDefault();
    const query = $('#knowledge-search-query').value.trim();
    if (!query || state.searching) return;
    state.searching = true;
    $('#knowledge-search-button').disabled = true;
    $('#knowledge-search-query').readOnly = true;
    $('#knowledge-search-results').replaceChildren(node('p', 'study-muted', '正在检索本机学习资料…'));
    try {
      const data = await api('/api/knowledge/search?' + new URLSearchParams({ q: query, limit: '4' }), {}, 30000);
      const sources = Array.isArray(data.sources) ? data.sources : [];
      const container = $('#knowledge-search-results');
      container.replaceChildren(node('p', 'study-search-result-count', sources.length ? `找到 ${sources.length} 条相关片段 · ${data.retrieval_method || '本地词法检索'}` : '没有找到相关学习片段。试试更具体的关键词，或先导入有关资料。'));
      sources.forEach((source, index) => { const card = sourceCard(source, index); card.open = index === 0; container.append(card); });
    } catch (error) { $('#knowledge-search-results').replaceChildren(node('p', 'study-inline-error', error.message)); }
    finally { state.searching = false; $('#knowledge-search-button').disabled = false; $('#knowledge-search-query').readOnly = false; }
  }

  function buildTrace(result) {
    const wrapper = node('div', 'study-trace');
    const steps = Array.isArray(result.steps) ? result.steps : [];
    const calls = Array.isArray(result.tool_calls) ? result.tool_calls : [];
    if (!steps.length && !calls.length) { wrapper.append(node('p', 'study-muted', '本次返回未包含执行轨迹。')); return wrapper; }
    if (steps.length) {
      const list = node('ol', 'study-trace-steps');
      steps.forEach((step, index) => {
        const item = node('li');
        const status = String(step.status || '');
        item.append(node('span', 'study-trace-index', number(step.index ?? (index + 1))));
        const info = node('div');
        info.append(node('strong', '', step.name || '处理步骤'), node('p', '', step.message || ''), node('small', '', ({success:'完成',done:'完成',completed:'完成',error:'失败',failed:'失败',skipped:'跳过',running:'执行中'})[status] || status || '状态未提供'));
        item.append(info);
        list.append(item);
      });
      wrapper.append(list);
    }
    if (calls.length) {
      const details = node('details', 'study-tool-calls');
      details.append(node('summary', '', `实际工具调用 · ${calls.length} 次`));
      calls.forEach(call => {
        const item = node('div', 'study-tool-call');
        item.append(node('strong', '', call.tool || '工具'));
        if (call.input !== undefined) item.append(node('pre', '', typeof call.input === 'string' ? call.input : JSON.stringify(call.input, null, 2)));
        item.append(node('p', '', typeof call.result_summary === 'object' ? JSON.stringify(call.result_summary) : call.result_summary || '未提供执行摘要'));
        details.append(item);
      });
      wrapper.append(details);
    }
    return wrapper;
  }
  function appendChat(role, text, result = null) {
    $('#assistant-welcome')?.remove();
    const article = node('article', 'study-chat-message ' + (role === 'user' ? 'study-chat-user' : 'study-chat-answer'));
    const header = node('div', 'study-chat-message-header');
    const mark = node('span', 'study-chat-avatar');
    mark.append(role === 'user' ? document.createTextNode('我') : icon('spark'));
    header.append(mark, node('strong', '', role === 'user' ? '你' : '知识助手'));
    if (result) {
      const cloud = result.mode === 'llm' && result.llm_used === true;
      header.append(node('span', 'study-message-mode' + (cloud ? ' cloud' : ''), cloud ? '实际调用模型' : '离线检索摘录 · 未调用大模型'));
    }
    article.append(header, node('div', 'study-answer-text', text));
    if (result) {
      const sources = Array.isArray(result.sources) ? result.sources : [];
      if (sources.length) {
        const citations = node('section', 'study-answer-citations');
        citations.append(node('h3', '', `引用依据 · ${sources.length} 条原文片段`));
        sources.forEach((source, index) => citations.append(sourceCard(source, index)));
        article.append(citations);
      } else article.append(node('p', 'study-answer-no-source', '本次回答未返回资料引用，请谨慎核对。'));
      if (Array.isArray(result.warnings) && result.warnings.length) article.append(node('p', 'study-answer-no-source', result.warnings.map(String).join('\n')));
      const traceDetails = node('details', 'study-answer-trace');
      traceDetails.append(node('summary', '', '查看本次执行轨迹'), buildTrace(result));
      article.append(traceDetails);
      const footer = node('div', 'study-answer-footer');
      const copy = node('button', 'text-button', '复制回答');
      copy.type = 'button';
      copy.addEventListener('click', async () => { try { await navigator.clipboard.writeText(String(text)); notify('回答已复制。', 'success'); } catch (_) { notify('浏览器未允许复制，请选择文字后手动复制。'); } });
      footer.append(copy, node('span', '', `${(Number(result.elapsed_ms || 0) / 1000).toFixed(1)}s · ${result.agent_type === 'rule_tool_assistant' ? '规则工具流程' : '资料检索流程'}`));
      article.append(footer);
    }
    $('#assistant-chat-log').append(article);
    $('#assistant-chat-log').scrollTop = $('#assistant-chat-log').scrollHeight;
    return article;
  }
  function setChatting(value) {
    state.chatting = value;
    $('#assistant-send').disabled = value;
    $('#assistant-message').readOnly = value;
    $('#assistant-enable-llm').disabled = value || !state.status?.llm_configured;
    $('#assistant-use-agent').disabled = value;
    $('#clear-assistant-chat').disabled = value;
    $('#refresh-assistant-status').disabled = value;
    $('#assistant-send').classList.toggle('busy', value);
    updateAssistantMode();
  }
  async function askAssistant(event) {
    event.preventDefault();
    if (state.chatting) return;
    const message = $('#assistant-message').value.trim();
    if (!message) return;
    if (message.length > 4000) return errorAt('#assistant-chat-error', '问题超过 4000 字符，请精简后再提交。');
    errorAt('#assistant-chat-error');
    const enableLLM = $('#assistant-enable-llm').checked && state.status?.llm_configured === true;
    const previousHistory = state.history.slice(-8);
    appendChat('user', message);
    $('#assistant-message').value = '';
    const pending = appendChat('assistant', enableLLM ? state.status?.llm_scope === 'local' ? '正在检索资料并请求本机模型生成…' : '正在检索资料并请求已配置的云模型…' : '正在检索你的学习资料，整理相关摘录…');
    pending.classList.add('study-chat-pending');
    setChatting(true);
    try {
      const result = await api('/api/assistant/chat', { method: 'POST', body: JSON.stringify({ message, enable_llm: enableLLM, use_agent: $('#assistant-use-agent').checked, history: previousHistory }) }, 90000);
      pending.remove();
      appendChat('assistant', result.reply || '服务未返回回答正文。', result);
      state.history.push({ role: 'user', content: message }, { role: 'assistant', content: String(result.reply || '') });
      state.history = state.history.slice(-8);
      $('#assistant-latest-trace').replaceChildren(buildTrace(result));
      const cloud = result.mode === 'llm' && result.llm_used === true;
      const actual = node('p', 'study-actual-mode', cloud ? '本次实际调用了模型服务。' : '本次使用离线检索摘录，未调用大语言模型。');
      $('#assistant-latest-trace').prepend(actual);
    } catch (error) {
      pending.remove();
      const failure = appendChat('assistant', '本次请求未完成：' + error.message);
      failure.classList.add('study-chat-failed');
      errorAt('#assistant-chat-error', error.message + ' 问题已放回输入框，可以编辑后重试。');
      $('#assistant-message').value = message;
      $('#assistant-latest-trace').replaceChildren(node('p', 'study-inline-error', '请求失败，服务未返回可展示的工具轨迹。'));
    } finally { setChatting(false); $('#assistant-message').focus(); }
  }

  function renderStructureReview(result) {
    state.latestOCR = result;
    const candidates = Array.isArray(result.structure_candidates) ? result.structure_candidates : [];
    const visible = result.recognition_mode === 'chinese' && (typeof result.raw_text === 'string' || candidates.length > 0);
    $('#structure-review').hidden = !visible;
    if (!visible) return;
    const raw = typeof result.raw_text === 'string' ? result.raw_text : String(result.text || '');
    $('#structure-raw-text').textContent = raw;
    $('#structure-corrected-text').value = state.corrected.get(positiveId(result.record_id)) || String(result.text || raw);
    $('#structure-save-status').textContent = '尚未确认校对；候选不自动覆盖原始识别。';
    $('#save-structure-correction').disabled = !positiveId(result.record_id);
    const list = $('#structure-candidate-list');
    list.replaceChildren();
    const original = node('button', 'structure-original-button', '保留原始识别作为校对起点');
    original.type = 'button';
    original.addEventListener('click', () => { $('#structure-corrected-text').value = raw; $('#structure-save-status').textContent = '已填入原始识别，请核对后再保存。'; });
    list.append(original);
    candidates.forEach((candidate, index) => {
      const candidateText = typeof candidate === 'string' ? candidate : candidate.text;
      if (typeof candidateText !== 'string' || !candidateText.trim()) return;
      const card = node('div', 'structure-candidate');
      card.append(node('small', '', `整字结构候选 ${index + 1}`), node('pre', '', candidateText));
      if (candidate.reason) card.append(node('p', '', candidate.reason));
      if (Number.isFinite(Number(candidate.support))) card.append(node('small', '', `${number(candidate.support)} 个视图支持${Number.isFinite(Number(candidate.confidence)) ? ' · 模型分数 ' + (Math.max(0, Math.min(1, Number(candidate.confidence))) * 100).toFixed(1) + '%（非正确率）' : ''}`));
      const choose = node('button', 'secondary-button', '采用候选，继续校对');
      choose.type = 'button';
      choose.addEventListener('click', () => { $('#structure-corrected-text').value = candidateText; $('#structure-save-status').textContent = '候选已填入编辑区，尚未保存。请核对全文后确认。'; $('#structure-corrected-text').focus(); });
      card.append(choose);
      list.append(card);
    });
  }
  async function saveStructureCorrection() {
    const result = state.latestOCR;
    const id = positiveId(result?.record_id), text = $('#structure-corrected-text').value.trim();
    if (!id || !text) { $('#structure-save-status').textContent = '请先填写核对后的完整文字。'; return; }
    const button = $('#save-structure-correction');
    button.disabled = true;
    $('#structure-save-status').textContent = '正在保存人工校对…';
    try {
      await api(`/api/history/${id}`, { method: 'PATCH', body: JSON.stringify({ corrected_text: text }) });
      state.corrected.set(id, text);
      if (state.latestOCR?.record_id === result.record_id) $('#structure-save-status').textContent = '人工校对已保存；原始模型结果独立保留。现在可整理到知识库。';
      notify('人工校对已保存，原始识别结果保留。', 'success');
    } catch (error) { if (state.latestOCR?.record_id === result.record_id) $('#structure-save-status').textContent = '校对保存失败：' + error.message; }
    finally { if (state.latestOCR?.record_id === result.record_id) button.disabled = false; }
  }

  $('#knowledge-file').addEventListener('change', event => { acceptKnowledgeFile(event.target.files?.[0]); event.target.value = ''; });
  $('#knowledge-file-form').addEventListener('submit', importFile);
  for (const name of ['dragenter', 'dragover']) $('#knowledge-drop-zone').addEventListener(name, event => { event.preventDefault(); if (!state.importing) $('#knowledge-drop-zone').classList.add('dragging'); });
  for (const name of ['dragleave', 'drop']) $('#knowledge-drop-zone').addEventListener(name, event => { event.preventDefault(); $('#knowledge-drop-zone').classList.remove('dragging'); });
  $('#knowledge-drop-zone').addEventListener('drop', event => { if (event.dataTransfer.files.length !== 1) return errorAt('#knowledge-import-error', '每次请选择一份学习资料。'); acceptKnowledgeFile(event.dataTransfer.files[0]); });
  $('#refresh-knowledge').addEventListener('click', refreshLibrary);
  $('#refresh-assistant-status').addEventListener('click', loadStatus);
  $('#add-text-document').addEventListener('click', () => openDraft());
  $('#knowledge-search-form').addEventListener('submit', searchKnowledge);
  $('#study-draft-form').addEventListener('submit', saveDraft);
  $('#study-draft-text').addEventListener('input', updateDraftCount);
  for (const selector of ['#close-study-draft', '#cancel-study-draft']) $(selector).addEventListener('click', () => { if (!state.draftBusy) $('#study-draft-dialog').close(); });
  $('#study-draft-dialog').addEventListener('cancel', event => { if (state.draftBusy) event.preventDefault(); });
  $('#close-study-document').addEventListener('click', () => $('#study-document-dialog').close());
  $('#study-document-dialog').addEventListener('close', () => { state.previewRequest++; state.previewId = null; });
  $('#study-document-ask').addEventListener('click', () => { const title = state.previewTitle; $('#study-document-dialog').close(); go('assistant'); $('#assistant-message').value = `请查找与“${title}”有关的资料，并概括其中的重点。`; $('#assistant-message').focus(); });
  $('#assistant-enable-llm').addEventListener('change', updateAssistantMode);
  $('#assistant-chat-form').addEventListener('submit', askAssistant);
  $('#assistant-message').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); askAssistant(event); } });
  $$('[data-study-prompt]').forEach(button => button.addEventListener('click', () => { $('#assistant-message').value = button.dataset.studyPrompt; $('#assistant-message').focus(); }));
  $('#clear-assistant-chat').addEventListener('click', () => { if (state.chatting) return; state.history = []; $('#assistant-chat-log').replaceChildren(node('p', 'study-cleared-chat', '本页对话已清空。学习资料仍保存在本机，可以继续提问。')); errorAt('#assistant-chat-error'); $('#assistant-latest-trace').replaceChildren(node('p', 'study-muted', '下一次提问后展示实际执行轨迹。')); });
  $('#save-structure-correction').addEventListener('click', saveStructureCorrection);
  document.addEventListener('inklab:knowledge-draft', event => openDraft(event.detail || {}));
  document.addEventListener('inklab:ocr-result', event => renderStructureReview(event.detail?.result || {}));
  document.addEventListener('inklab:correction-saved', event => { const id = positiveId(event.detail?.record_id); if (!id) return; if (typeof event.detail?.text === 'string' && event.detail.text.trim()) state.corrected.set(id, event.detail.text); else state.corrected.delete(id); });
  document.addEventListener('inklab:record-open', event => {
    const record = event.detail?.record;
    if (!record || !$('#feedback-form')) return;
    const button = node('button', 'secondary-button organize-record-button', '整理校对文字到知识库');
    button.type = 'button';
    button.addEventListener('click', () => openDraft({ record_id: record.id, text: $('#corrected-text')?.value.trim() || record.corrected_text || record.text || '', title: `手写学习笔记 #${record.id}`, source: 'correction' }));
    $('#feedback-form').append(button);
  });
  document.addEventListener('inklab:page', event => { if (event.detail?.page === 'knowledge') refreshLibrary(); if (event.detail?.page === 'assistant') loadStatus(); });
  // Never persist the online opt-in: a fresh page always starts offline.
  $('#assistant-enable-llm').checked = false;
  updateAssistantMode();
  refreshLibrary();
})();
