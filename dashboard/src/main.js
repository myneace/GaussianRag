import './style.css'

const API_BASE = 'http://localhost:8001';
const md = window.markdownit({
  html: true,
  linkify: true,
  typographer: true,
  highlight: function (str, lang) {
    if (lang && Prism.languages[lang]) {
      try {
        return Prism.highlight(str, Prism.languages[lang], lang);
      } catch (__) {}
    }
    return ''; // use external default escaping
  }
});

// State
let nodes = [];
let currentTab = 'manifold';
let activeThinkingId = null;

// DOM Elements
const tabs = document.querySelectorAll('.nav-item');
const views = document.querySelectorAll('.view');
const agentLog = document.getElementById('agent-log');
const nodeCountBadge = document.getElementById('node-count');
const viewTitle = document.getElementById('view-title');
const viewSubtitle = document.getElementById('view-subtitle');

// Initialize
init();

async function init() {
  setupTabs();
  setupSessions();
  setupUpload();
  setupChat();
  setupRealtime();
  await refreshNodes();
  renderPlot();
}

function setupTabs() {
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const tabName = tab.getAttribute('data-tab');
      switchTab(tabName);
    });
  });
}

function switchTab(tabName) {
  currentTab = tabName;
  
  // Update UI
  tabs.forEach(t => t.classList.toggle('active', t.getAttribute('data-tab') === tabName));
  views.forEach(v => v.classList.toggle('active', v.id === `${tabName}-view`));
  
  // Update header
  if (tabName === 'manifold') {
    viewTitle.textContent = 'Knowledge Manifold';
    viewSubtitle.textContent = '3D Gaussian Field Visualization';
    renderPlot();
  } else if (tabName === 'chat') {
    viewTitle.textContent = 'AI Knowledge Assistant';
    viewSubtitle.textContent = 'Gaussian RAG-powered Insights';
  } else if (tabName === 'upload') {
    viewTitle.textContent = 'Ingest Knowledge';
    viewSubtitle.textContent = 'Upload and Dissolve Documents';
  }
}

function addLog(message, type = 'system') {
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  const time = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  entry.innerHTML = `<span style="opacity: 0.5">[${time}]</span> ${message}`;
  agentLog.prepend(entry);
}

async function refreshNodes() {
  try {
    const response = await fetch(`${API_BASE}/api/nodes`);
    nodes = await response.json();
    nodeCountBadge.textContent = `${nodes.length} Nodes`;
  } catch (err) {
    console.error('Failed to fetch nodes:', err);
  }
}

function setupRealtime() {
  const evtSource = new EventSource(`${API_BASE}/api/events`);
  
  evtSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    const data = payload.data;

    switch (payload.type) {
      case 'ingestion_started':
        addLog(`Started ingestion for: ${data.document_id}`, 'system');
        break;
      case 'chunking_complete':
        addLog(`Document split into ${data.count} chunks.`, 'system');
        break;
      case 'ai_chunking_started':
        addLog(`AI Chunker Agent activated for ${data.count} chunks...`, 'system');
        break;
      case 'agent_proposition_extracted':
        addLog(`[Agent] Extracted ${data.count} propositions from chunk ${data.chunk_index}`, 'success');
        break;
      case 'entities_extracted':
        addLog(`[NLP] Extracted entities: ${data.entities.join(', ')}`, 'system');
        break;
      case 'node_created':
        addLog(`[Manifold] Created ${data.type} node: ${data.id.split('-').pop()}`, 'success');
        break;
      case 'ingestion_complete':
        addLog(`Ingestion complete! Total nodes: ${data.total_nodes}`, 'success');
        refreshNodes().then(() => {
          if (currentTab === 'manifold') renderPlot();
        });
        break;
      case 'node_added':
        addLog(`Node added: ${data.id}`, 'success');
        break;
      case 'query_received':
        addLog(`Query received: "${data.text.substring(0, 30)}..."`, 'system');
        break;
      case 'query_encoding':
        addLog(`Encoding query into Gaussian space...`, 'system');
        updateThinkingActivity('Encoding query into manifold...');
        break;
      case 'query_retrieval_started':
        addLog(`Retrieving top ${data.top_k} matching nodes...`, 'system');
        updateThinkingActivity(`Searching for top ${data.top_k} relevant nodes...`);
        break;
      case 'query_retrieval_complete':
        addLog(`Retrieved ${data.count} relevant nodes from manifold.`, 'success');
        updateThinkingActivity(`Retrieved ${data.count} nodes.`);
        break;
      case 'query_generation_started':
        addLog(`AI Agent generating response based on retrieved knowledge...`, 'system');
        updateThinkingActivity(`Synthesizing answer from knowledge...`);
        break;
      case 'query_generation_complete':
        addLog(`Response generated successfully.`, 'success');
        break;
      case 'system_log':
        addLog(data.message, 'system');
        break;
    }
  };
}

function updateThinkingActivity(text) {
  if (!activeThinkingId) return;
  const msg = document.getElementById(activeThinkingId);
  if (!msg) return;

  const activity = msg.querySelector('.thinking-activity');
  if (!activity) return;

  const step = document.createElement('div');
  step.className = 'activity-step active';
  step.textContent = text;
  
  // Mark previous steps as complete
  activity.querySelectorAll('.activity-step').forEach(s => s.classList.remove('active'));
  
  activity.appendChild(step);
  const chatMessages = document.getElementById('chat-messages');
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderPlot() {
  if (nodes.length === 0) {
    document.getElementById('plot').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#64748b">No nodes found. Upload a document to start.</div>';
    return;
  }

  const x = nodes.map(n => n.mu[0]);
  const y = nodes.map(n => n.mu[1]);
  const z = nodes.map(n => n.mu[2]);
  const sizes = nodes.map(n => Math.max(10, n.sigma * 800)); // Dynamic sizing based on variance
  const colors = nodes.map(n => {
    const type = n.metadata.node_type || 'anchor';
    return type === 'anchor' ? '#38bdf8' : '#818cf8';
  });
  const texts = nodes.map(n => `<b>${n.id}</b><br>${n.text.substring(0, 100)}...`);

  const scatter = {
    x, y, z,
    mode: 'markers',
    type: 'scatter3d',
    marker: {
      size: sizes,
      color: colors,
      opacity: 0.8,
      line: { color: 'rgba(255,255,255,0.2)', width: 0.5 }
    },
    text: texts,
    hoverinfo: 'text'
  };

  // Add connections if metadata has anchor_id
  const idToIdx = {};
  nodes.forEach((n, i) => idToIdx[n.id] = i);
  
  const lineX = [], lineY = [], lineZ = [];
  nodes.forEach((n, i) => {
    const anchorId = n.metadata.anchor_id;
    if (anchorId && idToIdx[anchorId] !== undefined) {
      const aIdx = idToIdx[anchorId];
      lineX.push(x[i], x[aIdx], null);
      lineY.push(y[i], y[aIdx], null);
      lineZ.push(z[i], z[aIdx], null);
    }
  });

  const filaments = {
    x: lineX, y: lineY, z: lineZ,
    mode: 'lines',
    type: 'scatter3d',
    line: { color: 'rgba(255,255,255,0.05)', width: 1 },
    hoverinfo: 'none'
  };

  const layout = {
    scene: {
      xaxis: { title: '', showgrid: true, zeroline: false, showticklabels: false, gridcolor: '#1e293b' },
      yaxis: { title: '', showgrid: true, zeroline: false, showticklabels: false, gridcolor: '#1e293b' },
      zaxis: { title: '', showgrid: true, zeroline: false, showticklabels: false, gridcolor: '#1e293b' },
      camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } },
      bgcolor: '#020617'
    },
    paper_bgcolor: '#020617',
    margin: { l: 0, r: 0, b: 0, t: 0 },
    showlegend: false,
    hovermode: 'closest'
  };

  Plotly.react('plot', [filaments, scatter], layout);
}

function setupUpload() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const selectFileBtn = document.getElementById('select-file-btn');
  const progress = document.getElementById('ingest-progress');
  const progressFill = document.querySelector('.progress-fill');

  selectFileBtn.onclick = () => fileInput.click();

  fileInput.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    await uploadFile(file);
  };

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    progress.classList.remove('hidden');
    progressFill.style.width = '30%';
    addLog(`Ingesting ${file.name}...`);

    try {
      const response = await fetch(`${API_BASE}/api/ingest`, {
        method: 'POST',
        body: formData
      });
      const result = await response.json();
      progressFill.style.width = '100%';
      addLog(`Successfully ingested ${result.nodes_added} nodes!`, 'success');
      
      setTimeout(async () => {
        progress.classList.add('hidden');
        progressFill.style.width = '0%';
        await setupSessions();
        switchTab('manifold');
      }, 1000);
  } catch (err) {
      console.error('Upload failed:', err);
      addLog(`Error: Ingestion failed.`, 'error');
      progress.classList.add('hidden');
    }
  }
}

function setupChat() {
  const chatInput = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const chatMessages = document.getElementById('chat-messages');

  const sendMessage = async () => {
    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = '';
    appendMessage('user', text);
    
    // Create new AI bubble with thinking area
    activeThinkingId = 'ai-' + Date.now();
    const aiBubble = createAIBubble(activeThinkingId);

    try {
      const response = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await response.json();
      
      // Finalize the bubble with answer and references
      finalizeAIBubble(activeThinkingId, data.answer, data.retrieved);
      activeThinkingId = null;
    } catch (err) {
      console.error('Chat failed:', err);
      finalizeAIBubble(activeThinkingId, 'Sorry, I encountered an error while processing your request.', []);
      activeThinkingId = null;
    }
  };

  function createAIBubble(id) {
    const msg = document.createElement('div');
    msg.className = `message ai thinking active-thinking`;
    msg.id = id;
    msg.innerHTML = `
      <div class="message-content">
        <div class="thinking-header">
          <span>Thinking Process</span>
          <button class="toggle-thinking">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
          </button>
        </div>
        <div class="thinking-activity">
          <div class="activity-step active">Analyzing query...</div>
        </div>
        <div class="answer-text hidden"></div>
      </div>
    `;
    
    const toggle = msg.querySelector('.toggle-thinking');
    const header = msg.querySelector('.thinking-header');
    const toggleAction = (e) => {
      if (e) e.stopPropagation();
      const activity = msg.querySelector('.thinking-activity');
      activity.classList.toggle('collapsed');
      toggle.classList.toggle('rotated');
    };
    
    header.onclick = toggleAction;
    toggle.onclick = toggleAction;
    
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
  }

  function finalizeAIBubble(id, answer, retrieved) {
    const msg = document.getElementById(id);
    if (!msg) return;

    msg.classList.remove('active-thinking');
    const content = msg.querySelector('.message-content');
    const answerEl = msg.querySelector('.answer-text');
    
    // Render Markdown
    answerEl.innerHTML = md.render(answer);
    answerEl.classList.remove('hidden');
    
    // Highlight code blocks
    Prism.highlightAllUnder(msg);

    // Add Copy Button
    addCopyButton(msg, answer);

    // Create and render references locally for this bubble
    if (retrieved && retrieved.length > 0) {
      const refPanel = document.createElement('div');
      refPanel.className = 'references-panel';
      refPanel.innerHTML = `
        <div class="status-header">
          <span>Retrieved References</span>
          <button class="toggle-refs">
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
          </button>
        </div>
        <div class="references-list"></div>
      `;
      
      const refList = refPanel.querySelector('.references-list');
      const toggleBtn = refPanel.querySelector('.toggle-refs');
      const header = refPanel.querySelector('.status-header');
      
      const toggleRefs = (e) => {
        if (e) e.stopPropagation();
        refList.classList.toggle('collapsed');
        toggleBtn.classList.toggle('rotated');
      };
      
      header.onclick = toggleRefs;
      toggleBtn.onclick = toggleRefs;
      
      retrieved.forEach(ref => {
        const item = document.createElement('div');
        item.className = 'reference-item';
        item.innerHTML = `
          <div class="ref-id">${ref.id} <span style="opacity:0.5;font-weight:normal">(conf: ${ref.confidence.toFixed(2)})</span></div>
          <div class="ref-text">${ref.text.substring(0, 150)}...</div>
        `;
        refList.appendChild(item);
      });
      
      content.appendChild(refPanel);
      
      // Automatically collapse after a short delay
      setTimeout(() => {
        if (!refList.classList.contains('collapsed')) {
          refList.classList.add('collapsed');
          toggleBtn.classList.add('rotated');
        }
      }, 1500);
    }

    // Automatically collapse thinking after a short delay
    setTimeout(() => {
      const activity = msg.querySelector('.thinking-activity');
      const toggle = msg.querySelector('.toggle-thinking');
      if (activity && !activity.classList.contains('collapsed')) {
        activity.classList.add('collapsed');
        toggle.classList.add('rotated');
      }
    }, 1500);
  }

  window.chatHelpers = {
    appendMessage,
    createAIBubble,
    finalizeAIBubble,
    addCopyButton,
    clearChat: () => {
      chatMessages.innerHTML = '';
      appendMessage('ai', "Hello! I'm your Gaussian RAG assistant. Ask me anything about the ingested knowledge.");
    }
  };

  sendBtn.onclick = sendMessage;
  chatInput.onkeydown = (e) => { if (e.key === 'Enter') sendMessage(); };

  function appendMessage(role, text, extraClass = '', id = null) {
    const msg = document.createElement('div');
    msg.className = `message ${role} ${extraClass}`;
    if (id) msg.id = id;
    msg.innerHTML = `<div class="message-content">${text}</div>`;
    
    if (role !== 'ai' || !extraClass.includes('thinking')) {
      addCopyButton(msg, text);
    }

    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msg;
  }

  function addCopyButton(msg, text) {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.title = 'Copy to clipboard';
    btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
    
    btn.onclick = async () => {
      try {
        await navigator.clipboard.writeText(text);
        btn.classList.add('copied');
        btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;
        setTimeout(() => {
          btn.classList.remove('copied');
          btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
        }, 2000);
      } catch (err) {
        console.error('Failed to copy:', err);
      }
    };
    
    msg.appendChild(btn);
  }
}

async function setupSessions() {
  const sessionList = document.getElementById('session-list');
  try {
    const response = await fetch(`${API_BASE}/api/sessions`);
    const manifolds = await response.json();
    
    sessionList.innerHTML = '';
    manifolds.forEach(manifold => {
      const group = document.createElement('div');
      group.className = 'manifold-group';
      
      // Prettify manifold name
      let prettyName = manifold.name.replace(/_\d+$/, '').replace(/_/g, ' ');
      prettyName = prettyName.charAt(0).toUpperCase() + prettyName.slice(1);
      
      const header = document.createElement('div');
      header.className = 'manifold-header';
      header.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22h14a2 2 0 0 0 2-2V7.5L14.5 2H6a2 2 0 0 0-2 2v4"/></svg>
        <span>${prettyName}</span>
      `;
      group.appendChild(header);
      
      const chatsContainer = document.createElement('div');
      chatsContainer.className = 'chats-container';
      
      manifold.chats.forEach((chat, index) => {
        const item = document.createElement('div');
        item.className = `session-item ${chat.active ? 'active' : ''}`;
        
        const date = new Date(chat.timestamp * 1000);
        const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        item.innerHTML = `<span>Chat ${manifold.chats.length - index}</span> <span style="font-size: 0.7rem; opacity: 0.5;">${timeStr}</span>`;
        item.onclick = async () => {
          await switchSession(chat.id);
          const chatTab = document.querySelector('[data-tab="chat"]');
          if (chatTab) chatTab.click();
        };
        chatsContainer.appendChild(item);
      });
      
      group.appendChild(chatsContainer);
      sessionList.appendChild(group);
    });
  } catch (err) {
    console.error('Failed to fetch sessions:', err);
  }

  // Handle New Chat button
  const newChatBtn = document.getElementById('new-chat-btn');
  if (newChatBtn) {
    newChatBtn.onclick = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/sessions/new_chat`, { method: 'POST' });
        if (response.ok) {
          const data = await response.json();
          await switchSession(data.session_id);
          const chatTab = document.querySelector('[data-tab="chat"]');
          if (chatTab) chatTab.click();
        } else {
          // If it fails (e.g., no active session), just go to ingest
          const uploadTab = document.querySelector('[data-tab="upload"]');
          if (uploadTab) uploadTab.click();
        }
      } catch (err) {
        console.error('Failed to create new chat:', err);
        const uploadTab = document.querySelector('[data-tab="upload"]');
        if (uploadTab) uploadTab.click();
      }
    };
  }
}

async function switchSession(sessionId) {
  try {
    await fetch(`${API_BASE}/api/sessions/switch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    });
    addLog(`Switched to session: ${sessionId}`);
    await setupSessions();
    await refreshNodes();
    renderPlot();
    await loadChatHistory(sessionId);
  } catch (err) {
    console.error('Failed to switch session:', err);
  }
}

async function loadChatHistory(sessionId) {
  if (!window.chatHelpers) return;
  try {
    const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/history`);
    const history = await response.json();
    
    window.chatHelpers.clearChat();
    
    for (const item of history) {
      window.chatHelpers.appendMessage('user', item.user);
      const aiId = 'ai-hist-' + Math.random().toString(36).substr(2, 9);
      window.chatHelpers.createAIBubble(aiId);
      // Immediately finalize history items without the "Thinking" delay
      const msg = document.getElementById(aiId);
      if (msg) {
        msg.classList.remove('active-thinking');
        const content = msg.querySelector('.message-content');
        const answerEl = msg.querySelector('.answer-text');
        answerEl.innerHTML = md.render(item.ai);
        answerEl.classList.remove('hidden');
        
        // Hide thinking process for history
        const header = msg.querySelector('.thinking-header');
        const activity = msg.querySelector('.thinking-activity');
        if (header) header.style.display = 'none';
        if (activity) activity.style.display = 'none';

        // Add references if any
        if (item.retrieved && item.retrieved.length > 0) {
          const refPanel = document.createElement('div');
          refPanel.className = 'references-panel';
          refPanel.innerHTML = `
            <div class="status-header">
              <span>Retrieved References</span>
              <button class="toggle-refs rotated">
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
              </button>
            </div>
            <div class="references-list collapsed"></div>
          `;
          
          const refList = refPanel.querySelector('.references-list');
          const toggleBtn = refPanel.querySelector('.toggle-refs');
          const refHeader = refPanel.querySelector('.status-header');
          
          const toggleRefs = (e) => {
            if (e) e.stopPropagation();
            refList.classList.toggle('collapsed');
            toggleBtn.classList.toggle('rotated');
          };
          
          refHeader.onclick = toggleRefs;
          toggleBtn.onclick = toggleRefs;
          
          item.retrieved.forEach(ref => {
            const rItem = document.createElement('div');
            rItem.className = 'reference-item';
            rItem.innerHTML = `
              <div class="ref-id">${ref.id} <span style="opacity:0.5;font-weight:normal">(conf: ${ref.confidence.toFixed(2)})</span></div>
              <div class="ref-text">${ref.text.substring(0, 150)}...</div>
            `;
            refList.appendChild(rItem);
          });
          content.appendChild(refPanel);
        }
        
        // Add copy button
        window.chatHelpers.addCopyButton(msg, item.ai);
      }
    }
    
    // Auto scroll to bottom after loading history
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;

  } catch (err) {
    console.error('Failed to load chat history:', err);
  }
}
