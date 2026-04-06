/**
 * Paramesh AI Support Widget
 * Drop-in chat bubble for any page. Uses Claude CLI via /api/support-chat.
 * Include with: <script src="/static/support-widget.js"></script>
 */
(function() {
  const STYLE = `
    #pw-bubble { position:fixed; bottom:24px; right:24px; z-index:9999;
      width:52px; height:52px; border-radius:50%; background:#22D3EE; color:#0A0A0B;
      display:flex; align-items:center; justify-content:center; cursor:pointer;
      font-size:24px; box-shadow:0 4px 20px rgba(34,211,238,0.3);
      transition:transform 0.2s, box-shadow 0.2s; }
    #pw-bubble:hover { transform:scale(1.08); box-shadow:0 6px 28px rgba(34,211,238,0.4); }
    #pw-panel { position:fixed; bottom:88px; right:24px; z-index:9999;
      width:340px; max-height:480px; background:#161618; border:1px solid #27272A;
      border-radius:14px; display:none; flex-direction:column; overflow:hidden;
      box-shadow:0 12px 40px rgba(0,0,0,0.5); font-family:'Outfit',system-ui,sans-serif; }
    #pw-panel.open { display:flex; }
    #pw-header { padding:14px 16px; border-bottom:1px solid #27272A;
      font-size:0.9rem; font-weight:600; color:#FAFAFA;
      display:flex; justify-content:space-between; align-items:center; }
    #pw-header button { background:none; border:none; color:#52525B; cursor:pointer; font-size:18px; }
    #pw-msgs { flex:1; overflow-y:auto; padding:12px; display:flex; flex-direction:column; gap:8px;
      min-height:200px; max-height:340px; }
    .pw-msg { max-width:85%; padding:8px 12px; border-radius:10px; font-size:0.85rem; line-height:1.5; word-wrap:break-word; }
    .pw-msg.bot { align-self:flex-start; background:#1C2128; color:#A1A1AA; border-bottom-left-radius:4px; }
    .pw-msg.user { align-self:flex-end; background:#22D3EE; color:#0A0A0B; border-bottom-right-radius:4px; }
    .pw-msg.typing { color:#52525B; font-size:0.8rem; font-style:italic; }
    #pw-input-row { padding:8px 12px; border-top:1px solid #27272A; display:flex; gap:8px; }
    #pw-input { flex:1; background:#0A0A0B; border:1px solid #27272A; border-radius:8px;
      padding:8px 12px; color:#FAFAFA; font-family:inherit; font-size:0.85rem; outline:none; }
    #pw-input:focus { border-color:#22D3EE; }
    #pw-send { background:#22D3EE; border:none; border-radius:8px; padding:8px 12px;
      cursor:pointer; color:#0A0A0B; font-size:14px; }
    @media(max-width:480px) { #pw-panel { right:8px; left:8px; width:auto; bottom:80px; } }
  `;

  // Inject styles
  const style = document.createElement('style');
  style.textContent = STYLE;
  document.head.appendChild(style);

  // Create bubble
  const bubble = document.createElement('div');
  bubble.id = 'pw-bubble';
  bubble.innerHTML = '?';
  bubble.title = 'Need help?';
  document.body.appendChild(bubble);

  // Create panel
  const panel = document.createElement('div');
  panel.id = 'pw-panel';
  panel.innerHTML = `
    <div id="pw-header">
      <span>Paramesh AI Support</span>
      <button onclick="document.getElementById('pw-panel').classList.remove('open')">&times;</button>
    </div>
    <div id="pw-msgs">
      <div class="pw-msg bot">Having an issue or question about Paramesh AI? I can help with pricing, supported formats, and troubleshooting.</div>
    </div>
    <div id="pw-input-row">
      <input id="pw-input" placeholder="Ask a question..." />
      <button id="pw-send">&#8593;</button>
    </div>
  `;
  document.body.appendChild(panel);

  // Toggle
  bubble.addEventListener('click', () => {
    panel.classList.toggle('open');
    if (panel.classList.contains('open')) document.getElementById('pw-input').focus();
  });

  // Rate limiting: 5 messages per session, 3 second cooldown
  const MAX_MESSAGES = 5;
  let messageCount = 0;
  let lastSendTime = 0;
  const COOLDOWN_MS = 3000;

  // Send
  async function send() {
    const input = document.getElementById('pw-input');
    const msgs = document.getElementById('pw-msgs');
    const text = input.value.trim();
    if (!text) return;

    // Rate limit checks
    const now = Date.now();
    if (now - lastSendTime < COOLDOWN_MS) {
      return; // silent cooldown
    }
    if (messageCount >= MAX_MESSAGES) {
      const limitMsg = document.createElement('div');
      limitMsg.className = 'pw-msg bot';
      limitMsg.textContent = 'You\'ve reached the message limit for this session. For more help, please visit our contact page at /contact.';
      msgs.appendChild(limitMsg);
      msgs.scrollTop = msgs.scrollHeight;
      input.value = '';
      return;
    }

    messageCount++;
    lastSendTime = now;

    // User message
    const userMsg = document.createElement('div');
    userMsg.className = 'pw-msg user';
    userMsg.textContent = text;
    msgs.appendChild(userMsg);
    input.value = '';
    msgs.scrollTop = msgs.scrollHeight;

    // Typing
    const typing = document.createElement('div');
    typing.className = 'pw-msg typing';
    typing.textContent = 'Thinking...';
    msgs.appendChild(typing);
    msgs.scrollTop = msgs.scrollHeight;

    try {
      const res = await fetch('/api/support-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      typing.remove();

      const botMsg = document.createElement('div');
      botMsg.className = 'pw-msg bot';
      botMsg.textContent = data.reply || 'Sorry, I could not process that.';
      msgs.appendChild(botMsg);
    } catch (e) {
      typing.remove();
      const errMsg = document.createElement('div');
      errMsg.className = 'pw-msg bot';
      errMsg.textContent = 'Connection error. Please try again.';
      msgs.appendChild(errMsg);
    }
    msgs.scrollTop = msgs.scrollHeight;
  }

  document.getElementById('pw-send').addEventListener('click', send);
  document.getElementById('pw-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); send(); }
  });
})();
