"""
NTIS 전문가 추천 시스템의 기능을 웹 브라우저에서 직접 테스트할 수 있도록 제공하는
대화형 플레이그라운드(Playground) HTML 템플릿 모듈입니다.
시스템 상태 확인 및 /recommend, /search/candidates 엔드포인트 호출 기능을 포함합니다.
"""

from __future__ import annotations

from textwrap import dedent


PLAYGROUND_HTML = dedent(
    """\
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>NTIS 전문가 추천 시스템 - 테스트 베드</title>
      <link rel="stylesheet" as="style" crossorigin href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />
      <style>
        :root {
          color-scheme: light dark;
          --bg: #0f172a;
          --panel: rgba(30, 41, 59, 0.7);
          --panel-border: rgba(255, 255, 255, 0.1);
          --accent: #10b981;
          --accent-glow: rgba(16, 185, 129, 0.2);
          --text-main: #f8fafc;
          --text-muted: #94a3b8;
          --user-msg: #334155;
          --assistant-msg: rgba(30, 41, 59, 0.8);
          --error: #ef4444;
          --ok: #10b981;
          --warn: #f59e0b;
          --shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
          --font: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, system-ui, Roboto, "Helvetica Neue", "Segoe UI", "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", sans-serif;
        }

        @media (prefers-color-scheme: light) {
          :root {
            --bg: #f8fafc;
            --panel: rgba(255, 255, 255, 0.8);
            --panel-border: rgba(0, 0, 0, 0.05);
            --text-main: #1e293b;
            --text-muted: #64748b;
            --user-msg: #e2e8f0;
            --assistant-msg: #ffffff;
            --shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
          }
        }

        * { box-sizing: border-box; }
        body {
          margin: 0;
          min-height: 100vh;
          font-family: var(--font);
          color: var(--text-main);
          background-color: var(--bg);
          background-image: 
            radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%);
          line-height: 1.6;
          -webkit-font-smoothing: antialiased;
        }

        .container {
          max-width: 1200px;
          margin: 2rem auto;
          padding: 0 1.5rem;
          display: grid;
          gap: 2rem;
        }

        header {
          text-align: center;
          margin-bottom: 1rem;
          animation: fadeInDown 0.8s ease-out;
        }

        .badge {
          display: inline-block;
          padding: 0.25rem 0.75rem;
          background: var(--accent-glow);
          color: var(--accent);
          border-radius: 9999px;
          font-size: 0.875rem;
          font-weight: 600;
          margin-bottom: 1rem;
        }

        h1 {
          font-size: 2.5rem;
          font-weight: 800;
          margin: 0;
          letter-spacing: -0.025em;
          background: linear-gradient(to right, #10b981, #3b82f6);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        p.desc {
          color: var(--text-muted);
          font-size: 1.125rem;
          margin-top: 0.5rem;
        }

        .glass-panel {
          background: var(--panel);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          border: 1px solid var(--panel-border);
          border-radius: 1.5rem;
          box-shadow: var(--shadow);
          padding: 2rem;
          transition: transform 0.3s ease;
        }

        .status-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 1rem 1.5rem;
          background: rgba(0, 0, 0, 0.05);
          border-radius: 100px;
          margin-bottom: 2rem;
        }

        .status-indicator {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .dot {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background: var(--text-muted);
        }

        .dot.ok { background: var(--ok); box-shadow: 0 0 12px var(--ok); }
        .dot.error { background: var(--error); box-shadow: 0 0 12px var(--error); }
        .dot.loading { background: var(--warn); animation: pulse 1.5s infinite; }

        .app-layout {
          display: grid;
          grid-template-columns: 400px 1fr;
          gap: 2rem;
          align-items: start;
        }

        @media (max-width: 1024px) {
          .app-layout { grid-template-columns: 1fr; }
        }

        .form-section {
          position: sticky;
          top: 2rem;
        }

        .input-group {
          margin-bottom: 1.5rem;
        }

        label {
          display: block;
          font-weight: 600;
          margin-bottom: 0.5rem;
          font-size: 0.9rem;
          color: var(--text-muted);
        }

        textarea, input, select {
          width: 100%;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid var(--panel-border);
          border-radius: 0.75rem;
          padding: 0.75rem 1rem;
          color: var(--text-main);
          font-family: inherit;
          font-size: 1rem;
          transition: border-color 0.2s, box-shadow 0.2s;
        }

        textarea:focus, input:focus {
          outline: none;
          border-color: var(--accent);
          box-shadow: 0 0 0 3px var(--accent-glow);
        }

        textarea { min-height: 120px; resize: vertical; }

        .endpoint-selector {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1.5rem;
        }

        .btn-toggle {
          flex: 1;
          padding: 0.75rem;
          border-radius: 0.75rem;
          border: 1px solid var(--panel-border);
          background: transparent;
          color: var(--text-muted);
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }

        .btn-toggle.active {
          background: var(--accent);
          color: white;
          border-color: var(--accent);
        }

        .btn-primary {
          width: 100%;
          padding: 1rem;
          background: linear-gradient(135deg, #10b981, #059669);
          color: white;
          border: none;
          border-radius: 1rem;
          font-weight: 700;
          font-size: 1.1rem;
          cursor: pointer;
          transition: transform 0.2s, box-shadow 0.2s;
          box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.4);
        }

        .btn-primary:hover {
          transform: translateY(-2px);
          box-shadow: 0 20px 25px -5px rgba(16, 185, 129, 0.5);
        }

        .btn-primary:disabled {
          opacity: 0.6;
          cursor: not-allowed;
          transform: none;
        }

        .chat-container {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
          min-height: 600px;
        }

        .msg {
          padding: 1.25rem 1.5rem;
          border-radius: 1.25rem;
          max-width: 85%;
          position: relative;
          animation: slideInUp 0.4s ease-out;
        }

        .msg.user {
          align-self: flex-end;
          background: var(--user-msg);
          border-bottom-right-radius: 0.25rem;
        }

        .msg.assistant {
          align-self: flex-start;
          background: var(--assistant-msg);
          border-bottom-left-radius: 0.25rem;
          border: 1px solid var(--panel-border);
        }

        .msg .role {
          font-size: 0.75rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          margin-bottom: 0.5rem;
          opacity: 0.6;
        }

        .expert-card {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid var(--panel-border);
          border-radius: 1rem;
          padding: 1.25rem;
          margin-top: 1rem;
        }

        .expert-card h4 {
          margin: 0 0 0.5rem 0;
          color: var(--accent);
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .stat-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
          gap: 0.75rem;
          margin: 1rem 0;
        }

        .stat-box {
          background: rgba(0, 0, 0, 0.2);
          padding: 0.5rem;
          border-radius: 0.5rem;
          text-align: center;
        }

        .stat-box .label { font-size: 0.7rem; color: var(--text-muted); display: block; }
        .stat-box .value { font-weight: 800; font-size: 1rem; }

        .badge-list {
          display: flex;
          flex-wrap: wrap;
          gap: 0.4rem;
          margin-top: 0.5rem;
        }

        .tag {
          padding: 2px 8px;
          border-radius: 4px;
          font-size: 0.75rem;
          background: rgba(16, 185, 129, 0.1);
          color: #10b981;
        }

        details {
          background: rgba(0, 0, 0, 0.1);
          border-radius: 1rem;
          padding: 0.75rem;
          margin-top: 1rem;
        }

        summary {
          cursor: pointer;
          font-weight: 600;
          font-size: 0.9rem;
          color: var(--text-muted);
        }

        pre {
          background: #000;
          color: #10b981;
          padding: 1rem;
          border-radius: 0.5rem;
          overflow: auto;
          font-size: 0.85rem;
          margin-top: 0.5rem;
        }

        @keyframes fadeInDown {
          from { opacity: 0; transform: translateY(-20px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideInUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulse {
          0% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.2); opacity: 0.6; }
          100% { transform: scale(1); opacity: 1; }
        }

        .loading-dots:after {
          content: ' .';
          animation: dots 1.5s steps(5, end) infinite;
        }

        @keyframes dots {
          0%, 20% { content: ' .'; }
          40% { content: ' ..'; }
          60% { content: ' ...'; }
          80%, 100% { content: ''; }
        }

        /* 로그 콘솔 스타일 */
        .log-console {
          margin-top: 2rem;
          background: #000;
          color: #d1d5db;
          border-radius: 1rem;
          padding: 1.5rem;
          font-family: 'Fira Code', 'Courier New', Courier, monospace;
          font-size: 0.85rem;
          box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
          max-height: 400px;
          overflow-y: auto;
          border: 1px solid var(--panel-border);
        }
        .log-console .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
          color: var(--accent);
          font-weight: 700;
          border-bottom: 1px solid #333;
          padding-bottom: 0.5rem;
        }
        .log-line {
          margin-bottom: 2px;
          white-space: pre-wrap;
          word-break: break-all;
          line-height: 1.4;
        }
        .log-INFO { color: #10b981; }
        .log-DEBUG { color: #94a3b8; }
        .log-WARNING { color: #f59e0b; }
        .log-ERROR { color: #ef4444; }
        .log-CRITICAL { color: #ef4444; font-weight: bold; background: rgba(239, 68, 68, 0.1); }
      </style>
    </head>
    <body>
      <div class="container">
        <header>
          <div class="badge">Experimental</div>
          <h1>NTIS 전문가 AI 추천 시스템</h1>
          <p class="desc">준비 상태를 확인하고 전문가 추천 결과를 즉시 테스트해보세요.</p>
        </header>

        <section class="glass-panel status-bar">
          <div class="status-indicator">
            <div class="dot" id="statusDot"></div>
            <div>
              <strong id="statusTitle">상태 확인 중...</strong>
              <div id="statusText" style="font-size: 0.8rem; color: var(--text-muted)">시스템 모듈을 연결 중입니다.</div>
            </div>
          </div>
          <button type="button" class="btn-toggle" id="refreshStatusButton" style="width: auto; padding: 0.5rem 1rem;">새로고침</button>
        </section>

        <section class="glass-panel" style="padding: 1.25rem 1.5rem;">
          <details>
            <summary>Readiness details</summary>
            <pre id="readinessDetails">{}</pre>
          </details>
        </section>

        <main class="app-layout">
          <section class="glass-panel form-section">
            <div class="endpoint-selector">
              <button type="button" class="btn-toggle active" data-endpoint="/recommend">추천 전문가</button>
              <button type="button" class="btn-toggle" data-endpoint="/search/candidates">후보자 목록</button>
            </div>

            <form id="chatForm">
              <div class="input-group">
                <label for="queryInput">자연어 질의</label>
                <textarea id="queryInput" required placeholder="예: 인공지능 반도체 분야에서 최근 국책 과제 수행 경험이 있는 박사급 전문가를 추천해줘."></textarea>
              </div>

              <div class="input-group">
                <label for="topKInput">추천 인원 수 (top_k)</label>
                <input id="topKInput" type="number" min="1" max="5" placeholder="기본값 사용">
              </div>

              <details>
                <summary>고급 필터 설정</summary>
                <div style="margin-top: 1rem">
                  <div class="input-group">
                    <label for="filtersInput">필터 재정의 (JSON 형식)</label>
                    <textarea id="filtersInput" placeholder='{&#10;  "degree_slct_nm": "박사",&#10;  "art_sci_slct_nm": "SCIE"&#10;}'></textarea>
                  </div>
                  <div class="input-group">
                    <label for="excludeInput">제외할 기관</label>
                    <textarea id="excludeInput" placeholder="줄바꿈 또는 쉼표로 구분하여 입력"></textarea>
                  </div>
                </div>
              </details>

              <div style="margin-top: 2rem; display: flex; gap: 1rem">
                <button type="submit" class="btn-primary" id="submitButton">쿼리 실행</button>
                <button type="button" class="btn-toggle" id="clearButton" style="flex: 0 0 auto;">초기화</button>
              </div>
            </form>
          </section>

          <section class="glass-panel chat-container">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--panel-border); padding-bottom: 1rem;">
              <h3 style="margin:0">분석 결과 대화창</h3>
              <div class="badge" id="modeChip" style="margin:0">/recommend</div>
            </div>
            <div id="chatLog" style="flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 1rem;"></div>
            
            <!-- 서버 로그 콘솔 영역 -->
            <div class="log-console" id="logConsole" style="display: none;">
              <div class="header">
                <span>📟 SERVER REAL-TIME LOGS</span>
                <button type="button" class="btn-toggle" style="padding: 2px 8px; font-size: 0.7rem;" onclick="document.getElementById('logContent').innerHTML=''">Clear</button>
              </div>
              <div id="logContent"></div>
            </div>
          </section>
        </main>
      </div>

      <script>
        const state = { endpoint: '/recommend', loading: false };
        const $ = (id) => document.getElementById(id);
        const chatLog = $('chatLog');
        const submitButton = $('submitButton');
        const modeChip = $('modeChip');
        const statusDot = $('statusDot');
        const statusTitle = $('statusTitle');
        const statusText = $('statusText');
        const readinessDetails = $('readinessDetails');
        const modeButtons = [...document.querySelectorAll('.endpoint-selector .btn-toggle')];

        const escapeHtml = (val) => String(val ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        
        function pushMessage(role, html, isError = false) {
          const div = document.createElement('div');
          div.className = `msg ${role}`;
          if (isError) div.style.borderColor = 'var(--error)';
          
          const label = role === 'user' ? '사용자 질의' : 'AI 분석 결과';
          div.innerHTML = `<div class="role">${label}</div>${html}`;
          
          chatLog.appendChild(div);
          chatLog.scrollTop = chatLog.scrollHeight;
        }

        function clearLog() {
          chatLog.innerHTML = '';
          pushMessage('assistant', '<p>검색할 내용을 입력하신 후 <b>쿼리 실행</b> 버튼을 눌러주세요.</p>');
        }

        async function updateStatus() {
          statusDot.className = 'dot loading';
          statusTitle.textContent = '확인 중...';
          try {
            const res = await fetch('/health/ready');
            const data = await res.json();
            readinessDetails.textContent = JSON.stringify(data, null, 2);
            if (res.ok && data.ready) {
              statusDot.className = 'dot ok';
              statusTitle.textContent = '시스템 정상';
              statusText.textContent = `${data.collection_name} 데이터베이스가 연결되었습니다.`;
            } else {
              statusDot.className = 'dot error';
              statusTitle.textContent = '시스템 오류';
              statusText.textContent = data.issues?.join(', ') || '준비 상태 확인 실패';
            }
          } catch (e) {
            statusDot.className = 'dot error';
            statusTitle.textContent = '연결 불가';
            statusText.textContent = '서버에 연결할 수 없습니다.';
            readinessDetails.textContent = JSON.stringify({ error: String(e) }, null, 2);
          }
        }

        function setMode(ep) {
          state.endpoint = ep;
          modeChip.textContent = ep;
          modeButtons.forEach(b => b.classList.toggle('active', b.dataset.endpoint === ep));
        }

        async function handleSearch(e) {
          e.preventDefault();
          if (state.loading) return;

          const query = $('queryInput').value.trim();
          if (!query) return;

          const payload = {
            query,
            top_k: parseInt($('topKInput').value) || undefined,
            filters_override: $('filtersInput').value ? JSON.parse($('filtersInput').value) : {},
            exclude_orgs: $('excludeInput').value.split(/[\\n,]/).map(s => s.trim()).filter(Boolean)
          };

          pushMessage('user', `<p>${escapeHtml(query)}</p>`);
          state.loading = true;
          submitButton.disabled = true;
          submitButton.innerHTML = '<span class="loading-dots">분석 중</span>';

          try {
            const res = await fetch(state.endpoint, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (!res.ok) throw new Error(data.detail || '요청 처리 중 오류가 발생했습니다.');
            
            let html = '';
            if (state.endpoint === '/recommend') {
              html = `
                <p><b>분석 요약:</b> ${escapeHtml(data.intent_summary)}</p>
                <div class="stat-grid">
                  <div class="stat-box"><span class="label">검색 대상</span><span class="value">${data.retrieved_count}명</span></div>
                  <div class="stat-box"><span class="label">최종 추천</span><span class="value">${data.recommendations.length}명</span></div>
                </div>
                ${data.recommendations.map(r => `
                  <div class="expert-card">
                    <h4>#${r.rank} ${escapeHtml(r.name)} <span style="font-size: 0.8rem; color: var(--text-muted); font-weight: 500; margin-left: 0.5rem;">${escapeHtml(r.organization || '소속 미상')}</span> <span class="tag" style="margin-left: auto;">${r.fit}</span></h4>
                    <div style="font-size: 0.9rem; margin-bottom: 0.5rem"><b>추천 사유:</b> <ul>${r.reasons.map(v => `<li>${escapeHtml(v)}</li>`).join('')}</ul></div>
                    <details>
                      <summary>수행 증거 및 실적</summary>
                      <ul style="font-size: 0.85rem; margin-top: 0.5rem">
                        ${r.evidence.map(ev => `<li>[${ev.type}] ${escapeHtml(ev.title)} (${ev.date})</li>`).join('')}
                      </ul>
                    </details>
                  </div>
                `).join('')}
              `;
            } else {
              html = `
                <p><b>분석 요약:</b> ${escapeHtml(data.intent_summary)}</p>
                <p>검색된 후보자 ${data.candidates.length}명의 목록입니다.</p>
                ${data.candidates.map(c => `
                  <div class="expert-card">
                    <h4>${escapeHtml(c.name)} <span style="font-size: 0.8rem; color: var(--text-muted)">${escapeHtml(c.organization)}</span></h4>
                    <div class="stat-grid">
                      <div class="stat-box"><span class="label">논문</span><span class="value">${c.counts.article_cnt}</span></div>
                      <div class="stat-box"><span class="label">특허</span><span class="value">${c.counts.patent_cnt}</span></div>
                      <div class="stat-box"><span class="label">과제</span><span class="value">${c.counts.project_cnt}</span></div>
                      <div class="stat-box"><span class="label">점수</span><span class="value">${c.shortlist_score.toFixed(2)}</span></div>
                    </div>
                  </div>
                `).join('')}
              `;
            }
            pushMessage('assistant', html);

            // 서버 로그 출력 로직 추가
            if (data.trace && data.trace.server_logs) {
              const logConsole = $('logConsole');
              const logContent = $('logContent');
              logConsole.style.display = 'block';
              
              data.trace.server_logs.forEach(msg => {
                const line = document.createElement('div');
                line.className = 'log-line';
                
                // 레벨에 따른 색상 결정
                if (msg.includes('[INFO]')) line.classList.add('log-INFO');
                else if (msg.includes('[DEBUG]')) line.classList.add('log-DEBUG');
                else if (msg.includes('[WARNING]')) line.classList.add('log-WARNING');
                else if (msg.includes('[ERROR]')) line.classList.add('log-ERROR');
                else if (msg.includes('[CRITICAL]')) line.classList.add('log-CRITICAL');
                
                line.textContent = msg;
                logContent.appendChild(line);
              });
              logConsole.scrollTop = logConsole.scrollHeight;
            }
          } catch (err) {
            pushMessage('assistant', `<p style="color: var(--error)"><b>오류 발생:</b> ${escapeHtml(err.message)}</p>`, true);
          } finally {
            state.loading = false;
            submitButton.disabled = false;
            submitButton.textContent = '쿼리 실행';
          }
        }

        modeButtons.forEach(b => b.addEventListener('click', () => setMode(b.dataset.endpoint)));
        $('chatForm').addEventListener('submit', handleSearch);
        $('clearButton').addEventListener('click', clearLog);
        $('refreshStatusButton').addEventListener('click', updateStatus);
        $('queryInput').addEventListener('keydown', e => {
          if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') $('chatForm').requestSubmit();
        });

        clearLog();
        updateStatus();
      </script>
    </body>
    </html>
    """
)
