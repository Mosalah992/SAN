/* ─── Boot Sequence Animation ────────────────────────────── */

const BOOT_CHECKS = [
  { label: 'loading security modules',      delay: 180, status: 'ok' },
  { label: 'initializing analytical baselines', delay: 220, status: 'ok' },
  { label: 'mounting SEIR epidemic model',  delay: 190, status: 'ok' },
  { label: 'connecting to agent state',     delay: 260, status: 'dynamic', key: 'agent' },
  { label: 'establishing event feeds',      delay: 200, status: 'ok' },
  { label: 'calibrating baseline alignment', delay: 170, status: 'ok' },
  { label: 'starting WebSocket channel',    delay: 210, status: 'dynamic', key: 'ws' },
];

function pad(s, len) {
  return s + '.'.repeat(Math.max(0, len - s.length));
}

export async function runBoot(dynamicResults = {}) {
  return new Promise((resolve) => {
    const screen = document.getElementById('boot-screen');
    const linesEl = document.getElementById('boot-lines');
    if (!screen || !linesEl) { resolve(); return; }

    let i = 0;
    let elapsed = 0;

    function addLine() {
      if (i >= BOOT_CHECKS.length) {
        // All done — add READY line then fade out
        setTimeout(() => {
          const ready = document.createElement('div');
          ready.className = 'boot-line';
          ready.style.marginTop = '16px';
          ready.innerHTML = `
            <span style="color:var(--cyan);font-size:13px;letter-spacing:.1em">
              SYSTEM READY<span class="boot-cursor"></span>
            </span>`;
          ready.style.animationDelay = '0ms';
          linesEl.appendChild(ready);
        }, 100);

        setTimeout(() => {
          screen.classList.add('fading');
          setTimeout(resolve, 420);
        }, 700);
        return;
      }

      const check = BOOT_CHECKS[i];
      elapsed += check.delay;

      setTimeout(() => {
        const line = document.createElement('div');
        line.className = 'boot-line';
        line.style.animationDelay = '0ms';

        let statusText, statusClass;
        if (check.status === 'dynamic') {
          const result = dynamicResults[check.key];
          statusText  = result === false ? 'WARN' : 'OK';
          statusClass = result === false ? 'bl-warn' : 'bl-ok';
        } else {
          statusText  = check.status === 'ok' ? 'OK' : 'FAIL';
          statusClass = check.status === 'ok' ? 'bl-ok' : 'bl-fail';
        }

        line.innerHTML = `
          <span class="bl-label">&gt; ${pad(check.label, 42)}</span>
          <span class="bl-status ${statusClass}">${statusText}</span>`;
        linesEl.appendChild(line);

        i++;
        addLine();
      }, elapsed);
    }

    addLine();
  });
}
