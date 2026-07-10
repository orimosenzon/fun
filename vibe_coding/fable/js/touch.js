// Touch controls for mobile: a virtual joystick (steer + pitch, same
// airplane-yoke convention as the arrow keys — pull the knob down to climb)
// plus hold-buttons for throttle/collective/boosts and a small utility menu.
// Everything uses Pointer Events so it also works with a mouse for testing.
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export function makeTouchControls(actions) {
  const state = {
    steer: 0, pitch: 0,
    throttleUp: false, throttleDown: false,
    collUp: false, collDown: false,
    boostRear: false,
  };

  const el = (cls, parent, tag = 'div') => {
    const d = document.createElement(tag);
    d.className = cls;
    parent.appendChild(d);
    return d;
  };

  const root = el('touch-ui', document.body);

  // --- joystick: x = steer, y = pitch (down = nose up, like a real yoke) ---
  const joyBase = el('tjoy', root);
  const joyKnob = el('tjoy-knob', joyBase);
  const JOY_R = 46;
  let joyId = null, joyCX = 0, joyCY = 0;

  function joyMove(e) {
    let dx = e.clientX - joyCX, dy = e.clientY - joyCY;
    const d = Math.hypot(dx, dy);
    if (d > JOY_R) { dx *= JOY_R / d; dy *= JOY_R / d; }
    joyKnob.style.transform = `translate(${dx}px, ${dy}px)`;
    // gentle expo curve: fine authority near center, full deflection at the rim
    const expo = v => 0.4 * v + 0.6 * v * Math.abs(v);
    state.steer = expo(clamp(dx / JOY_R, -1, 1));
    state.pitch = expo(clamp(dy / JOY_R, -1, 1));
  }
  function joyReset() {
    state.steer = 0; state.pitch = 0;
    joyKnob.style.transform = 'translate(0,0)';
  }
  joyBase.addEventListener('pointerdown', e => {
    joyId = e.pointerId;
    const r = joyBase.getBoundingClientRect();
    joyCX = r.left + r.width / 2; joyCY = r.top + r.height / 2;
    try { joyBase.setPointerCapture(joyId); } catch { /* some browsers reject synthetic/edge-case pointer ids */ }
    joyMove(e);
    e.preventDefault();
  });
  joyBase.addEventListener('pointermove', e => { if (e.pointerId === joyId) { joyMove(e); e.preventDefault(); } });
  const joyEnd = e => { if (e.pointerId === joyId) { joyId = null; joyReset(); } };
  joyBase.addEventListener('pointerup', joyEnd);
  joyBase.addEventListener('pointercancel', joyEnd);

  // --- press-and-hold buttons (throttle, collective, boosts) ---
  function holdBtn(cls, parent, label, onDown, onUp) {
    const b = el(cls, parent, 'button');
    b.type = 'button';
    b.textContent = label;
    b.addEventListener('pointerdown', e => {
      try { b.setPointerCapture(e.pointerId); } catch { /* some browsers reject synthetic/edge-case pointer ids */ }
      onDown();
      b.classList.add('active');
      e.preventDefault();
    });
    const release = () => { onUp(); b.classList.remove('active'); };
    b.addEventListener('pointerup', release);
    b.addEventListener('pointercancel', release);
    return b;
  }

  const side = el('tside', root);
  const thrCol = el('tvert-pair', side);
  const thrBox = el('tvert', thrCol);
  el('tlabel', thrBox).textContent = 'מצערת';
  holdBtn('tbtn', thrBox, '▲', () => state.throttleUp = true, () => state.throttleUp = false);
  holdBtn('tbtn', thrBox, '▼', () => state.throttleDown = true, () => state.throttleDown = false);
  const colBox = el('tvert', thrCol);
  el('tlabel', colBox).textContent = 'גובה';
  holdBtn('tbtn', colBox, '▲', () => state.collUp = true, () => state.collUp = false);
  holdBtn('tbtn', colBox, '▼', () => state.collDown = true, () => state.collDown = false);

  // RCS pulse buttons: a tap = one metered impulse (handled in physics)
  function pulseBtn(parent, label, which) {
    const b = el('tbtn tsmall', parent, 'button');
    b.type = 'button';
    b.textContent = label;
    b.addEventListener('pointerdown', e => {
      actions.onPulse(which);
      b.classList.add('active');
      setTimeout(() => b.classList.remove('active'), 160);
      e.preventDefault();
    });
  }
  const rollRow = el('troll', side);
  pulseBtn(rollRow, '↺', 'L');
  pulseBtn(rollRow, '⬆', 'C');
  pulseBtn(rollRow, '↻', 'R');

  holdBtn('tboost', side, '🔥', () => state.boostRear = true, () => state.boostRear = false);

  // --- one-shot utility menu ---
  const menu = el('tmenu', root);
  function tapBtn(label, fn) {
    const b = el('tbtn tsmall', menu, 'button');
    b.type = 'button';
    b.textContent = label;
    b.addEventListener('click', fn);
  }
  tapBtn('📷', actions.onCamera);
  tapBtn('T', actions.onAssist);
  tapBtn('⟲', actions.onReset);
  tapBtn('🔊', actions.onMute);
  tapBtn('?', actions.onHelp);
  if (actions.onLang) tapBtn('א/A', actions.onLang);

  return state;
}
