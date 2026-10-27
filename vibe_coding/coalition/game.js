"use strict";

const QUARTERS = 16;
const AP_PER_QUARTER = 3;
const LEAVE_BELOW = 15;
const WARN_BELOW = 30;

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const rand = (a, b) => a + Math.floor(Math.random() * (b - a + 1));
const shuffle = (arr) => {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = rand(0, i); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
};
const partyById = (id) => S.parties.find((p) => p.id === id);

// ---------- אחסון (רק נוחות: שמירת משחק ושיאים) ----------
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
  del(k) { try { localStorage.removeItem(k); } catch {} },
};

// ---------- סאונד ----------
const audio = {
  ctx: null, sfx: true, music: false, musicTimer: null,
  ensure() { if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)(); },
  tone(freq, dur = 0.12, type = "square", vol = 0.06, when = 0) {
    this.ensure();
    const t = this.ctx.currentTime + when;
    const o = this.ctx.createOscillator(), g = this.ctx.createGain();
    o.type = type; o.frequency.value = freq;
    g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(this.ctx.destination); o.start(t); o.stop(t + dur);
  },
  play(kind) {
    if (!this.sfx) return;
    const seq = {
      click: [[660, 0.05]],
      good: [[523, 0.1], [659, 0.1], [784, 0.15]],
      bad: [[300, 0.15], [220, 0.25]],
      fall: [[392, 0.2], [330, 0.2], [262, 0.2], [196, 0.5]],
      win: [[523, 0.12], [659, 0.12], [784, 0.12], [1047, 0.4]],
    }[kind] || [];
    let w = 0;
    for (const [f, d] of seq) { this.tone(f, d, "square", 0.05, w); w += d * 0.9; }
  },
  toggleMusic() {
    this.music = !this.music;
    clearInterval(this.musicTimer);
    if (!this.music) return;
    // מנגינת מארש קצרה בלולאה
    const notes = [392, 392, 523, 392, 659, 587, 523, 494, 440, 440, 587, 440, 698, 659, 587, 523];
    let i = 0;
    this.musicTimer = setInterval(() => {
      this.tone(notes[i % notes.length], 0.2, "triangle", 0.035);
      if (i % 4 === 0) this.tone(notes[i % notes.length] / 2, 0.3, "sine", 0.03);
      i++;
    }, 260);
  },
};

// ---------- מצב ----------
let S = null;        // משחק נוכחי
let round = null;    // { players: [...], idx, results: [] }

function newGameState(playerName) {
  return {
    player: playerName,
    quarter: 1,
    polls: 48,
    budget: 12,
    ap: AP_PER_QUARTER,
    ministries: shuffle(MINISTRIES).slice(0, 4),
    parties: PARTIES.map((p) => ({ id: p.id, seats: p.seats, sat: rand(50, 65), inCoalition: true, met: false, rebelled: false })),
    deck: shuffle(EVENTS.map((_, i) => i)),
    event: null,
    resolved: false,
    pressUses: 0,
    log: [],
    over: false,
  };
}

const base = (id) => PARTIES.find((p) => p.id === id);
const seats = () => S.parties.filter((p) => p.inCoalition).reduce((s, p) => s + p.seats, 0);

function log(msg) { S.log.unshift(`ר${S.quarter}: ${msg}`); S.log = S.log.slice(0, 40); }

function drawEvent() {
  if (!S.deck.length) S.deck = shuffle(EVENTS.map((_, i) => i));
  S.event = S.deck.pop();
  S.resolved = false;
}

// ---------- השפעות ----------
function applyFx(fx) {
  if (fx.polls) S.polls = clamp(S.polls + fx.polls, 0, 100);
  if (fx.budget) S.budget += fx.budget;
  if (fx.ap) S.ap = Math.max(0, S.ap + fx.ap);
  if (fx.ministry) S.ministries.splice(0, -fx.ministry);
  for (const p of S.parties) {
    if (!p.inCoalition) continue;
    let d = fx.all || 0;
    if (fx.p && fx.p[p.id]) d += fx.p[p.id];
    p.sat = clamp(p.sat + d, 0, 100);
  }
}

function fxText(fx) {
  const parts = [];
  const sgn = (n) => (n > 0 ? "+" + n : "−" + Math.abs(n));
  if (fx.polls) parts.push(`📊${sgn(fx.polls)}`);
  if (fx.budget) parts.push(`💰${sgn(fx.budget)}`);
  if (fx.ap) parts.push(`⚡${sgn(fx.ap)}`);
  if (fx.ministry) parts.push(`🎖️${sgn(fx.ministry)}`);
  if (fx.all) parts.push(`כולם${sgn(fx.all)}`);
  for (const [id, d] of Object.entries(fx.p || {})) {
    const p = partyById(id);
    if (p && p.inCoalition) parts.push(`${base(id).icon}${sgn(d)}`);
  }
  return parts.join(" · ");
}

// ---------- פעולות ----------
const ACTIONS = {
  funds(id) {
    if (S.ap < 1 || S.budget < 3) return;
    const p = partyById(id);
    S.ap--; S.budget -= 3; S.polls = clamp(S.polls - 2, 0, 100); p.sat = clamp(p.sat + 20, 0, 100);
    log(`העברת 3 מיליארד ל${base(id).name}. "זה לחיזוק הפריפריה".`);
    audio.play("good");
  },
  meet(id) {
    const p = partyById(id);
    if (S.ap < 1 || p.met) return;
    S.ap--; p.met = true; p.sat = clamp(p.sat + 8, 0, 100);
    log(`פגישת לילה עם ${base(id).leader}. הוגשו בורקס.`);
    audio.play("click");
  },
  appoint(id) {
    if (S.ap < 1 || !S.ministries.length) return;
    const p = partyById(id);
    const m = S.ministries.shift();
    S.ap--; p.sat = clamp(p.sat + 30, 0, 100); S.polls = clamp(S.polls - 1, 0, 100);
    log(`${base(id).leader} מונה ל${m}.`);
    audio.play("good");
  },
  press() {
    if (S.ap < 1) return;
    S.ap--; S.pressUses++;
    const gain = Math.max(-3, rand(0, 8) - S.pressUses);
    S.polls = clamp(S.polls + gain, 0, 100);
    log(gain > 0 ? `מסיבת עיתונאים בפריים טיים: +${gain} בסקרים.` : `מסיבת עיתונאים נוספת. הציבור כבר עבר ערוץ (${gain}).`);
    audio.play(gain > 0 ? "good" : "bad");
  },
  dog() {
    if (S.ap < 1) return;
    S.ap--;
    if (Math.random() < 0.7) { S.polls = clamp(S.polls + 3, 0, 100); log("סרטון עם הכלב קיבל מיליון צפיות. +3"); audio.play("good"); }
    else { S.polls = clamp(S.polls - 4, 0, 100); log("הכלב נשך כתב. −4"); audio.play("bad"); }
  },
  abroad() {
    if (S.ap < 2) return;
    S.ap -= 2; S.polls = clamp(S.polls + 8, 0, 100);
    for (const p of S.parties) if (p.inCoalition) p.sat = clamp(p.sat - 4, 0, 100);
    log("סיור מדיני באירופה. תמונות יפות, השותפים מרגישים מוזנחים.");
    audio.play("good");
  },
  early() {
    confirmBox("להקדים בחירות?", "הקדנציה תסתיים עכשיו. הניקוד יחושב לפי המצב הנוכחי, בלי בונוס שרידות מלאה.", () => endGame("early"));
  },
};

function choose(i) {
  if (S.resolved) return;
  const ev = EVENTS[S.event];
  const opt = ev.options[i];
  if (opt.needsMinistry && !S.ministries.length) return;
  if (opt.needsMinistry) log(`${S.ministries[0]} הוענק כדי לסגור את "${ev.title}".`);
  applyFx(opt.fx);
  S.resolved = true;
  S.chosen = i;
  log(`${ev.title}: ${opt.label}.`);
  audio.play("click");
  render();
}

// ---------- סוף רבעון ----------
function endQuarter() {
  if (!S.resolved) return;
  const events = [];

  S.budget += 3;
  if (S.budget < 0) { S.polls = clamp(S.polls - 5, 0, 100); events.push("גירעון! חברות הדירוג מורידות דירוג. −5 בסקרים"); }
  S.polls = clamp(S.polls + rand(-3, 2), 0, 100);

  for (const p of S.parties) {
    if (!p.inCoalition) continue;
    p.sat = clamp(p.sat - (p.id === "bayit" ? 4 : 8), 0, 100);
    p.met = false;
    if (S.polls < 25 && p.id !== "bayit") p.sat = clamp(p.sat - 4, 0, 100);
  }

  for (const p of S.parties) {
    if (!p.inCoalition || p.sat >= LEAVE_BELOW) continue;
    const b = base(p.id);
    if (p.id === "bayit") {
      if (!p.rebelled) {
        p.rebelled = true; p.seats -= 6; p.sat = 40;
        events.push(`6 מורדים פורשים מ${b.name} ומקימים את "הבית שלנו האמיתי".`);
      } else {
        p.seats -= 6; p.sat = 35;
        events.push(`עוד 6 מורדים פורשים מ${b.name}. כבר לא ברור של מי הבית.`);
      }
    } else {
      p.inCoalition = false;
      events.push(`${b.name} (${p.seats} מנדטים) פורשת מהקואליציה! ${b.leader}: "עשינו כל מה שיכולנו".`);
    }
  }

  if (S.quarter % 4 === 0 && S.ministries.length < MINISTRIES.length) {
    const free = MINISTRIES.filter((m) => !S.ministries.includes(m));
    S.ministries.push(free[rand(0, free.length - 1)]);
    events.push("שר התפטר בעקבות פרשה. תיק אחד התפנה.");
  }

  events.forEach((e) => log(e));

  if (seats() < 61) { save(); render(); return endGame("fell", events); }
  if (S.quarter >= QUARTERS) { save(); render(); return endGame("full", events); }

  S.quarter++;
  S.ap = AP_PER_QUARTER;
  drawEvent();
  save();
  render();
  if (events.length) {
    audio.play(events.some((e) => e.includes("פורש")) ? "bad" : "click");
    showModal(`<h2>סוף רבעון ${S.quarter - 1}</h2>${events.map((e) => `<p>${e}</p>`).join("")}<button class="big teal" onclick="closeModal()">הלאה</button>`);
  }
}

function score(kind) {
  const q = kind === "full" ? QUARTERS : S.quarter - 1;
  let s = q * 5 + S.polls;
  if (kind !== "fell") s += Math.max(0, seats() - 61) * 2 + Math.max(0, Math.min(S.budget, 20));
  if (kind === "full") s += 20;
  if (kind === "fell") s = Math.round(s / 2);
  return Math.max(0, Math.round(s));
}

function endGame(kind, events = []) {
  S.over = true;
  const sc = score(kind);
  const title = TITLES.find((t) => sc >= t.min).t;
  round.results.push({ name: S.player, score: sc, kind, quarter: S.quarter });
  const best = store.get("coalition61.best") || [];
  best.push({ name: S.player, score: sc, date: new Date().toLocaleDateString("he-IL") });
  best.sort((a, b) => b.score - a.score);
  store.set("coalition61.best", best.slice(0, 10));
  store.del("coalition61.save");

  const head = {
    fell: ["💥 הממשלה נפלה", `ברבעון ${S.quarter} הקואליציה ירדה ל-${seats()} מנדטים.`],
    full: ["🏆 שרדת קדנציה מלאה!", `בחירות: ${S.polls}% בסקרים, ${seats()} מנדטים.`],
    early: ["🗳️ בחירות מוקדמות", `הלכת לבחירות ברבעון ${S.quarter} עם ${S.polls}% בסקרים.`],
  }[kind];
  audio.play(kind === "fell" ? "fall" : "win");

  const next = round.idx + 1 < round.players.length;
  const btn = next
    ? `<button class="big teal" onclick="nextPlayer()">תורו של ${round.players[round.idx + 1]}</button>`
    : `<button class="big teal" onclick="finishRound()">לתוצאות</button>`;
  showModal(`<h2>${head[0]}</h2>
    ${events.map((e) => `<p>${e}</p>`).join("")}
    <p>${head[1]}</p>
    <div class="score">${sc}</div>
    <p>התואר שלך: <b>${title}</b></p>${btn}`);
}

function nextPlayer() {
  round.idx++;
  startGameFor(round.players[round.idx]);
  closeModal();
}

function finishRound() {
  const rows = round.results.slice().sort((a, b) => b.score - a.score);
  const kindTxt = { fell: "נפל", full: "קדנציה מלאה", early: "הקדים" };
  showModal(`<h2>${rows.length > 1 ? "🏆 " + rows[0].name + " ניצח!" : "סיכום"}</h2>
    <table style="width:100%">${rows.map((r, i) => `<tr><td>${i + 1}</td><td>${r.name}</td><td>${kindTxt[r.kind]}</td><td><b>${r.score}</b></td></tr>`).join("")}</table>
    <button class="big teal" onclick="closeModal();showMenu()">לתפריט</button>`);
}

// ---------- שמירה ----------
function save() { store.set("coalition61.save", { S, round }); }

// ---------- רינדור ----------
function setStat(id, val, cls) {
  const el = $(id);
  if (el.textContent !== String(val)) { el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); }
  el.textContent = val;
  el.className = cls || "";
}

function render() {
  const n = seats();
  $("hudPlayer").textContent = S.player;
  $("hudQuarter").textContent = `רבעון ${S.quarter} מתוך ${QUARTERS} · שנה ${Math.ceil(S.quarter / 4)}`;
  setStat("hudSeats", n, n < 64 ? "bad" : "");
  setStat("hudPolls", S.polls + "%", S.polls < 30 ? "bad" : S.polls > 60 ? "good" : "");
  setStat("hudBudget", "₪" + S.budget + "B", S.budget < 0 ? "bad" : "");
  setStat("hudAP", "⚡".repeat(S.ap) || "0");
  setStat("hudMin", S.ministries.length);

  $("parties").innerHTML = S.parties.map((p) => {
    const b = base(p.id);
    const lvl = p.sat < WARN_BELOW ? "low" : p.sat < 55 ? "mid" : "";
    const dis = !p.inCoalition || S.over;
    return `<div class="party ${p.inCoalition ? "" : "gone"} ${p.inCoalition && p.sat < WARN_BELOW ? "warn" : ""}" style="--c:${b.c}">
      <div class="icon">${b.icon}</div>
      <h3>${b.name}</h3>
      <div class="leader">${b.leader}</div>
      <div class="seats">${p.seats} מנדטים${p.inCoalition ? "" : " · באופוזיציה"}</div>
      <div class="meter" title="שביעות רצון ${p.sat}"><i class="${lvl}" style="width:${p.inCoalition ? p.sat : 0}%"></i></div>
      <div class="wants">${p.inCoalition ? (p.sat < WARN_BELOW ? "⚠️ מאיימים לפרוש! " : "") + "רוצים: " + b.wants : "הודיעו שהם 'לא יחזרו לעולם'."}</div>
      <div class="pbtns">
        <button onclick="act('funds','${p.id}')" ${dis || S.ap < 1 || S.budget < 3 ? "disabled" : ""} title="⚡1, 3 מיליארד, שביעות +20, סקרים −2">💰 כספים</button>
        <button onclick="act('meet','${p.id}')" ${dis || S.ap < 1 || p.met ? "disabled" : ""} title="⚡1, שביעות +10, פעם ברבעון">🌙 פגישה</button>
        <button onclick="act('appoint','${p.id}')" ${dis || S.ap < 1 || !S.ministries.length ? "disabled" : ""} title="⚡1, תיק אחד, שביעות +30">🎖️ תיק</button>
      </div>
    </div>`;
  }).join("");

  const ev = EVENTS[S.event];
  $("event").innerHTML = `<div class="emoji">${ev.emoji}</div>
    <span class="tag">${ev.tag}</span>
    <h2>${ev.title}</h2>
    <p>${ev.text}</p>
    ${ev.options.map((o, i) => {
      const off = S.resolved || S.over || (o.needsMinistry && !S.ministries.length);
      const picked = S.resolved && S.chosen === i;
      return `<button class="choice ${S.resolved && !picked ? "done" : ""}" onclick="choose(${i})" ${off ? "disabled" : ""}>
        ${picked ? "✔ " : ""}${o.label}<small>${fxText(o.fx) || "בלי השפעה ניכרת"}${o.needsMinistry ? " · דורש תיק פנוי" : ""}</small></button>`;
    }).join("")}
    ${S.resolved ? "" : `<p class="done">קודם מחליטים על המשבר, אחר כך אפשר לסיים את הרבעון.</p>`}`;

  const off = S.over;
  $("actions").innerHTML = `
    <button onclick="act('press')" ${off || S.ap < 1 ? "disabled" : ""}>🎤 מסיבת עיתונאים<small>⚡1 · סקרים, פוחת עם הזמן</small></button>
    <button onclick="act('dog')" ${off || S.ap < 1 ? "disabled" : ""}>🐕 סרטון עם הכלב<small>⚡1 · 70%: +3, אחרת −4</small></button>
    <button onclick="act('abroad')" ${off || S.ap < 2 ? "disabled" : ""}>✈️ סיור בחו"ל<small>⚡2 · סקרים +8, כולם −4</small></button>
    <button onclick="act('early')" ${off ? "disabled" : ""}>🗳️ בחירות מוקדמות<small>לסיים עכשיו</small></button>`;

  $("btnEnd").disabled = !S.resolved || S.over;
  $("log").innerHTML = S.log.map((l) => `<li>${l}</li>`).join("");
}

function act(name, arg) {
  if (S.over) return;
  ACTIONS[name](arg);
  save();
  render();
}

// ---------- חלונות ----------
function showModal(html) { $("modalBox").innerHTML = html; $("modal").hidden = false; }
function closeModal() { $("modal").hidden = true; }
function confirmBox(title, text, yes) {
  showModal(`<h2>${title}</h2><p>${text}</p>
    <button class="big teal" id="yesBtn">כן</button><button class="big" onclick="closeModal()">ביטול</button>`);
  $("yesBtn").onclick = () => { closeModal(); yes(); };
}

// ---------- תפריט ----------
function renderNames() {
  const n = +$("playerCount").value;
  const old = [...document.querySelectorAll("#names input")].map((i) => i.value);
  $("names").innerHTML = Array.from({ length: n }, (_, i) =>
    `<input placeholder="שם שחקן ${i + 1}" value="${old[i] || ""}" maxlength="20">`).join("");
}

function renderLeaderboard() {
  const best = store.get("coalition61.best") || [];
  $("leaderboard").innerHTML = best.length
    ? `<h3>היכל התהילה</h3><table>${best.map((b, i) => `<tr><td>${i + 1}</td><td>${b.name}</td><td>${b.score}</td><td>${b.date}</td></tr>`).join("")}</table>`
    : "";
}

function showMenu() {
  $("game").hidden = true; $("menu").hidden = false;
  $("btnResume").hidden = !store.get("coalition61.save");
  renderLeaderboard();
}

function startGameFor(name) {
  S = newGameState(name);
  drawEvent();
  log(`${name} הושבע לראשות הממשלה. 66 מנדטים, מה כבר יכול לקרות?`);
  save();
  render();
}

$("playerCount").onchange = renderNames;
$("btnNew").onclick = () => {
  audio.play("click");
  const names = [...document.querySelectorAll("#names input")].map((el, i) => el.value.trim() || `שחקן ${i + 1}`);
  round = { players: names, idx: 0, results: [] };
  $("menu").hidden = true; $("game").hidden = false;
  startGameFor(names[0]);
};
$("btnResume").onclick = () => {
  const s = store.get("coalition61.save");
  if (!s) return;
  S = s.S; round = s.round;
  $("menu").hidden = true; $("game").hidden = false;
  render();
};
$("btnEnd").onclick = endQuarter;
$("btnMusic").onclick = () => {
  audio.toggleMusic();
  $("btnMusic").innerHTML = `🎵<br>מוזיקה: ${audio.music ? "פועל" : "כבוי"}`;
};
$("btnSfx").onclick = () => {
  audio.sfx = !audio.sfx;
  $("btnSfx").innerHTML = `${audio.sfx ? "🔈" : "🔇"}<br>אפקטים: ${audio.sfx ? "פועל" : "כבוי"}`;
};

renderNames();
showMenu();
