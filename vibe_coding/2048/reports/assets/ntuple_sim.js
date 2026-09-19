/* מעבדת רשת ה-n-tuple: אותו אלגוריתם כמו rl/ntuple_td.py (TD(0) על afterstates), על הרשת הקטנה של הדפדפן
   (5 חלונות של 4 משבצות, 8 סימטריות, 40 קריאות ללוח), ב-JavaScript טהור. הקובץ גם מצייר את הלוחות הסטטיים
   של האיורים בסעיף השיטה (data-board / data-cells). נטען בכל הדו"חות, ופועל רק אם יש בדף .ntsim או .board[data-board]. */
(function () {
  const TUPLES_SMALL = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5], [1, 2, 5, 6], [5, 6, 9, 10]];
  const TUPLE_NAMES = ["שורה חיצונית", "שורה פנימית", "ריבוע בפינה", "ריבוע בצלע", "ריבוע במרכז"];
  const SYM_NAMES = ["כמו שהוא", "שיקוף", "סיבוב 90°", "90° ושיקוף", "סיבוב 180°", "180° ושיקוף", "סיבוב 270°", "270° ושיקוף"];
  const DIRS = ["↑ למעלה", "→ ימינה", "↓ למטה", "← שמאלה"];
  const fmt = (v, d) => (window.R ? R.fmt(v, d) : String(v));
  // משקל בטבלה: ספרות לפי הגודל, כדי ששינוי קטן לא ייעלם בעיגול
  const fw = (v) => (Math.abs(v) >= 1000 ? fmt(v, 0) : Math.abs(v) >= 10 ? fmt(v, 1) : fmt(v, 3));
  const fd = (v) => (Math.abs(v) >= 1 ? fmt(v, 2) : fmt(v, 4));
  const sgn = (v) => (v < 0 ? "−" : "+");

  // ------------------------------------------------------------------
  // חוקי המשחק על לוח שטוח של 16 מעריכים, עם טבלת שורות כמו ב-rl/game2048.py
  // ------------------------------------------------------------------
  const ROW_LEFT = new Uint16Array(65536), ROW_SCORE = new Uint32Array(65536);
  for (let code = 0; code < 65536; code++) {
    const tiles = [(code >> 12) & 15, (code >> 8) & 15, (code >> 4) & 15, code & 15].filter((v) => v);
    const out = [];
    let score = 0, i = 0;
    while (i < tiles.length) {
      if (i + 1 < tiles.length && tiles[i] === tiles[i + 1]) { const m = Math.min(tiles[i] + 1, 15); out.push(m); score += 2 ** m; i += 2; }
      else { out.push(tiles[i]); i += 1; }
    }
    while (out.length < 4) out.push(0);
    ROW_LEFT[code] = (out[0] << 12) | (out[1] << 8) | (out[2] << 4) | out[3];
    ROW_SCORE[code] = score;
  }
  // (משבצת ראשונה, צעד) לכל קו, לכל כיוון: 0 למעלה, 1 ימינה, 2 למטה, 3 שמאלה
  const LINE_START = [[0, 1, 2, 3], [3, 7, 11, 15], [12, 13, 14, 15], [0, 4, 8, 12]];
  const LINE_STEP = [4, -1, -4, 1];
  let slideChanged = false;
  function slide(board, a, out) {
    let score = 0;
    slideChanged = false;
    const step = LINE_STEP[a];
    for (let line = 0; line < 4; line++) {
      const c0 = LINE_START[a][line];
      const code = (board[c0] << 12) | (board[c0 + step] << 8) | (board[c0 + 2 * step] << 4) | board[c0 + 3 * step];
      const nw = ROW_LEFT[code];
      score += ROW_SCORE[code];
      if (nw !== code) slideChanged = true;
      out[c0] = (nw >> 12) & 15; out[c0 + step] = (nw >> 8) & 15; out[c0 + 2 * step] = (nw >> 4) & 15; out[c0 + 3 * step] = nw & 15;
    }
    return score;
  }
  function spawn(board) {
    let n = 0;
    for (let i = 0; i < 16; i++) if (board[i] === 0) n++;
    if (!n) return -1;
    let k = Math.floor(Math.random() * n);
    for (let i = 0; i < 16; i++) {
      if (board[i] === 0) {
        if (k === 0) { board[i] = Math.random() < 0.1 ? 2 : 1; return i; }
        k--;
      }
    }
    return -1;
  }
  // 8 הסימטריות בסדר של rl/ntuple_td.py: perm[i] = איזו משבצת מקורית נוחתת במיקום i
  function symmetries() {
    let m = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]];
    const rot90 = (a) => a[0].map((_, i) => a.map((row) => row[3 - i]));
    const fliplr = (a) => a.map((row) => row.slice().reverse());
    const out = [];
    for (let k = 0; k < 4; k++) { out.push(m.flat()); out.push(fliplr(m).flat()); m = rot90(m); }
    return out;
  }

  // ------------------------------------------------------------------
  // המנוע: הטבלאות, המאפיינים, המשחק הנוכחי ולולאת ה-TD
  // ------------------------------------------------------------------
  class Engine {
    constructor(tuples) {
      this.alpha = 0.1;
      this.setTuples(tuples || TUPLES_SMALL);
      this.board = new Uint8Array(16);
      this.after = [0, 1, 2, 3].map(() => new Uint8Array(16));
      this.idx = [0, 1, 2, 3].map(() => new Int32Array(this.nFeats));
      this.prevIdx = new Int32Array(this.nFeats);
      this.prevAfter = new Uint8Array(16);
      this.hasPrev = false;
      this.resetStats();
      this.newGame();
    }
    setTuples(tuples) {
      this.tuples = tuples;
      this.n = tuples[0].length;
      this.nTables = tuples.length;
      this.tableSize = 16 ** this.n;
      this.nFeats = 8 * this.nTables;
      this.feats = new Int8Array(this.nFeats * this.n);
      this.offsets = new Int32Array(this.nFeats);
      this.featTuple = new Int8Array(this.nFeats);
      this.featSym = new Int8Array(this.nFeats);
      let f = 0;
      symmetries().forEach((sym, s) => {
        tuples.forEach((cells, j) => {
          for (let k = 0; k < this.n; k++) this.feats[f * this.n + k] = sym[cells[k]];
          this.offsets[f] = j * this.tableSize;
          this.featTuple[f] = j; this.featSym[f] = s;
          f++;
        });
      });
      this.table = new Float32Array(this.nTables * this.tableSize);
      this.touched = 0;
    }
    resetStats() {
      this.games = 0; this.moves = 0; this.recent = []; this.history = []; this.bestScore = 0; this.reach2048 = 0;
      this.sumAbsDelta = 0; this.nUpdates = 0; this.source = "zeros";
    }
    resetTable() { this.table.fill(0); this.touched = 0; this.resetStats(); this.newGame(); }
    newGame() { this.board.fill(0); spawn(this.board); spawn(this.board); this.score = 0; this.gameMoves = 0; this.hasPrev = false; }
    featIdx(board, out) {
      const n = this.n, feats = this.feats;
      for (let f = 0; f < this.nFeats; f++) {
        let idx = 0;
        const base = f * n;
        for (let k = 0; k < n; k++) idx = idx * 16 + board[feats[base + k]];
        out[f] = this.offsets[f] + idx;
      }
    }
    valueIdx(idx) { let s = 0; for (let f = 0; f < this.nFeats; f++) s += this.table[idx[f]]; return s; }
    value(board) { const tmp = new Int32Array(this.nFeats); this.featIdx(board, tmp); return this.valueIdx(tmp); }
    apply(idx, delta) {
      const w = (this.alpha / this.nFeats) * delta;
      for (let f = 0; f < this.nFeats; f++) {
        const i = idx[f];
        if (this.table[i] === 0 && w !== 0) this.touched++;
        this.table[i] += w;
      }
      this.sumAbsDelta += Math.abs(delta); this.nUpdates++;
    }
    endEpisode() {
      this.games++;
      this.recent.push(this.score);
      if (this.recent.length > 50) this.recent.shift();
      if (this.score > this.bestScore) this.bestScore = this.score;
      let mx = 0; for (let i = 0; i < 16; i++) if (this.board[i] > mx) mx = this.board[i];
      if (mx >= 11) this.reach2048++;
      if (this.games % 10 === 0) this.history.push([this.games, this.recent.reduce((a, b) => a + b, 0) / this.recent.length]);
      this.newGame();
    }
    // מהלך אחד של TD(0) על afterstates, בדיוק train_chunk של rl/ntuple_td.py. detail=true מחזיר את כל מה שקרה, לתצוגה.
    step(detail) {
      const d = detail ? { before: Array.from(this.board), cands: [], hadPrev: this.hasPrev, prevAfter: Array.from(this.prevAfter) } : null;
      let best = -1, bestV = -Infinity, bestR = 0;
      for (let a = 0; a < 4; a++) {
        const sc = slide(this.board, a, this.after[a]);
        if (!slideChanged) { if (d) d.cands.push({ valid: false }); continue; }
        this.featIdx(this.after[a], this.idx[a]);
        const V = this.valueIdx(this.idx[a]);
        const v = sc + V;
        if (d) d.cands.push({ valid: true, board: Array.from(this.after[a]), r: sc, V, total: v });
        if (v > bestV) { bestV = v; best = a; bestR = sc; }
      }
      if (d) d.best = best;
      if (best < 0) {
        // סיום משחק: ה-afterstate האחרון נדחף לעבר 0
        if (this.hasPrev) {
          const oldV = this.valueIdx(this.prevIdx);
          const delta = 0 - oldV;
          if (d) d.update = this.describeUpdate(oldV, 0, 0, delta, true);
          this.apply(this.prevIdx, delta);
        }
        if (d) { d.gameOver = true; d.finalScore = this.score; d.finalMoves = this.gameMoves; }
        this.endEpisode();
        this.moves++;
        return d;
      }
      if (this.hasPrev) {
        const oldV = this.valueIdx(this.prevIdx);
        const delta = bestV - oldV;
        if (d) d.update = this.describeUpdate(oldV, bestR, bestV - bestR, delta, false);
        this.apply(this.prevIdx, delta);
      }
      this.prevIdx.set(this.idx[best]);
      this.prevAfter.set(this.after[best]);
      this.hasPrev = true;
      this.board.set(this.after[best]);
      this.score += bestR;
      this.gameMoves++;
      this.moves++;
      const cell = spawn(this.board);
      if (d) { d.spawnCell = cell; d.spawnExp = cell >= 0 ? this.board[cell] : 0; d.afterSpawn = Array.from(this.board); d.score = this.score; }
      return d;
    }
    describeUpdate(oldV, r, nextV, delta, terminal) {
      const w = (this.alpha / this.nFeats) * delta;
      const entries = [];
      for (let f = 0; f < this.nFeats; f++) {
        const i = this.prevIdx[f];
        const cells = Array.from(this.feats.subarray(f * this.n, (f + 1) * this.n));
        entries.push({ f, tuple: this.featTuple[f], sym: this.featSym[f], cells, index: i - this.offsets[f], before: this.table[i], exps: cells.map((c) => this.prevAfter[c]) });
      }
      return { oldV, r, nextV, delta, w, terminal, entries };
    }
    // לולאה צמודה לאימון מהיר: n מהלכים בלי תצוגה
    train(n) { for (let i = 0; i < n; i++) this.step(false); }
  }

  // ------------------------------------------------------------------
  // ציור לוחות
  // ------------------------------------------------------------------
  function renderBoard(el, board, opts) {
    opts = opts || {};
    el.classList.add("board");
    if (opts.size) el.style.setProperty("--bs", opts.size);
    const hl = new Set(opts.highlight || []);
    el.innerHTML = "";
    for (let i = 0; i < 16; i++) {
      const c = document.createElement("div");
      const v = board ? board[i] : 0;
      let cls = "cell";
      if (opts.indices) { c.textContent = i; }
      else if (v) { const val = 2 ** v; c.textContent = val; cls += val > 2048 ? " tsuper" : " t" + val; }
      if (hl.has(i)) cls += " hl";
      else if (opts.dimOthers && hl.size) cls += " dim";
      c.className = cls;
      if (opts.spawn === i) c.style.outline = "3px solid var(--series-1)";
      el.appendChild(c);
    }
    return el;
  }
  function parseList(s) { return s ? s.split(",").map((x) => parseInt(x.trim(), 10)) : []; }
  function renderStaticBoards(root) {
    root.querySelectorAll(".board[data-board], .board[data-cells]").forEach((el) => {
      const board = el.dataset.board ? parseList(el.dataset.board) : null;
      renderBoard(el, board, { indices: el.dataset.board == null, highlight: parseList(el.dataset.cells), dimOthers: el.dataset.dim != null, size: el.dataset.size, spawn: el.dataset.spawn != null ? +el.dataset.spawn : -1 });
    });
  }

  // ------------------------------------------------------------------
  // הווידג'ט
  // ------------------------------------------------------------------
  function mountSim(root) {
    const eng = new Engine();
    root.innerHTML = `
      <h4>מעבדה: רשת n-tuple לומדת כאן, בדפדפן</h4>
      <p class="desc">הרשת הקטנה של הדפדפן (5 חלונות של 4 משבצות, 8 סימטריות, 40 קריאות לכל לוח, 327,680 מספרים בטבלאות) מתחילה מטבלה של אפסים. כל "מהלך אחד" מראה את שלושת השלבים של האלגוריתם עם המספרים האמיתיים; "אימון" מריץ אלפי משחקים בשנייה ומצייר את עקומת הלמידה. אפשר גם לטעון את הטבלה שאומנה בפייתון (זו שמשחקת בדף המשחק) ולראות מה היא חושבת על לוח.</p>
      <div class="toolbar">
        <button class="b-step primary">מהלך אחד</button>
        <button class="b-game">עד סוף המשחק</button>
        <button class="b-train">אימון ▶</button>
        <button class="b-reset">איפוס לאפסים</button>
        <button class="b-load">טען את הטבלה המאומנת</button>
        <label>alpha <select class="alpha"><option value="0.1" selected>0.1</option><option value="0.03">0.03</option><option value="0.01">0.01</option><option value="0.3">0.3</option><option value="0">0 (בלי למידה)</option></select></label>
      </div>
      <div class="counters">
        <div class="counter"><div class="l">משחקים</div><div class="v c-games">0</div></div>
        <div class="counter"><div class="l">מהלכים (= עדכונים)</div><div class="v c-moves">0</div></div>
        <div class="counter"><div class="l">ניקוד ממוצע, 50 האחרונים</div><div class="v c-mean">–</div></div>
        <div class="counter"><div class="l">הגיעו ל-2048</div><div class="v c-2048">–</div></div>
        <div class="counter"><div class="l">כניסות בטבלה שנגעו בהן</div><div class="v c-touched">0<small> / 327,680</small></div></div>
        <div class="counter"><div class="l">מהירות</div><div class="v c-sps">–<small> מהלכים/שנייה</small></div></div>
      </div>
      <div class="live">
        <div class="now"><div class="live-board"></div><div class="live-cap">המשחק הנוכחי</div></div>
        <div class="chartwrap"><div class="chart live-chart"></div><div class="note" style="margin:0">ניקוד ממוצע של 50 המשחקים האחרונים, לפי מספר המשחקים שהרשת שיחקה. כל משחק הוא גם אימון.</div></div>
      </div>
      <div class="detail"></div>
    `;
    const $ = (s) => root.querySelector(s);
    const liveBoard = $(".live-board");
    const detail = $(".detail");
    const chart = R.liveLine($(".live-chart"), { series: [{ label: "ניקוד ממוצע (50 אחרונים)", color: "s4" }], xTitle: "משחקים", yTitle: "ניקוד" });
    let chartN = 0;
    let training = null;
    let lastSps = null;

    function counters() {
      $(".c-games").textContent = fmt(eng.games);
      $(".c-moves").textContent = fmt(eng.moves);
      $(".c-mean").textContent = eng.recent.length ? fmt(eng.recent.reduce((a, b) => a + b, 0) / eng.recent.length, 0) : "–";
      $(".c-2048").textContent = eng.games ? Math.round(100 * eng.reach2048 / eng.games) + "%" : "–";
      $(".c-touched").innerHTML = `${fmt(eng.touched)}<small> / ${fmt(eng.table.length)} (${(100 * eng.touched / eng.table.length).toFixed(1)}%)</small>`;
      $(".c-sps").innerHTML = lastSps ? `${fmt(lastSps, 0)}<small> מהלכים/שנייה</small>` : "–";
      while (chartN < eng.history.length) { const [g, m] = eng.history[chartN++]; chart.push(0, g, m); }
    }
    function drawLive() { renderBoard(liveBoard, eng.board, { size: "150px" }); $(".live-cap").textContent = `המשחק הנוכחי: ${fmt(eng.score)} נקודות, ${fmt(eng.gameMoves)} מהלכים`; }

    const tilesOf = (exps) => exps.map((e) => (e ? 2 ** e : "·")).join(" ");
    function showDetail(d) {
      const n = eng.nFeats;
      let h = "";
      // ---- שלב 1: ההחלטה ----
      h += `<h5>1. ההחלטה: ארבע החלקות, ולכל אחת "ניקוד מיידי + ערך ה-afterstate"</h5>`;
      h += `<div class="stage"><div class="now"><div class="d-before"></div><div>הלוח לפני המהלך, \\(s_t\\)</div></div><div class="cands">`;
      d.cands.forEach((c, a) => {
        if (!c.valid) { h += `<div class="cand invalid"><div class="dir">${DIRS[a]}</div><div class="d-cand" data-a="${a}"></div><div class="nums">לא משנה את הלוח</div></div>`; return; }
        h += `<div class="cand ${a === d.best ? "best" : ""}"><div class="dir">${DIRS[a]}${a === d.best ? " ✓" : ""}</div><div class="d-cand" data-a="${a}"></div>
              <div class="nums">r = ${fmt(c.r)}<br>V = ${fw(c.V)}<br><b>r + V = ${fw(c.total)}</b></div></div>`;
      });
      h += `</div></div>`;
      if (d.best >= 0) {
        const allZero = d.cands.every((c) => !c.valid || c.V === 0);
        h += `<p class="note">נבחר <b>${DIRS[d.best]}</b>, כי הסכום שלו הוא הגבוה ביותר. אין כאן תוחלת על האריח האקראי ואין הסתברויות: ארבעה מספרים, והגדול מנצח.${allZero ? " (כל ה-V כאן הם 0, כי הכניסות האלה עוד לא עודכנו מעולם, ולכן הבחירה היא לפי הניקוד המיידי בלבד, כמו החמדן; שוויון נשבר לפי הסדר למעלה, ימינה, למטה, שמאלה.)" : ""}</p>`;
      } else {
        h += `<p class="note"><b>אין מהלך חוקי: המשחק נגמר</b> עם ${fmt(d.finalScore)} נקודות אחרי ${fmt(d.finalMoves)} מהלכים. הרשת מתחילה משחק חדש.</p>`;
      }
      // ---- שלב 2: העדכון ----
      h += `<h5>2. הלמידה: ה-afterstate הקודם \\(s'_{t-1}\\) מתעדכן לעבר מה שקרה עכשיו</h5>`;
      if (!d.hadPrev) {
        h += `<p class="note">זה המהלך הראשון במשחק, ועדיין אין afterstate קודם לעדכן. העדכון הראשון יגיע במהלך הבא.</p>`;
      } else {
        const u = d.update;
        const target = u.r + u.nextV;
        const targetLine = u.terminal ? `target = 0 &nbsp;<i>(game over)</i>` : `target = r + V(s'<sub>t</sub>) = ${fmt(u.r)} + ${fw(u.nextV)} = <b>${fw(target)}</b>`;
        h += `<div class="stage"><div class="now"><div class="d-prev"></div><div>ה-afterstate הקודם, \\(s'_{t-1}\\)</div></div><div style="flex:1;min-width:260px">
              <div class="formula">${targetLine}<br>δ = target − V(s'<sub>t−1</sub>) = ${fw(target)} − ${fw(u.oldV)} = <b>${fw(u.delta)}</b><br>
              w ← w + (α/${n})·δ = w + (${eng.alpha}/${n})·(${fw(u.delta)}) = w <b>${sgn(u.w)} ${fd(Math.abs(u.w))}</b> &nbsp;<i>(each of the ${n} entries)</i></div>
              <p class="note">${u.terminal ? "המשחק נגמר, ולכן היעד הוא 0: מהלוח הזה לא יגיעו עוד נקודות." : "היעד (target) הוא \"מה שקיבלנו עכשיו ועוד מה שהטבלה חושבת על הלוח החדש\". "}ההפרש δ בין היעד לבין מה שהטבלה חשבה מתחלק שווה בין ${n} הכניסות שנקראו, וביחד V של ה-afterstate הקודם זז ב-α·δ = ${fw(eng.alpha * u.delta)}, עשירית מהדרך ליעד (כשאין כניסות כפולות). העדכון נכנס לטבלה מיד, ומהמהלך הבא הוא כבר משפיע על ההחלטות. עברו עם העכבר על כניסה כדי לראות את החלון שלה על הלוח.</p>
              </div></div>`;
        // טבלת הכניסות: שורה לכל חלון, עמודה לכל סימטריה
        h += `<div class="scroll"><table class="entries"><thead><tr><th>חלון \\ סימטריה</th>${SYM_NAMES.map((s) => `<th>${s}</th>`).join("")}</tr></thead><tbody>`;
        for (let j = 0; j < eng.nTables; j++) {
          h += `<tr><td>${TUPLE_NAMES[j]}<br><span class="tiles">${eng.tuples[j].join(",")}</span></td>`;
          for (let s = 0; s < 8; s++) {
            const e = u.entries.find((x) => x.tuple === j && x.sym === s);
            const after = eng.table[e.index + eng.offsets[e.f]];
            const diff = after - e.before;
            h += `<td class="e" data-f="${e.f}"><span class="tiles">${tilesOf(e.exps)}</span><br><span class="w">${fw(e.before)} → ${fw(after)}</span><br><span class="d ${diff < 0 ? "neg" : ""}">${sgn(diff)}${fd(Math.abs(diff))}</span></td>`;
          }
          h += `</tr>`;
        }
        h += `</tbody></table></div>`;
        h += `<p class="note">כל תא הוא כניסה אחת בטבלה: תוכן החלון (ארבעת האריחים, · = ריק), הערך לפני ואחרי, והשינוי. אותה כניסה יכולה להיקרא פעמיים (למשל ריבוע המרכז בסימטריות שונות עם אותו תוכן), ואז היא מתעדכנת פעמיים. שימו לב שהכניסות של חלונות שונים לא "יודעות" זו על זו: הקשר היחיד ביניהן הוא שהן סוכמות לאותו V.</p>`;
      }
      // ---- שלב 3: האריח ----
      if (d.best >= 0) {
        h += `<h5>3. הטבע משחק: אריח אקראי נופל, וזה הלוח של המהלך הבא</h5>`;
        h += `<div class="stage"><div class="now"><div class="d-after"></div><div>ה-afterstate שנבחר, \\(s'_t\\) (${fmt(d.cands[d.best].r)} נקודות)</div></div>
              <div class="arrow" style="font-size:24px;color:var(--muted);align-self:center">←</div>
              <div class="now"><div class="d-spawn"></div><div>אחרי האריח (${d.spawnExp ? 2 ** d.spawnExp : "?"} במשבצת ${d.spawnCell}), \\(s_{t+1}\\)</div></div>
              <p class="note" style="flex:1;min-width:220px">הערך שנלמד הוא של הלוח <b>לפני</b> האריח, ולכן הטבלה לא צריכה לנחש איפה הוא ייפול. במהלך הבא, \\(s'_t\\) יהיה "ה-afterstate הקודם" ויתעדכן לעבר מה שיקרה אז.</p></div>`;
      }
      detail.innerHTML = h;
      renderBoard(detail.querySelector(".d-before"), d.before, { size: "120px" });
      detail.querySelectorAll(".d-cand").forEach((el) => { const c = d.cands[+el.dataset.a]; renderBoard(el, c.valid ? c.board : d.before, { size: "96px" }); });
      if (d.hadPrev) {
        const prevEl = detail.querySelector(".d-prev");
        renderBoard(prevEl, d.prevAfter, { size: "120px" });
        detail.querySelectorAll("td.e").forEach((td) => {
          const e = d.update.entries[+td.dataset.f];
          td.addEventListener("mouseenter", () => renderBoard(prevEl, d.prevAfter, { size: "120px", highlight: e.cells, dimOthers: true }));
          td.addEventListener("mouseleave", () => renderBoard(prevEl, d.prevAfter, { size: "120px" }));
        });
      }
      if (d.best >= 0) {
        renderBoard(detail.querySelector(".d-after"), d.cands[d.best].board, { size: "120px" });
        renderBoard(detail.querySelector(".d-spawn"), d.afterSpawn, { size: "120px", spawn: d.spawnCell });
      }
      if (window.renderMathInElement) renderMathInElement(detail, { delimiters: [{ left: "\\(", right: "\\)", display: false }], throwOnError: false });
    }

    function stopTraining() {
      if (!training) return;
      clearTimeout(training);
      training = null;
      $(".b-train").textContent = "אימון ▶";
      [".b-step", ".b-game", ".b-reset", ".b-load"].forEach((s) => ($(s).disabled = false));
      counters(); drawLive();
    }
    function startTraining() {
      if (training) return stopTraining();
      $(".b-train").textContent = "עצור ⏸";
      [".b-step", ".b-game", ".b-reset", ".b-load"].forEach((s) => ($(s).disabled = true));
      detail.innerHTML = "";
      let lastDraw = 0, movesAtSec = eng.moves, tSec = performance.now();
      const tick = () => {
        const t0 = performance.now();
        while (performance.now() - t0 < 40) eng.train(2000);   // פרוסות של 40ms כדי שהדף יישאר רספונסיבי
        const now = performance.now();
        if (now - tSec > 1000) { lastSps = (eng.moves - movesAtSec) / ((now - tSec) / 1000); movesAtSec = eng.moves; tSec = now; }
        if (now - lastDraw > 150) { counters(); drawLive(); lastDraw = now; }
        training = setTimeout(tick, 0);
      };
      training = setTimeout(tick, 0);
    }
    $(".b-step").onclick = () => { stopTraining(); showDetail(eng.step(true)); counters(); drawLive(); };
    $(".b-game").onclick = () => {
      stopTraining();
      const g = eng.games;
      let d = null;
      while (eng.games === g) d = eng.step(true);
      showDetail(d); counters(); drawLive();
    };
    $(".b-train").onclick = startTraining;
    $(".b-reset").onclick = () => { stopTraining(); eng.resetTable(); lastSps = null; chart.reset(); chartN = 0; detail.innerHTML = ""; counters(); drawLive(); };
    $(".alpha").onchange = () => { eng.alpha = +$(".alpha").value; };
    $(".b-load").onclick = () => {
      stopTraining();
      const apply = () => {
        const p = window.AGENT_WEIGHTS && window.AGENT_WEIGHTS.ntuple;
        if (!p) { detail.innerHTML = `<p class="note">לא נמצאה הטבלה (game/weights/ntuple.js).</p>`; return; }
        const bin = atob(p.table.data);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const table = new Float32Array(bytes.buffer);
        if (JSON.stringify(p.tuples) !== JSON.stringify(eng.tuples)) eng.setTuples(p.tuples);
        eng.table.set(table);
        eng.touched = 0; for (let i = 0; i < table.length; i++) if (table[i] !== 0) eng.touched++;
        eng.resetStats(); eng.source = "python"; eng.newGame(); chart.reset(); chartN = 0; lastSps = null;
        const ev = p.eval || {};
        detail.innerHTML = `<p class="note">נטענה הטבלה שאומנה בפייתון (${fmt(p.transitions / 1e6, 0)} מיליון מהלכים, ${p.train_minutes} דקות על ליבת CPU אחת; בהערכה של ${fmt(ev.games)} משחקים: ממוצע ${fmt(ev.score_mean, 0)}, הגעה ל-2048 ב-${Math.round(100 * (ev.reach_2048 || 0))}%). זו בדיוק הטבלה שמשחקת בדף המשחק. לחצו "מהלך אחד" כדי לראות מה היא חושבת, או "אימון" כדי להמשיך לאמן אותה כאן.</p>`;
        counters(); drawLive();
      };
      if (window.AGENT_WEIGHTS && window.AGENT_WEIGHTS.ntuple) return apply();
      detail.innerHTML = `<p class="note">טוען את הטבלה (1.75MB)…</p>`;
      const s = document.createElement("script");
      s.src = "../game/weights/ntuple.js";
      s.onload = apply;
      s.onerror = () => { detail.innerHTML = `<p class="note">הטעינה נכשלה. הקובץ game/weights/ntuple.js נוצר על ידי rl/export_weights.py --agent ntuple.</p>`; };
      document.head.appendChild(s);
    };
    counters(); drawLive();
    root.dataset.ready = "1";
    root._engine = eng;
  }

  function init() {
    renderStaticBoards(document);
    document.querySelectorAll(".ntsim").forEach(mountSim);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
  window.NTupleSim = { Engine, renderBoard, symmetries, TUPLES_SMALL };
})();
