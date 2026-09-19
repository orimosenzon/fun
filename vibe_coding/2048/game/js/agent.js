// Agent: runs a trained network (exported by rl/export_weights.py) inside the browser
// and lets it play the game through window.game. Pure JS, no dependencies.
//
// Network (same as rl/common.py):
//   one-hot (16,4,4) -> Conv3x3(16->F, pad 1) -> ReLU -> Conv3x3(F->F, pad 1) -> ReLU
//   -> flatten (F*16, order c*16 + y*4 + x) -> Linear(F*16 -> H) -> ReLU -> head(s)
//   DQN head: Linear(H -> 4) = Q-values.   Actor-Critic and PPO: policy_head Linear(H -> 4), value_head Linear(H -> 1).
//   Afterstate TD ("asnet"): value_head only; the input is the board *after* a slide, and the decision is
//   argmax over r + V(afterstate) like the n-tuple agent.

class AgentNetwork {
  constructor(payload) {
    this.kind = payload.kind;
    this.F = payload.filters;
    this.H = payload.hidden;
    this.t = {};
    for (const [name, spec] of Object.entries(payload.tensors)) {
      this.t[name] = AgentNetwork.decodeF16(spec.data);
    }
    this.headName = this.kind === "dqn" ? "head" : "policy_head";
    this.rewardScale = payload.reward_scale || 1e-3;
  }

  // base64 float16 -> Float32Array
  static decodeF16(b64) {
    const bin = atob(b64);
    const n = bin.length / 2;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
      const s = (h & 0x8000) ? -1 : 1;
      const e = (h >> 10) & 0x1f;
      const f = h & 0x3ff;
      let v;
      if (e === 0) v = s * Math.pow(2, -14) * (f / 1024);
      else if (e === 0x1f) v = f ? NaN : s * Infinity;
      else v = s * Math.pow(2, e - 15) * (1 + f / 1024);
      out[i] = v;
    }
    return out;
  }

  // board: 16 exponents, row-major (index = y*4 + x), 0 = empty
  forward(board) {
    if (this.kind === "asnet") return this.forwardAfterstates(board);
    const F = this.F;
    // one-hot input: x[c][y][x]
    const inp = new Float32Array(16 * 16);
    for (let i = 0; i < 16; i++) inp[board[i] * 16 + i] = 1;
    const h1 = AgentNetwork.conv3x3(inp, 16, this.t["trunk.conv1.weight"], this.t["trunk.conv1.bias"], F);
    const h2 = AgentNetwork.conv3x3(h1, F, this.t["trunk.conv2.weight"], this.t["trunk.conv2.bias"], F);
    const fc = AgentNetwork.linear(h2, this.t["trunk.fc.weight"], this.t["trunk.fc.bias"], this.H, true);
    const logits = AgentNetwork.linear(fc, this.t[this.headName + ".weight"], this.t[this.headName + ".bias"], 4, false);
    const out = { logits: Array.from(logits) };
    if (this.kind !== "dqn") {
      out.value = AgentNetwork.linear(fc, this.t["value_head.weight"], this.t["value_head.bias"], 1, false)[0];
    }
    return out;
  }

  // trunk only, up to the hidden vector
  features(board) {
    const inp = new Float32Array(16 * 16);
    for (let i = 0; i < 16; i++) inp[board[i] * 16 + i] = 1;
    const h1 = AgentNetwork.conv3x3(inp, 16, this.t["trunk.conv1.weight"], this.t["trunk.conv1.bias"], this.F);
    const h2 = AgentNetwork.conv3x3(h1, this.F, this.t["trunk.conv2.weight"], this.t["trunk.conv2.bias"], this.F);
    return AgentNetwork.linear(h2, this.t["trunk.fc.weight"], this.t["trunk.fc.bias"], this.H, true);
  }

  // afterstate value network: four slides, V of each resulting board, in game points (the net learned on score/1000)
  forwardAfterstates(board) {
    const logits = [];
    for (let a = 0; a < 4; a++) {
      const after = BoardRules.afterstate(board, a);
      if (!after.changed) { logits.push(-Infinity); continue; }
      const v = AgentNetwork.linear(this.features(after.board), this.t["value_head.weight"], this.t["value_head.bias"], 1, false)[0];
      logits.push(after.score + v / this.rewardScale);
    }
    return { logits };
  }

  // input: Cin*16 (c, y, x); weight: (Cout, Cin, 3, 3); output Cout*16 with ReLU
  static conv3x3(inp, Cin, w, b, Cout) {
    const out = new Float32Array(Cout * 16);
    for (let o = 0; o < Cout; o++) {
      for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 4; x++) {
          let acc = b[o];
          for (let c = 0; c < Cin; c++) {
            const wBase = ((o * Cin + c) * 3) * 3;
            const iBase = c * 16;
            for (let ky = 0; ky < 3; ky++) {
              const yy = y + ky - 1;
              if (yy < 0 || yy > 3) continue;
              for (let kx = 0; kx < 3; kx++) {
                const xx = x + kx - 1;
                if (xx < 0 || xx > 3) continue;
                acc += inp[iBase + yy * 4 + xx] * w[wBase + ky * 3 + kx];
              }
            }
          }
          out[o * 16 + y * 4 + x] = acc > 0 ? acc : 0;
        }
      }
    }
    return out;
  }

  static linear(inp, w, b, nOut, relu) {
    const nIn = inp.length;
    const out = new Float32Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let acc = b[o];
      const base = o * nIn;
      for (let i = 0; i < nIn; i++) acc += inp[i] * w[base + i];
      out[o] = relu && acc < 0 ? 0 : acc;
    }
    return out;
  }
}

// N-tuple network (same as rl/ntuple_td.py, the "small" configuration): V(board) is a sum of table
// lookups, one per (window, symmetry). Each window is a fixed list of cells; its contents (exponents)
// are read as a base-16 number, which indexes that window's table. No neurons, no multiplications.
class NTupleNetwork {
  constructor(payload) {
    this.kind = "ntuple";
    this.tuples = payload.tuples;
    this.n = this.tuples[0].length;
    this.tableSize = 16 ** this.n;
    this.table = NTupleNetwork.decodeF32(payload.table.data);
    // symmetric sampling: every window is read at the 8 rotations/reflections of the board
    this.feats = [];
    this.offsets = [];
    for (const sym of NTupleNetwork.symmetries()) {
      this.tuples.forEach((cells, j) => {
        this.feats.push(cells.map((c) => sym[c]));
        this.offsets.push(j * this.tableSize);
      });
    }
  }

  // base64 float32 (little endian) -> Float32Array
  static decodeF32(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  // The 8 symmetries of the square as cell permutations, in the order of rl/ntuple_td.py:
  // for k in 0..3: rot90^k, then its left-right mirror. perm[i] = which original cell lands at position i.
  static symmetries() {
    let m = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]];
    const rot90 = (a) => a[0].map((_, i) => a.map((row) => row[3 - i]));   // numpy.rot90 (counter-clockwise)
    const fliplr = (a) => a.map((row) => row.slice().reverse());
    const out = [];
    for (let k = 0; k < 4; k++) {
      out.push(m.flat());
      out.push(fliplr(m).flat());
      m = rot90(m);
    }
    return out;
  }

  // table index of feature f on a board: the window's exponents read as a base-16 number
  index(board, f) {
    const cells = this.feats[f];
    let idx = 0;
    for (let j = 0; j < cells.length; j++) idx = idx * 16 + board[cells[j]];
    return this.offsets[f] + idx;
  }

  value(board) {
    let s = 0;
    for (let f = 0; f < this.feats.length; f++) s += this.table[this.index(board, f)];
    return s;
  }

  // one decision: four deterministic slides, then r + V(afterstate) for each; -Infinity for a move that changes nothing
  forward(board) {
    const logits = [];
    for (let a = 0; a < 4; a++) {
      const after = BoardRules.afterstate(board, a);
      logits.push(after.changed ? after.score + this.value(after.board) : -Infinity);
    }
    return { logits };
  }
}

// Pure game rules on a 16-cell exponent board (same rules as GameManager), used for the valid-move mask
// and for the n-tuple agent's afterstates.
const BoardRules = {
  // slide one line towards index 0, with the original merge rules. Returns {row, score}.
  slideRowLeft(row) {
    const tiles = row.filter((v) => v !== 0);
    const out = [];
    let score = 0;
    let i = 0;
    while (i < tiles.length) {
      if (i + 1 < tiles.length && tiles[i] === tiles[i + 1]) { out.push(tiles[i] + 1); score += 2 ** (tiles[i] + 1); i += 2; }
      else { out.push(tiles[i]); i += 1; }
    }
    while (out.length < 4) out.push(0);
    return { row: out, score };
  },
  // cell indices of line `line`, read in the direction of the move (first cell = where tiles pile up)
  lineIndices(dir, line) {
    const idx = [];
    for (let k = 0; k < 4; k++) {
      if (dir === 3) idx.push(line * 4 + k);          // left: row, left to right
      else if (dir === 1) idx.push(line * 4 + 3 - k); // right: row, right to left
      else if (dir === 0) idx.push(k * 4 + line);     // up: column, top to bottom
      else idx.push((3 - k) * 4 + line);              // down: column, bottom to top
    }
    return idx;
  },
  // direction: 0 up, 1 right, 2 down, 3 left. The board after the slide, before the random tile.
  afterstate(board, dir) {
    const out = board.slice();
    let score = 0, changed = false;
    for (let line = 0; line < 4; line++) {
      const idx = BoardRules.lineIndices(dir, line);
      const row = idx.map((i) => board[i]);
      const moved = BoardRules.slideRowLeft(row);
      score += moved.score;
      for (let k = 0; k < 4; k++) {
        if (moved.row[k] !== row[k]) changed = true;
        out[idx[k]] = moved.row[k];
      }
    }
    return { board: out, score, changed };
  },
  moveChanges(board, dir) {
    return BoardRules.afterstate(board, dir).changed;
  },
  validMoves(board) {
    return [0, 1, 2, 3].map((d) => BoardRules.moveChanges(board, d));
  },
  fromGrid(grid) {
    const board = new Array(16).fill(0);
    grid.eachCell((x, y, tile) => { if (tile) board[y * 4 + x] = Math.round(Math.log2(tile.value)); });
    return board;
  },
};

// Recorded games of the big n-tuple table (4 x 6-tuples, 268MB, stays in Python): rl/export_replays.py writes
// game/weights/ntuple_replays.js. Each move is two characters: the action (0-3) and the tile that then
// spawned, as a base-32 digit (cell index + 16 if the tile was a 4). The board is replayed with the real
// game rules, so the score and the animations are the game's own.
const REPLAY_DIGITS = "0123456789abcdefghijklmnopqrstuv";

// UI: the "let the agent play" panel
class AgentPlayer {
  constructor() {
    this.net = null;          // a network that decides (AgentNetwork or NTupleNetwork)
    this.replay = null;       // or a recorded game being replayed: {games, game, pos}
    this.timer = null;
    this.panel = document.querySelector(".agent-panel");
    this.select = this.panel.querySelector(".agent-select");
    this.playBtn = this.panel.querySelector(".agent-play");
    this.stepBtn = this.panel.querySelector(".agent-step");
    this.speed = this.panel.querySelector(".agent-speed");
    this.auto = this.panel.querySelector(".agent-auto");
    this.status = this.panel.querySelector(".agent-status");
    this.bars = Array.from(this.panel.querySelectorAll(".agent-bar"));
    this.select.addEventListener("change", () => this.load(this.select.value));
    this.playBtn.addEventListener("click", () => (this.timer ? this.stop() : this.play()));
    this.stepBtn.addEventListener("click", () => { this.stop(); this.step(); });
    document.addEventListener("keydown", (e) => { if (!e.altKey && !e.ctrlKey && !e.metaKey && e.key.toLowerCase() === "p" && this.ready()) this.timer ? this.stop() : this.play(); });
  }

  ready() { return !!(this.net || this.replay); }

  load(name) {
    this.stop();
    this.net = null;
    this.replay = null;
    this.showValues(null);
    if (!name) { this.status.textContent = ""; return; }
    if (name === "ntuple_replay") return this.loadScript("weights/ntuple_replays.js", () => window.AGENT_REPLAYS, (rec) => {
      this.replay = { games: rec.games, game: null, pos: 0 };
      this.startReplay(0);
    });
    this.loadScript(`weights/${name}.js`, () => window.AGENT_WEIGHTS && window.AGENT_WEIGHTS[name], (payload) => {
      this.net = payload.kind === "ntuple" ? new NTupleNetwork(payload) : new AgentNetwork(payload);
      const ev = payload.eval;
      const moves = payload.transitions ? `${(payload.transitions / 1e6).toFixed(0)}M moves` : "";
      let what = "";
      if (payload.kind === "asnet") what = "Afterstate TD with the convolutional network (the n-tuple algorithm, the neural representation). ";
      if (payload.kind === "ntuple") {
        what = `Browser-sized n-tuple network: ${payload.tuples.length} windows of ${payload.tuples[0].length} cells, ${(payload.table.size / 1e3).toFixed(0)}K table entries (${(4 * payload.table.size / 1e6).toFixed(1)}MB). `;
      }
      this.status.textContent = what + (ev ? `Trained on ${moves}; eval mean score ${Math.round(ev.score_mean).toLocaleString()}` +
        (ev.reach_2048 != null ? `, reaches 2048 in ${Math.round(100 * ev.reach_2048)}% of games.` : ".") : "Loaded.");
    });
  }

  // load a weights/replays script once, then call apply with the object it defines
  loadScript(src, getter, apply) {
    const have = getter();
    if (have) return apply(have);
    this.status.textContent = "Loading…";
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => { const obj = getter(); if (obj) apply(obj); else this.status.textContent = `${src} did not define its data`; };
    s.onerror = () => { this.status.textContent = `Could not load ${src} (run rl/export_weights.py or rl/export_replays.py)`; };
    document.head.appendChild(s);
  }

  // ---- replay of a recorded game ----
  startReplay(index) {
    const r = this.replay;
    r.game = r.games[index];
    r.pos = 0;
    window.game.loadBoard(r.game.start);
    this.status.textContent = `Replaying a game recorded in Python by the full 4×6 n-tuple table (268MB): ` +
      `${r.game.score.toLocaleString()} points, ${r.game.moves.toLocaleString()} moves, best tile ${r.game.max_tile.toLocaleString()}. ` +
      `Game ${index + 1} of ${r.games.length}.`;
  }

  replayStep() {
    const r = this.replay, g = window.game;
    if (g.won && !g.keepPlayingFlag) g.keepPlaying();
    if (r.pos >= r.game.moves) {
      if (this.auto.checked) { this.startReplay((r.games.indexOf(r.game) + 1) % r.games.length); return true; }
      this.stop();
      return false;
    }
    const action = +r.game.steps[2 * r.pos];
    const spawn = REPLAY_DIGITS.indexOf(r.game.steps[2 * r.pos + 1]);
    const cell = spawn & 15;
    g.forcedTile = { x: cell % 4, y: Math.floor(cell / 4), value: spawn & 16 ? 4 : 2 };
    const board = BoardRules.fromGrid(g.grid);
    const valid = BoardRules.validMoves(board);
    this.showValues({ logits: valid.map((v, a) => (a === action ? 1 : 0)) }, valid, action, "replay");
    if (!g.move(action)) { this.status.textContent = "Replay out of sync with the board (this should not happen)."; this.stop(); return false; }
    r.pos += 1;
    return true;
  }

  // one decision: read the board, pick the best valid move, play it
  step() {
    if (!this.ready() || !window.game) return false;
    if (this.replay) return this.replayStep();
    const g = window.game;
    if (g.won && !g.keepPlayingFlag) g.keepPlaying();
    if (g.over) {
      if (this.auto.checked) { g.restart(); return true; }
      this.stop();
      return false;
    }
    const board = BoardRules.fromGrid(g.grid);
    const valid = BoardRules.validMoves(board);
    const out = this.net.forward(board);
    let best = -1, bestV = -Infinity;
    for (let a = 0; a < 4; a++) if (valid[a] && out.logits[a] > bestV) { bestV = out.logits[a]; best = a; }
    this.showValues(out, valid, best, this.net.kind);
    if (best < 0) { this.stop(); return false; }
    g.move(best);
    return true;
  }

  play() {
    if (!this.ready()) return;
    this.playBtn.textContent = "Pause";
    // slider: right = faster. delay in ms between moves
    const tick = () => { if (!this.step()) return; this.timer = setTimeout(tick, 620 - +this.speed.value); };
    this.timer = setTimeout(tick, 0);
  }

  stop() {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.playBtn.textContent = "Play";
  }

  // show what the agent thinks of the four directions: Q-values (DQN), probabilities (actor-critic, PPO),
  // "immediate score + value of the board after the slide" (n-tuple), or just the recorded move (replay)
  showValues(out, valid, best, kind) {
    if (!out) { this.bars.forEach((b) => { b.style.setProperty("--w", "0%"); b.querySelector("span").textContent = ""; b.classList.remove("best"); }); return; }
    let vals = out.logits.slice();
    let label, widths;
    if (kind === "replay") {
      label = (v) => (v ? "recorded" : "");
      widths = vals.map((v) => (v ? 100 : 0));
    } else if (kind === "dqn" || kind === "ntuple" || kind === "asnet") {
      const scale = kind === "dqn" ? 1000 : 1;    // DQN learned on score/1000; the n-tuple and afterstate values are already in points
      label = (v) => (v * scale).toFixed(0);
      const lo = Math.min(...vals.filter((_, a) => valid[a])), hi = Math.max(...vals.filter((_, a) => valid[a]));
      widths = vals.map((v, a) => (valid[a] ? (hi > lo ? 30 + 70 * (v - lo) / (hi - lo) : 100) : 0));
    } else {
      const m = Math.max(...vals.filter((_, a) => valid[a]));
      const ex = vals.map((v, a) => (valid[a] ? Math.exp(v - m) : 0));
      const z = ex.reduce((s, v) => s + v, 0);
      vals = ex.map((v) => v / z);
      label = (v) => (100 * v).toFixed(0) + "%";
      widths = vals.map((v) => 100 * v);
    }
    this.bars.forEach((b, a) => {
      b.style.setProperty("--w", widths[a] + "%");
      b.querySelector("span").textContent = valid[a] ? label(vals[a]) : "×";
      b.classList.toggle("best", a === best);
    });
  }
}

window.addEventListener("load", () => {
  if (document.querySelector(".agent-panel")) window.agentPlayer = new AgentPlayer();
});
