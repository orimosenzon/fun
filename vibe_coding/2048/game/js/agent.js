// Agent: runs a trained network (exported by rl/export_weights.py) inside the browser
// and lets it play the game through window.game. Pure JS, no dependencies.
//
// Network (same as rl/common.py):
//   one-hot (16,4,4) -> Conv3x3(16->F, pad 1) -> ReLU -> Conv3x3(F->F, pad 1) -> ReLU
//   -> flatten (F*16, order c*16 + y*4 + x) -> Linear(F*16 -> H) -> ReLU -> head(s)
//   DQN head: Linear(H -> 4) = Q-values.   Actor-Critic: policy_head Linear(H -> 4), value_head Linear(H -> 1).

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

// Pure game rules on a 16-cell exponent board (same rules as GameManager), used for the valid-move mask.
const BoardRules = {
  slideRowLeft(row) {
    const tiles = row.filter((v) => v !== 0);
    const out = [];
    let i = 0;
    while (i < tiles.length) {
      if (i + 1 < tiles.length && tiles[i] === tiles[i + 1]) { out.push(tiles[i] + 1); i += 2; }
      else { out.push(tiles[i]); i += 1; }
    }
    while (out.length < 4) out.push(0);
    return out;
  },
  // direction: 0 up, 1 right, 2 down, 3 left. Returns true if the move changes the board.
  moveChanges(board, dir) {
    for (let line = 0; line < 4; line++) {
      const idx = [];
      for (let k = 0; k < 4; k++) {
        if (dir === 3) idx.push(line * 4 + k);          // left: row, left to right
        else if (dir === 1) idx.push(line * 4 + 3 - k); // right: row, right to left
        else if (dir === 0) idx.push(k * 4 + line);     // up: column, top to bottom
        else idx.push((3 - k) * 4 + line);              // down: column, bottom to top
      }
      const row = idx.map((i) => board[i]);
      const moved = BoardRules.slideRowLeft(row);
      for (let k = 0; k < 4; k++) if (moved[k] !== row[k]) return true;
    }
    return false;
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

// UI: the "let the agent play" panel
class AgentPlayer {
  constructor() {
    this.net = null;
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
    document.addEventListener("keydown", (e) => { if (!e.altKey && !e.ctrlKey && !e.metaKey && e.key.toLowerCase() === "p" && this.net) this.timer ? this.stop() : this.play(); });
  }

  load(name) {
    this.stop();
    this.net = null;
    if (!name) { this.status.textContent = ""; return; }
    const apply = () => {
      const payload = window.AGENT_WEIGHTS && window.AGENT_WEIGHTS[name];
      if (!payload) { this.status.textContent = "Weights not found"; return; }
      this.net = new AgentNetwork(payload);
      const ev = payload.eval;
      this.status.textContent = ev ? `Loaded. Trained on ${(payload.transitions / 1e6).toFixed(0)}M moves; eval mean score ${Math.round(ev.score_mean).toLocaleString()}.` : "Loaded.";
      this.showValues(null);
    };
    if (window.AGENT_WEIGHTS && window.AGENT_WEIGHTS[name]) return apply();
    this.status.textContent = "Loading weights…";
    const s = document.createElement("script");
    s.src = `weights/${name}.js`;
    s.onload = apply;
    s.onerror = () => { this.status.textContent = `Could not load weights/${name}.js (run rl/export_weights.py)`; };
    document.head.appendChild(s);
  }

  // one decision: read the board, pick the best valid move, play it
  step() {
    if (!this.net || !window.game) return false;
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
    this.showValues(out, valid, best);
    if (best < 0) { this.stop(); return false; }
    g.move(best);
    return true;
  }

  play() {
    if (!this.net) return;
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

  // show Q-values (DQN) or probabilities (actor-critic) for the four directions
  showValues(out, valid, best) {
    if (!out) { this.bars.forEach((b) => { b.style.setProperty("--w", "0%"); b.querySelector("span").textContent = ""; b.classList.remove("best"); }); return; }
    let vals = out.logits.slice();
    let label;
    if (this.net.kind === "dqn") {
      label = (v) => (v * 1000).toFixed(0);     // Q in game points (reward scale 1/1000)
      const lo = Math.min(...vals.filter((_, a) => valid[a])), hi = Math.max(...vals.filter((_, a) => valid[a]));
      var widths = vals.map((v, a) => (valid[a] ? (hi > lo ? 30 + 70 * (v - lo) / (hi - lo) : 100) : 0));
    } else {
      const m = Math.max(...vals.filter((_, a) => valid[a]));
      const ex = vals.map((v, a) => (valid[a] ? Math.exp(v - m) : 0));
      const z = ex.reduce((s, v) => s + v, 0);
      vals = ex.map((v) => v / z);
      label = (v) => (100 * v).toFixed(0) + "%";
      var widths = vals.map((v) => 100 * v);
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
