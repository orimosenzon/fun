// fight.js: לוגיקת הקרב. קוד טהור.
// הפגיעה נקבעת גיאומטרית: נקודת הפגיעה של האגרוף או הרגל מול ראש וגו היריב.

import { DIM, solve, clonePose, mirrorPose, shiftPose, lerpPose, sub, len, distToSeg } from './body.js';
import { TECH, KAMAE, STAND, poseAt, techFor } from './techniques.js';

const KAMAE_M = mirrorPose(KAMAE);
export const ARENA = 4.2;       // חצי אורך הזירה
const MIN_GAP = 0.44;           // מרחק מינימלי בין האגנים
const BUFFER = 0.22;            // כמה זמן פקודה מחכה לסוף הטכניקה הנוכחית

export class Fighter {
  constructor(idx, x) {
    this.idx = idx;
    this.dir = idx === 0 ? 1 : -1;
    this.reset(x);
  }
  reset(x) {
    this.x = x; this.mir = false; this.hp = 100;
    this.cur = clonePose(STAND);
    this.tech = null; this.t = 0; this.from = null;
    this.hitDone = false; this.hold = false;
    this.hitstop = 0; this.slow = 0; this.kb = 0;
    this.buf = null; this.bufAge = 0;
    this.last = null; this.lastAge = 9;     // הטכניקה האחרונה, לתצוגה
    this.blockedAt = -9; this.techSeq = 0;
    this.kimeAge = 9; this.kimeNow = false;
    this.J = solve(this.cur);
  }
  get kamae() { return this.mir ? KAMAE_M : KAMAE; }
  get busy() { return !!this.tech; }
  get down() { return this.tech && this.tech.kind === 'ko'; }

  start(tech, scale = 1) {
    this.tech = tech; this.t = 0; this.scale = scale;
    this.from = clonePose(this.cur);
    this.hitDone = false; this.hold = false; this.techSeq++;
    if (tech.name && tech.kind !== 'cine') { this.last = tech; this.lastAge = 0; }
  }

  // נקודה מקומית -> עולם
  world(p) { return [this.x + this.dir * p[0], p[1], this.dir * p[2]]; }

  update(dt, inp, canAct) {
    this.lastAge += dt; this.kimeAge += dt;
    if (this.kb) {
      const s = this.kb * Math.min(1, dt * 14);
      this.x -= this.dir * s; this.kb -= s;
      if (Math.abs(this.kb) < 0.002) this.kb = 0;
    }
    if (this.hitstop > 0) { this.hitstop -= dt; return; }

    // קליטת פקודות
    if (inp && canAct) {
      for (const a of inp.actions) { this.buf = { a, level: inp.level }; this.bufAge = 0; }
      this.hold = !!inp.hold;
    }
    if (this.buf) { this.bufAge += dt; if (this.bufAge > BUFFER) this.buf = null; }

    if (this.tech) {
      let d = dt / (this.scale || 1);
      if (this.slow > 0) { this.slow -= dt; d *= 0.4; }
      const tk = this.tech;
      const nt = this.t + d;
      // החזקת חסימה כל עוד המקש לחוץ
      const tPrev = this.t;
      if (tk.hold && this.hold && this.t <= tk.hold && nt > tk.hold) this.t = tk.hold;
      else this.t = nt;
      for (const k of tk.keys) if (k.kime && tPrev < k.t && this.t >= k.t) { this.kimeAge = 0; this.kimeNow = true; }
      // חסימה ששוחררה מתקצרת, כדי לאפשר התקפת נגד מהירה
      if (tk.kind === 'block' && !this.hold && this.t > 0.22 && this.t < 0.42) this.t = 0.42;
      if (this.t >= tk.dur) {
        if (tk.stay) { this.t = tk.dur; this.cur = poseAt(tk, tk.dur, this.from, this.mir, this.cur); }
        else this.finish();
      } else {
        this.cur = poseAt(tk, this.t, this.from, this.mir, this.cur);
      }
    }
    if (!this.tech && canAct) {
      // מצב חופשי: פקודה ממתינה, תנועה, או חזרה לקמאה
      if (this.buf) {
        const tk = techFor(this.buf.a, this.buf.level);
        this.buf = null;
        if (tk) { this.start(tk); return; }
      }
      if (inp && inp.move) { this.start(inp.move > 0 ? TECH.stepF : TECH.stepB); return; }
    }
    if (!this.tech) this.cur = lerpPose(this.cur, this.kamae, Math.min(1, dt * 14), this.cur);
  }

  finish() {
    const tk = this.tech;
    this.x += this.dir * tk.advance;
    if (tk.sw) this.mir = !this.mir;
    this.tech = null;
    this.cur = clonePose(this.kamae);
  }

  // לוחם שנקטע באמצע התקדמות: מעבירים את נקודת המוצא אל מתחת לגוף כדי שלא "יוחזר" אחורה
  rebase() {
    const dx = this.cur.pel[0];
    if (Math.abs(dx) < 0.05) return;
    this.x += this.dir * dx;
    this.cur = shiftPose(this.cur, -dx);
  }

  solve() { this.J = solve(this.cur); return this.J; }

  // האם חוסם כרגע התקפה מהסוג הזה
  blocks(cls) {
    const tk = this.tech;
    if (!tk || tk.kind !== 'block') return false;
    if (this.t < tk.block.from || this.t > tk.block.to) return false;
    return tk.block.covers.includes(cls);
  }
  // נמצא בתוך התקפה שעוד לא הגיעה ליעדה
  get attacking() { return this.tech && this.tech.kind === 'attack' && !this.hitDone; }
  get hitLimb() {
    const l = this.tech.hit.limb;
    if (!this.mir) return l;
    return l.endsWith('L') ? l.slice(0, -1) + 'R' : l.slice(0, -1) + 'L';
  }
}

export class Match {
  constructor() {
    this.f = [new Fighter(0, -1.2), new Fighter(1, 1.2)];
    this.events = [];
    this.restart();
  }
  restart() {
    this.f[0].reset(-1.2); this.f[1].reset(1.2);
    this.phase = 'rei'; this.time = 0; this.winner = -1; this.endT = 0;
    for (const f of this.f) f.start(TECH.rei);
    this.events.push({ type: 'rei' });
  }
  get dist() { return this.f[1].x - this.f[0].x; }

  step(dt, inputs) {
    this.time += dt;
    const [a, b] = this.f;
    const fight = this.phase === 'fight';
    a.update(dt, inputs[0], fight);
    b.update(dt, inputs[1], fight);

    if (this.phase === 'rei' && !a.tech && !b.tech) {
      this.phase = 'fight';
      this.events.push({ type: 'hajime' });
    }
    a.solve(); b.solve();
    for (const f of this.f) if (f.kimeNow) { f.kimeNow = false; this.events.push({ type: 'kime', who: f.idx, kind: f.tech ? f.tech.kind : '', id: f.tech ? f.tech.id : '' }); }

    if (fight) {
      const ha = this.contact(a, b), hb = this.contact(b, a);
      if (ha) this.resolve(a, b, ha);
      if (hb) this.resolve(b, a, hb);
      if (this.phase === 'fight' && (a.hp <= 0 || b.hp <= 0)) {
        this.winner = a.hp <= 0 && b.hp <= 0 ? -1 : a.hp <= 0 ? 1 : 0;
        this.phase = 'ko'; this.endT = this.time;
        this.events.push({ type: 'ko', winner: this.winner });
      }
    }
    if (this.phase === 'ko' && this.time - this.endT > 1.4) {
      this.phase = 'done';
      for (const f of this.f) if (f.hp > 0) { f.rebase(); f.start(TECH.yame); }
      this.events.push({ type: 'yame', winner: this.winner });
    }
    this.separate();
  }

  // האם נקודת הפגיעה של התוקף נגעה בגוף היריב
  contact(att, def) {
    const tk = att.tech;
    if (!tk || !tk.hit || att.hitDone || def.down) return null;
    if (att.t < tk.hit.from || att.t > tk.hit.to) return null;
    const s = att.world(att.J[att.hitLimb]);
    const head = def.world(def.J.head);
    if (len(sub(s, head)) < DIM.headR + 0.07) return { at: s, zone: 'J' };
    const p0 = def.world(def.J.pel), p1 = def.world(def.J.chest);
    if (distToSeg(s, p0, p1) < DIM.torsoR + 0.05) return { at: s, zone: s[1] > p1[1] + 0.06 ? 'J' : s[1] > p0[1] + 0.22 ? 'C' : 'G' };
    return null;
  }

  resolve(att, def, c) {
    const tk = att.tech, h = tk.hit;
    if (h.cls === 'J') c.zone = 'J';   // התקפת ג'ודאן שנגעה בדרך למעלה נחשבת לפי היעד
    att.hitDone = true;
    if (def.blocks(h.cls)) {
      att.slow = 0.28; att.hitstop = 0.04;
      def.kb += 0.06; def.blockedAt = this.time;
      this.events.push({ type: 'block', who: def.idx, at: c.at, tech: def.tech.id });
      return;
    }
    const counter = def.attacking;           // דה-אי: פגיעה ביריב בזמן שהוא תוקף
    const dmg = Math.round(h.dmg * (counter ? 1.5 : 1));
    def.hp = Math.max(0, def.hp - dmg);
    att.hitstop = 0.09;
    def.rebase();
    def.kb += 0.10 + dmg * 0.012;
    def.buf = null;
    if (def.hp <= 0) def.start(TECH.ko);
    else def.start(c.zone === 'J' ? TECH.hitJ : TECH.hitC, Math.max(0.8, Math.min(1.5, dmg / 12)));
    this.events.push({ type: 'hit', who: att.idx, at: c.at, dmg, counter, zone: c.zone, tech: tk.id, kiai: dmg >= 14 });
  }

  // גופים לא עוברים זה דרך זה, ואין יציאה מהזירה
  separate() {
    const [a, b] = this.f;
    if (!a.down && !b.down) {
      const pa = a.x + a.cur.pel[0], pb = b.x - b.cur.pel[0];
      const over = MIN_GAP - (pb - pa);
      if (over > 0) {
        const am = a.tech && a.tech.advance > 0, bm = b.tech && b.tech.advance > 0;
        const wa = am && !bm ? 0 : bm && !am ? 1 : 0.5;
        a.x -= over * wa; b.x += over * (1 - wa);
      }
    }
    for (const f of this.f) {
      if (f.x < -ARENA) { const o = -ARENA - f.x; f.x += o; if (f === a && b.x - a.x < 0.5) b.x += o; }
      if (f.x > ARENA) { const o = f.x - ARENA; f.x -= o; if (f === b && b.x - a.x < 0.5) a.x -= o; }
    }
  }
}
