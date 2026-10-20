// כפות הידיים: "סינרגיות" אחיזה. במקום לשלוט ב-20 מפרקים בנפרד, בוחרים צורת אחיזה ורמת סגירה,
// וכל מפרק מקבל יעד. המפעילים מוגבלים בכוח, כך שאצבע שפוגשת חפץ נעצרת ולוחצת עליו (כמו גיד מתוח).
import { FINGERS } from '../robot/spec.js';

// לכל צורה: יעד לכל מפרק כשהיד סגורה לגמרי. [abd, mcp, pip, dip]
export const GRASPS = {
  open:   { name: 'פתוחה', thumb: [0.15, 0.0, 0.05, 0.05], fingers: [0, 0.05, 0.08, 0.06] },
  power:  { name: 'אחיזת כוח', thumb: [1.2, 0.55, 0.6, 0.5], fingers: [0, 1.35, 1.45, 1.0] },
  cylinder: { name: 'אחיזה גלילית', thumb: [1.25, 0.45, 0.4, 0.35], fingers: [0, 1.05, 1.15, 0.8] },
  // עטיפה: כף היד צמודה לגליל, ארבע האצבעות נכרכות סביבו, והאגודל נח מעליו בלי להיתפס בשפה
  wrap:   { name: 'עטיפה', thumb: [0.05, -0.15, 0.05, 0.05], fingers: [0, 1.5, 1.6, 1.15] },
  pinch:  { name: 'צביטה', thumb: [1.15, 0.48, 0.42, 0.3], fingers: [0, 0.9, 0.75, 0.45], only: ['index', 'middle'], rest: [0, 1.4, 1.5, 1.0] },
  point:  { name: 'הצבעה', thumb: [1.0, 0.5, 0.7, 0.6], fingers: [0, 1.5, 1.6, 1.1], except: { index: [0, 0, 0.05, 0.05] } },
  peace:  { name: 'וי', thumb: [1.1, 0.5, 0.7, 0.6], fingers: [0, 1.5, 1.6, 1.1], except: { index: [-0.2, 0, 0.05, 0.05], middle: [0.2, 0, 0.05, 0.05] } },
  flat:   { name: 'שטוחה', thumb: [0.0, 0.0, 0.0, 0.0], fingers: [0, 0, 0, 0] },
  hook:   { name: 'וו', thumb: [0.2, 0.1, 0.1, 0.1], fingers: [0, 0.25, 1.5, 1.1] },
  spread: { name: 'פרושה', thumb: [0.0, -0.2, 0.0, 0.0], fingers: [0, -0.1, 0.0, 0.0], abd: true },
};
const ABD = { index: -0.22, middle: -0.05, ring: 0.12, little: 0.28 };

export class Hands {
  constructor(world) {
    this.w = world;
    this.state = {};
    for (const s of ['l', 'r']) {
      this.state[s] = { shape: 'open', close: 0, target: 0, speed: 3.0, act: {} };
      for (const f of FINGERS) for (const k of ['abd', 'mcp', 'pip', 'dip']) {
        this.state[s].act[`${f.key}_${k}`] = world.id('mjOBJ_ACTUATOR', `${s}_${f.key}_${k}`);
      }
    }
  }
  // מעבר בין צורות נעשה בהדרגה (0.3 ש'), מהיעדים האחרונים אל החדשים. בלי זה האצבעות "קופצות" ומעיפות חפצים
  set(side, shape, close = 1, speed = 3) {
    const st = this.state[side];
    if (st.shape !== shape) { st.from = st.last ? { ...st.last } : null; st.blend = 0; }
    st.shape = shape; st.target = close; st.speed = Math.max(0.5, speed);
  }
  release(side, T = 0.5) { const st = this.state[side]; st.target = 0; st.speed = 1 / Math.max(0.1, T); }
  control() {
    const ctrl = this.w.data.ctrl, dt = 0.002;
    for (const s of ['l', 'r']) {
      const st = this.state[s];
      st.close += Math.sign(st.target - st.close) * Math.min(Math.abs(st.target - st.close), st.speed * dt);
      if (st.blend < 1) st.blend = Math.min(1, (st.blend || 0) + dt / 0.3);
      const G = GRASPS[st.shape] || GRASPS.open, O = GRASPS.open;
      st.last = st.last || {};
      for (const f of FINGERS) {
        let full;
        if (f.key === 'thumb') full = G.thumb;
        else if (G.except?.[f.key]) full = G.except[f.key];
        else if (G.only && !G.only.includes(f.key)) full = G.rest;
        else full = G.fingers;
        const open = f.key === 'thumb' ? O.thumb : O.fingers;
        const k4 = ['abd', 'mcp', 'pip', 'dip'];
        for (let i = 0; i < 4; i++) {
          let q = open[i] + (full[i] - open[i]) * st.close;
          if (i === 0 && f.key !== 'thumb') q = (G.abd ? 1.3 : 0.25 * (1 - st.close)) * ABD[f.key] + (full[0] || 0);
          const key = `${f.key}_${k4[i]}`;
          if (st.from && st.blend < 1 && st.from[key] != null) q = st.from[key] + (q - st.from[key]) * st.blend;
          st.last[key] = q;
          ctrl[st.act[key]] = q;
        }
      }
    }
  }
}
