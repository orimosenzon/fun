// בקר כל-הגוף (Whole-Body Control): מתרגם משימות (מרכז מסה, אגן, רגליים, ידיים, תנוחה) למומנטים במפרקים.
//
// שלושה שלבים בכל צעד בקרה (500 פעם בשנייה):
// 1. מוצאים תאוצות מפרקים q̈ שמקיימות הכי טוב את כל המשימות: מינימום של Σ wᵢ‖Jᵢq̈ + J̇ᵢq̇ − aᵢ‖².
// 2. מחלקים את העומס בין כפות הרגליים: שש המשוואות של הבסיס הצף (שאין לו מנוע) קובעות
//    כמה כוח ומומנט הרצפה צריכה לתת. אחר כך חותכים לגבולות הפיזיקליים: הרצפה רק דוחפת,
//    החיכוך מוגבל, ומרכז הלחץ חייב להישאר בתוך הסוליה.
// 3. המומנטים: τ = M q̈ + h − Jᶜᵀ f, ונחתכים לגבולות המנוע (מומנט-מהירות).
import { cholFactor, cholSolveF, quatToMat, rotError, matVec3, matT3, clamp } from './linalg.js';
import { JOINTS } from '../robot/spec.js';
import { G } from '../robot/spec.js';
import { torqueLimit } from './actuators.js';

// משקלות האילוצים "הקשים": רגל עומדת לא זזה, ומשוואות התנועה של הבסיס. שניהם באותו סדר גודל,
// הרבה מעל משקלות המשימות, כך שהמשימות מתפשרות ביניהן רק בתוך מה שהפיזיקה מרשה.
const STANCE_W = 2e6, RHO = 600;

export class WBC {
  constructor(world) {
    const { mj, model: m } = world;
    this.w = world; this.mj = mj; this.m = m;
    const nv = this.nv = m.nv;
    // הדרגות של הגוף: 6 של הבסיס + 30 המפרקים של הגוף (בלי אצבעות)
    this.joints = JOINTS.map(j => {
      const jid = world.jointId(j.name);
      return { ...j, jid, dof: m.jnt_dofadr[jid], qadr: m.jnt_qposadr[jid], act: world.id('mjOBJ_ACTUATOR', j.name) };
    });
    this.B = [0, 1, 2, 3, 4, 5, ...this.joints.map(j => j.dof)];
    const nB = this.nB = this.B.length;
    this.ids = {
      pelvis: world.bodyId('pelvis'), torso: world.bodyId('torso'), head: world.bodyId('head'),
      l_foot: world.bodyId('l_foot'), r_foot: world.bodyId('r_foot'),
      l_hand: world.bodyId('l_hand'), r_hand: world.bodyId('r_hand'),
      l_sole: world.siteId('l_sole'), r_sole: world.siteId('r_sole'),
      l_grip: world.siteId('l_grip'), r_grip: world.siteId('r_grip'),
    };
    const DB = mj.DoubleBuffer;
    this.bufP = new DB(3 * nv); this.bufR = new DB(3 * nv);
    this.bufDP = new DB(3 * nv); this.bufDR = new DB(3 * nv);
    this.bufM = new DB(nv * nv);
    this.H = new Float64Array(nB * nB); this.g = new Float64Array(nB);
    this.qdd = new Float64Array(nB);
    this.tau = new Float64Array(this.joints.length);
    this.out = {};   // נתונים לתצוגה
  }

  // ---------- גישה למצב ----------
  sitePos(id) { const d = this.w.data.site_xpos; return [d[3 * id], d[3 * id + 1], d[3 * id + 2]]; }
  bodyMat(id) { return Array.from(this.w.data.xmat.subarray(9 * id, 9 * id + 9)); }
  com() { const s = this.w.data.subtree_com; const i = this.ids.pelvis; return [s[3 * i], s[3 * i + 1], s[3 * i + 2]]; }

  // יעקוביאן של נקודה על גוף: מחזיר [Jp, Jr] כמערכי Float64Array של 3×nB, ואת J̇q̇ (6)
  jac(body, point, withDot = true) {
    const { mj, m } = this, d = this.w.data, nv = this.nv, nB = this.nB, B = this.B;
    mj.mj_jac(m, d, this.bufP, this.bufR, point, body);
    const P = this.bufP.GetView(), R = this.bufR.GetView();
    const Jp = new Float64Array(3 * nB), Jr = new Float64Array(3 * nB);
    for (let r = 0; r < 3; r++) for (let c = 0; c < nB; c++) { Jp[r * nB + c] = P[r * nv + B[c]]; Jr[r * nB + c] = R[r * nv + B[c]]; }
    let dot = [0, 0, 0, 0, 0, 0];
    if (withDot) {
      mj.mj_jacDot(m, d, this.bufDP, this.bufDR, point, body);
      const DP = this.bufDP.GetView(), DR = this.bufDR.GetView(), qv = d.qvel;
      for (let r = 0; r < 3; r++) {
        let a = 0, b = 0;
        for (let c = 0; c < nv; c++) { a += DP[r * nv + c] * qv[c]; b += DR[r * nv + c] * qv[c]; }
        dot[r] = a; dot[r + 3] = b;
      }
    }
    return { Jp, Jr, dot };
  }
  jacCom() {
    const { mj, m } = this, d = this.w.data, nv = this.nv, nB = this.nB, B = this.B;
    mj.mj_jacSubtreeCom(m, d, this.bufP, this.ids.pelvis);
    const P = this.bufP.GetView();
    const Jc = new Float64Array(3 * nB);
    for (let r = 0; r < 3; r++) for (let c = 0; c < nB; c++) Jc[r * nB + c] = P[r * nv + B[c]];
    return Jc;
  }
  // מהירות של נקודה/סיבוב: J q̇ (על כל הדרגות, כולל אצבעות שלא נכללות ב-B; זניח)
  vel(J) {
    const qv = this.w.data.qvel, B = this.B, nB = this.nB, rows = J.length / nB;
    const out = new Array(rows).fill(0);
    for (let r = 0; r < rows; r++) { let s = 0; for (let c = 0; c < nB; c++) s += J[r * nB + c] * qv[B[c]]; out[r] = s; }
    return out;
  }

  // ---------- הוספת משימה למערכת הריבועים הפחותים ----------
  addTask(J, a, w) {
    const nB = this.nB, H = this.H, g = this.g, rows = a.length;
    for (let r = 0; r < rows; r++) {
      const off = r * nB, ar = a[r];
      for (let i = 0; i < nB; i++) {
        const ji = J[off + i];
        if (ji === 0) continue;
        const wji = w * ji;
        g[i] += wji * ar;
        for (let k = 0; k < nB; k++) H[i * nB + k] += wji * J[off + k];
      }
    }
  }
  addJointTask(i, a, w) { this.H[i * this.nB + i] += w; this.g[i] += w * a; }

  // ---------- צעד בקרה ----------
  // cmd: { stance:{l,r}, comAcc:[3], pelvisR, pelvisKp, torsoR, feet:{l:{pos,vel,acc,R}}, hands:{l:{pos,vel,acc,R,w}},
  //        post: Float64Array(30) יעדי מפרקים, postKp, postW: Float64Array(30), loads:{l:[F], r:[F]}, cop:{l:0.5} }
  compute(cmd) {
    const d = this.w.data, nB = this.nB, B = this.B, H = this.H, g = this.g;
    H.fill(0); g.fill(0);
    const qv = d.qvel;

    // --- מרכז המסה ---
    const Jc = this.jacCom();
    const comVel = this.vel(Jc);
    this.addTask(Jc, cmd.comAcc, cmd.comW ?? 5000);
    this.out.comVel = comVel;

    // --- אוריינטציית האגן ---
    {
      const { Jr } = this.jac(this.ids.pelvis, Array.from(d.xpos.subarray(3 * this.ids.pelvis, 3 * this.ids.pelvis + 3)), false);
      const R = this.bodyMat(this.ids.pelvis);
      const e = rotError(cmd.pelvisR, R), wv = this.vel(Jr);
      const kp = cmd.pelvisKp ?? 300, kd = 2 * Math.sqrt(kp);
      this.addTask(Jr, [kp * e[0] - kd * wv[0], kp * e[1] - kd * wv[1], kp * e[2] - kd * wv[2]], cmd.pelvisW ?? 1500);
    }
    // --- אוריינטציית הגו (חלש יותר: מאפשר הטיה קדימה כשמושיטים יד) ---
    if (cmd.torsoR) {
      const { Jr } = this.jac(this.ids.torso, Array.from(d.xpos.subarray(3 * this.ids.torso, 3 * this.ids.torso + 3)), false);
      const R = this.bodyMat(this.ids.torso);
      const e = rotError(cmd.torsoR, R), wv = this.vel(Jr);
      const kp = 120, kd = 2 * Math.sqrt(kp);
      this.addTask(Jr, [kp * e[0] - kd * wv[0], kp * e[1] - kd * wv[1], kp * e[2] - kd * wv[2]], cmd.torsoW ?? 30);
    }

    // --- כפות הרגליים: רגל עומדת = תאוצה אפס (משקל גבוה), רגל מניפה = עקיבה אחרי מסלול ---
    const contacts = [];
    for (const s of ['l', 'r']) {
      const sole = this.ids[s + '_sole'], foot = this.ids[s + '_foot'];
      const p = this.sitePos(sole);
      const J = this.jac(foot, p, true);
      const R = this.bodyMat(foot);
      if (cmd.stance[s]) {
        // ייצוב קטן נגד זחילה, בנוסף לתאוצה אפס
        const vl = this.vel(J.Jp), wr = this.vel(J.Jr);
        this.addTask(J.Jp, [-J.dot[0] - 20 * vl[0], -J.dot[1] - 20 * vl[1], -J.dot[2] - 20 * vl[2]], STANCE_W);
        this.addTask(J.Jr, [-J.dot[3] - 20 * wr[0], -J.dot[4] - 20 * wr[1], -J.dot[5] - 20 * wr[2]], STANCE_W);
        contacts.push({ s, p, R, J });
      } else if (cmd.feet && cmd.feet[s]) {
        const f = cmd.feet[s];
        const vl = this.vel(J.Jp), wr = this.vel(J.Jr);
        const kp = f.kp ?? 500, kd = 2 * Math.sqrt(kp);
        const a = [0, 1, 2].map(i => (f.acc?.[i] || 0) + kp * (f.pos[i] - p[i]) + kd * ((f.vel?.[i] || 0) - vl[i]) - J.dot[i]);
        this.addTask(J.Jp, a, f.w ?? 400);
        const e = rotError(f.R, R);
        const kr = 300, kdr = 2 * Math.sqrt(kr);
        this.addTask(J.Jr, [0, 1, 2].map(i => kr * e[i] - kdr * wr[i] - J.dot[3 + i]), f.w ?? 400);
      }
    }
    this.out.contacts = contacts.map(c => c.s);

    // --- ידיים ---
    const handJ = {};
    for (const s of ['l', 'r']) {
      const h = cmd.hands?.[s];
      const site = this.ids[s + '_grip'], body = this.ids[s + '_hand'];
      const p = this.sitePos(site);
      if (!h && !(cmd.loads?.[s])) continue;
      const J = this.jac(body, p, !!h);
      handJ[s] = { J, p };
      if (!h) continue;
      const vl = this.vel(J.Jp);
      const kp = h.kp ?? 160, kd = 2 * Math.sqrt(kp);
      const a = [0, 1, 2].map(i => (h.acc?.[i] || 0) + kp * (h.pos[i] - p[i]) + kd * ((h.vel?.[i] || 0) - vl[i]) - J.dot[i]);
      this.addTask(J.Jp, a, h.w ?? 80);
      if (h.R) {
        const R = this.bodyMat(body), wr = this.vel(J.Jr);
        const e = rotError(h.R, R), kr = h.kr ?? 120, kdr = 2 * Math.sqrt(kr);
        this.addTask(J.Jr, [0, 1, 2].map(i => kr * e[i] - kdr * wr[i] - J.dot[3 + i]), h.wr ?? 20);
      }
    }

    // --- תנוחה: כל מפרק נמשך בעדינות אל היעד שלו ---
    const nj = this.joints.length;
    const kpP = cmd.postKp ?? 80, kdP = 2 * Math.sqrt(kpP);
    for (let k = 0; k < nj; k++) {
      const j = this.joints[k];
      const q = d.qpos[j.qadr], qd = qv[j.dof];
      const a = kpP * (cmd.post[k] - q) - kdP * qd;
      this.addJointTask(6 + k, a, cmd.postW[k]);
    }
    // גבולות רכים: כשמפרק מתקרב לעצירה המכנית (פחות מ-8°), משימה חזקה דוחפת אותו חזרה.
    // בלי זה הברך העומדת מתיישרת עד הסוף, פוגעת בעצירה, ומקבלת מכה שמפילה את הרובוט
    // הברך (מול הסינגולריות של רגל ישרה) מקבלת גבול חזק; שאר המפרקים גבול עדין, כדי שלא יתחרו בשיווי המשקל
    for (let k = 0; k < nj; k++) {
      const j = this.joints[k], q = d.qpos[j.qadr], qd = qv[j.dof];
      const knee = j.name.endsWith('knee'), margin = knee ? 0.14 : 0.06, W = knee ? 300 : 20;
      const lo = j.range[0] + margin, hi = j.range[1] - margin;
      if (q < lo) this.addJointTask(6 + k, 400 * (lo - q) - 30 * qd, W * Math.min(1, (lo - q) / margin + 0.2));
      else if (q > hi) this.addJointTask(6 + k, 400 * (hi - q) - 30 * qd, W * Math.min(1, (q - hi) / margin + 0.2));
    }
    // רגולריזציה קטנה על הכול
    for (let i = 0; i < nB; i++) H[i * nB + i] += 1e-3;

    // --- הדינמיקה: M ו-h על דרגות הגוף ---
    const { mj, m } = this, nv = this.nv;
    mj.mj_fullM(m, d, this.bufM);
    const M = this.bufM.GetView();
    const bias = d.qfrc_bias, passive = d.qfrc_passive;
    const h = new Float64Array(nB);
    for (let r = 0; r < nB; r++) h[r] = bias[B[r]] - passive[B[r]];
    // עומסים חיצוניים ידועים (חפץ ביד): h −= Jᵀ F
    for (const s of ['l', 'r']) {
      const F = cmd.loads?.[s];
      if (!F || !handJ[s]) continue;
      const Jp = handJ[s].J.Jp;
      for (let c = 0; c < nB; c++) h[c] -= Jp[c] * F[0] + Jp[nB + c] * F[1] + Jp[2 * nB + c] * F[2];
    }
    const Mb = new Float64Array(nB * nB);
    for (let r = 0; r < nB; r++) for (let c = 0; c < nB; c++) Mb[r * nB + c] = M[B[r] * nv + B[c]];

    // --- כוחות המגע: 4 פינות לכל סוליה, בכל פינה פירמידת חיכוך עם 4 צלעות ---
    // f = Σ βᵢ eᵢ, β ≥ 0: כך כל האילוצים (רק דוחף, חיכוך, מרכז לחץ בתוך הסוליה) הם פשוט אי-שליליות.
    const Q = this.contactBasis(contacts);        // nB × nβ: הכוח המוכלל של כל צלע
    const nb = Q.n;

    // --- פתרון משותף ל-q̈ ול-β ---
    // מינימום: Σ w‖J q̈ − a‖²  +  ρ‖M_b q̈ + h_b − Q_b β‖²  +  ε‖β‖²   בכפוף ל-β ≥ 0
    // (שש השורות של הבסיס הצף הן משוואות התנועה שאין להן מנוע; ρ גדול אוכף אותן כמעט במדויק)
    const rho = RHO, eps = 2e-3;
    for (let i = 0; i < nB; i++) for (let k = 0; k < nB; k++) {
      let s2 = 0; for (let r = 0; r < 6; r++) s2 += Mb[r * nB + i] * Mb[r * nB + k];
      H[i * nB + k] += rho * s2;
    }
    for (let i = 0; i < nB; i++) { let s2 = 0; for (let r = 0; r < 6; r++) s2 += Mb[r * nB + i] * h[r]; g[i] -= rho * s2; }
    // A_qβ = −ρ M_bᵀ Q_b  (nB × nb)
    const Aqb = new Float64Array(nB * nb);
    for (let i = 0; i < nB; i++) for (let k = 0; k < nb; k++) {
      let s2 = 0; for (let r = 0; r < 6; r++) s2 += Mb[r * nB + i] * Q.m[r * nb + k];
      Aqb[i * nb + k] = -rho * s2;
    }
    cholFactor(H, nB);
    const y = cholSolveF(H, g, nB);
    const X = new Float64Array(nB * nb), col = new Float64Array(nB);
    for (let k = 0; k < nb; k++) {
      for (let i = 0; i < nB; i++) col[i] = Aqb[i * nb + k];
      const xs = cholSolveF(H, col, nB);
      for (let i = 0; i < nB; i++) X[i * nb + k] = xs[i];
    }
    // הבעיה המצומצמת ב-β: P = A_ββ − A_qβᵀ X,  q = b_β − A_qβᵀ y
    let beta = new Float64Array(nb);
    if (nb) {
      const P = new Float64Array(nb * nb), qv2 = new Float64Array(nb);
      for (let i = 0; i < nb; i++) {
        let s2 = 0; for (let r = 0; r < 6; r++) s2 += Q.m[r * nb + i] * h[r];
        let t2 = 0; for (let c = 0; c < nB; c++) t2 += Aqb[c * nb + i] * y[c];
        qv2[i] = rho * s2 - t2;
        for (let k = i; k < nb; k++) {
          let a2 = 0; for (let r = 0; r < 6; r++) a2 += Q.m[r * nb + i] * Q.m[r * nb + k];
          let b2 = 0; for (let c = 0; c < nB; c++) b2 += Aqb[c * nb + i] * X[c * nb + k];
          const v2 = rho * a2 - b2 + (i === k ? eps * rho * 1e-3 + eps : 0);
          P[i * nb + k] = v2; P[k * nb + i] = v2;
        }
      }
      // ירידה בקואורדינטות עם הצמדה לאפס, מתחילים מהפתרון הקודם
      beta = this.warmBeta(contacts, nb);
      for (let it = 0; it < 60; it++) {
        let change = 0;
        for (let i = 0; i < nb; i++) {
          let s2 = qv2[i];
          for (let k = 0; k < nb; k++) if (k !== i) s2 -= P[i * nb + k] * beta[k];
          const nbv = Math.max(0, s2 / P[i * nb + i]);
          change = Math.max(change, Math.abs(nbv - beta[i]));
          beta[i] = nbv;
        }
        if (change < 0.05) break;
      }
      this.saveBeta(contacts, beta);
    }
    // q̈ = y − X β
    const qdd = this.qdd;
    for (let i = 0; i < nB; i++) { let s2 = y[i]; for (let k = 0; k < nb; k++) s2 -= X[i * nb + k] * beta[k]; qdd[i] = s2; }

    // --- מומנטים: τ = (M q̈ + h − Q β) בשורות המפרקים ---
    const tau = this.tau;
    for (let k = 0; k < nj; k++) {
      const c = 6 + k;
      let t = h[c];
      for (let i = 0; i < nB; i++) t += Mb[c * nB + i] * qdd[i];
      for (let i = 0; i < nb; i++) t -= Q.m[c * nb + i] * beta[i];
      const j = this.joints[k];
      // פיצוי חיכוך יבש: מוסיפים את החיכוך הצפוי בכיוון התנועה (או התאוצה המתוכננת, כשהמפרק עומד)
      const dir = qv[j.dof] + 0.03 * qdd[c];
      t += j.frictionloss * Math.tanh(dir / 0.02);
      const lim = torqueLimit(j, qv[j.dof], t);
      tau[k] = clamp(t, -lim, lim);
    }
    // שארית הדינמיקה של הבסיס (לבדיקה): כמה הפתרון "מרמה" את הפיזיקה
    let res = 0;
    for (let r = 0; r < 6; r++) { let s2 = h[r]; for (let i = 0; i < nB; i++) s2 += Mb[r * nB + i] * qdd[i]; for (let i = 0; i < nb; i++) s2 -= Q.m[r * nb + i] * beta[i]; res = Math.max(res, Math.abs(s2)); }
    this.out.baseResidual = res;
    this.out.wrenches = this.wrenchesFromBeta(contacts, beta);
    this.out.comJ = Jc;
    return tau;
  }

  // בסיס הכוחות: לכל רגל עומדת 4 פינות × 4 צלעות פירמידה
  contactBasis(contacts) {
    const nB = this.nB, mu = 0.7 / Math.SQRT2;   // פירמידה פנימית לחרוט החיכוך
    const hx = 0.5 * G.footLen * 0.92, hy = 0.5 * G.footW * 0.9;
    const cols = [];
    for (const c of contacts) {
      const R = c.R, ex = [R[0], R[3], R[6]], ey = [R[1], R[4], R[7]], ez = [R[2], R[5], R[8]];
      c.edges = [];
      for (const [sx, sy] of [[1, 1], [1, -1], [-1, -1], [-1, 1]]) {
        const r = [ex[0] * sx * hx + ey[0] * sy * hy, ex[1] * sx * hx + ey[1] * sy * hy, ex[2] * sx * hx + ey[2] * sy * hy];
        for (const [a, b] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          let e = [ez[0] + mu * (a * ex[0] + b * ey[0]), ez[1] + mu * (a * ex[1] + b * ey[1]), ez[2] + mu * (a * ex[2] + b * ey[2])];
          const L = Math.hypot(...e); e = e.map(x => x / L);
          const tq = [r[1] * e[2] - r[2] * e[1], r[2] * e[0] - r[0] * e[2], r[0] * e[1] - r[1] * e[0]];
          const colv = new Float64Array(nB);
          for (let i = 0; i < nB; i++) colv[i] = c.J.Jp[i] * e[0] + c.J.Jp[nB + i] * e[1] + c.J.Jp[2 * nB + i] * e[2] + c.J.Jr[i] * tq[0] + c.J.Jr[nB + i] * tq[1] + c.J.Jr[2 * nB + i] * tq[2];
          cols.push(colv);
          c.edges.push({ r, e });
        }
      }
    }
    const n = cols.length, m = new Float64Array(nB * n);
    for (let k = 0; k < n; k++) for (let i = 0; i < nB; i++) m[i * n + k] = cols[k][i];
    return { m, n };
  }
  warmBeta(contacts, nb) {
    const b = new Float64Array(nb);
    this.betaMem = this.betaMem || {};
    contacts.forEach((c, ci) => { const mem = this.betaMem[c.s]; if (mem) b.set(mem, ci * 16); else b.fill(15, ci * 16, ci * 16 + 16); });
    return b;
  }
  saveBeta(contacts, beta) {
    this.betaMem = {};
    contacts.forEach((c, ci) => { this.betaMem[c.s] = beta.slice(ci * 16, ci * 16 + 16); });
  }
  wrenchesFromBeta(contacts, beta) {
    return contacts.map((c, ci) => {
      const F = [0, 0, 0], T = [0, 0, 0];
      c.edges.forEach(({ r, e }, k) => {
        const b = beta[ci * 16 + k];
        for (let i = 0; i < 3; i++) F[i] += b * e[i];
        T[0] += b * (r[1] * e[2] - r[2] * e[1]); T[1] += b * (r[2] * e[0] - r[0] * e[2]); T[2] += b * (r[0] * e[1] - r[1] * e[0]);
      });
      return { s: c.s, p: c.p, R: c.R, w: [...F, ...T] };
    });
  }

  apply(tau) {
    const ctrl = this.w.data.ctrl;
    for (let k = 0; k < this.joints.length; k++) ctrl[this.joints[k].act] = tau[k];
  }
}
