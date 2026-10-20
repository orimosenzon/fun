// בונה את קובץ ה-MJCF (פורמט המודל של MuJoCo) של הרובוט מתוך spec.js.
// הגיאומטריה כאן היא לחישוב מסה, אינרציה והתנגשויות. הציור ב-render/ בנוי על אותן מידות.
import { G, JOINTS, FINGERS, MASS, FINGER_KP, FINGER_FORCE, NOMINAL } from './spec.js';

const n = (x) => +x.toFixed(5);
const v3 = (a) => a.map(n).join(' ');
const J = Object.fromEntries(JOINTS.map(j => [j.name, j]));

function joint(name, axis, extra = '') {
  const j = J[name];
  return `<joint name="${name}" axis="${axis}" range="${v3(j.range)}" armature="${j.armature}" damping="${j.damping}" frictionloss="${j.frictionloss}" ${extra}/>`;
}

// מחלקות התנגשות: הרובוט לא מתנגש בעצמו (חוץ מכפות הידיים מול הגו), אבל כן ברצפה ובחפצים.
const ROBOT = 'contype="2" conaffinity="1" group="3"';
const HANDC = 'contype="6" conaffinity="1" group="3" condim="4" friction="1.1 0.02 0.0005"';
const TORSOC = 'contype="2" conaffinity="5" group="3"';

function finger(side, f, palmSign) {
  // palmSign: כיוון נורמל כף היד (−y ליד שמאל, +y לימין). ציר הכפיפה מסודר כך שחיובי = כפיפה לכיוון כף היד בשתי הידיים
  const fx = side === 'l' ? '-1 0 0' : '1 0 0';
  const P = `${side}_${f.key}`;
  const [l1, l2, l3] = f.segs, r = f.r;
  const m = MASS.finger;
  const SMALL = '<inertial pos="0 0 0" mass="0.006" diaginertia="2e-7 2e-7 2e-7"/>';
  const cap = (len, rr) => `<geom type="capsule" fromto="0 0 0 0 0 ${-len}" size="${rr}" mass="${m}" ${HANDC}/>`;
  if (f.key === 'thumb') {
    // האגודל: מפרק ראשון סביב הציר האורכי של כף היד (עימות, opposition) מעביר אותו ממישור כף היד
    // אל מול האצבעות. אחריו הטיה קבועה קדימה-מטה, ושלושה מפרקי כפיפה.
    const pos = [0.03, palmSign * 0.011, -0.02];
    const ang = 26 * Math.PI / 180;
    const q = [Math.cos(-ang / 2), 0, Math.sin(-ang / 2), 0];
    return `<body name="${P}_0" pos="${v3(pos)}">
      <joint name="${P}_abd" axis="0 0 ${side === 'l' ? -1 : 1}" range="-0.2 1.6" armature="0.0004" damping="0.02"/>${SMALL}
      <body name="${P}_1" quat="${v3(q)}"><joint name="${P}_mcp" axis="${fx}" range="-0.3 1.0" armature="0.0004" damping="0.02"/>${cap(l1, r)}
        <body name="${P}_2" pos="0 0 ${-l1}"><joint name="${P}_pip" axis="${fx}" range="0 1.2" armature="0.0003" damping="0.015"/>${cap(l2, r * 0.95)}
          <body name="${P}_3" pos="0 0 ${-l2}"><joint name="${P}_dip" axis="${fx}" range="0 1.3" armature="0.0002" damping="0.01"/>${cap(l3, r * 0.9)}
            <site name="${P}_tip" pos="0 0 ${-l3}" size="0.006"/>
          </body></body></body></body>`;
  }
  const pos = [f.x, 0, -G.palmLen];
  return `<body name="${P}_0" pos="${v3(pos)}">
    <joint name="${P}_abd" axis="0 1 0" range="-0.3 0.3" armature="0.0003" damping="0.02"/>${SMALL}
    <body name="${P}_1"><joint name="${P}_mcp" axis="${fx}" range="-0.3 1.6" armature="0.0004" damping="0.02"/>${cap(l1, r)}
      <body name="${P}_2" pos="0 0 ${-l1}"><joint name="${P}_pip" axis="${fx}" range="0 1.75" armature="0.0003" damping="0.015"/>${cap(l2, r * 0.95)}
        <body name="${P}_3" pos="0 0 ${-l2}"><joint name="${P}_dip" axis="${fx}" range="0 1.3" armature="0.0002" damping="0.01"/>${cap(l3, r * 0.9)}
          <site name="${P}_tip" pos="0 0 ${-l3}" size="0.006"/>
        </body></body></body></body>`;
}

function arm(side) {
  const s = side === 'l' ? 1 : -1;
  const palmSign = -s;   // כף היד פונה פנימה, אל הגוף
  const fx = side === 'l' ? '-1 0 0' : '1 0 0';
  const P = side;
  const sh = [0, s * G.shoulderY, G.shoulderZ - G.waistZ];
  return `<body name="${P}_upper_arm" pos="${v3(sh)}">
    ${joint(`${P}_shoulder_pitch`, '0 1 0')}
    ${joint(`${P}_shoulder_roll`, '1 0 0')}
    ${joint(`${P}_shoulder_yaw`, '0 0 1')}
    <geom name="${P}_uarm" type="capsule" fromto="0 0 -0.04 0 0 ${-G.upperArm + 0.03}" size="0.047" mass="${MASS.upperArm}" ${ROBOT}/>
    <body name="${P}_forearm" pos="0 0 ${-G.upperArm}">
      ${joint(`${P}_elbow`, '0 1 0')}
      ${joint(`${P}_forearm_roll`, '0 0 1')}
      <geom name="${P}_farm" type="capsule" fromto="0 0 -0.03 0 0 ${-G.forearm + 0.035}" size="0.044" mass="${MASS.forearm}" ${ROBOT}/>
      <body name="${P}_hand" pos="0 0 ${-G.forearm}">
        <joint name="${P}_wrist_flex" axis="${fx}" range="${v3(J[`${P}_wrist_flex`].range)}" armature="${J[`${P}_wrist_flex`].armature}" damping="${J[`${P}_wrist_flex`].damping}" frictionloss="${J[`${P}_wrist_flex`].frictionloss}"/>
        <joint name="${P}_wrist_dev" axis="0 ${s} 0" range="${v3(J[`${P}_wrist_dev`].range)}" armature="${J[`${P}_wrist_dev`].armature}" damping="${J[`${P}_wrist_dev`].damping}" frictionloss="${J[`${P}_wrist_dev`].frictionloss}"/>
        <geom name="${P}_palm" type="box" pos="0 0 ${-G.palmLen / 2 - 0.005}" size="${G.palmW / 2} ${G.palmT / 2} ${G.palmLen / 2}" mass="${MASS.palm}" ${HANDC}/>
        <site name="${P}_palm_c" pos="0 ${palmSign * (G.palmT / 2 + 0.03)} ${-G.palmLen * 0.75}" size="0.008"/>
        <site name="${P}_grip" pos="0.008 ${palmSign * 0.03} ${-G.palmLen - 0.03}" size="0.008"/>
        ${FINGERS.map(f => finger(side, f, palmSign)).join('\n')}
      </body>
    </body>
  </body>`;
}

function leg(side) {
  const s = side === 'l' ? 1 : -1;
  const P = side;
  const hip = [0, s * G.hipY, G.hipZ - G.pelvisZ];
  const fl = G.footLen, fw = G.footW, fb = G.footBack;
  const footC = [fl / 2 - fb, 0, -G.ankleZ + 0.017];
  return `<body name="${P}_thigh" pos="${v3(hip)}">
    ${joint(`${P}_hip_yaw`, '0 0 1')}
    ${joint(`${P}_hip_roll`, '1 0 0')}
    ${joint(`${P}_hip_pitch`, '0 1 0')}
    <geom name="${P}_thigh_g" type="capsule" fromto="0 0 -0.05 0 0 ${-G.thigh + 0.06}" size="0.068" mass="${MASS.thigh}" ${TORSOC}/>
    <body name="${P}_shin" pos="0 0 ${-G.thigh}">
      ${joint(`${P}_knee`, '0 1 0')}
      <geom name="${P}_shin_g" type="capsule" fromto="0 0 -0.04 0 0 ${-G.shin + 0.06}" size="0.05" mass="${MASS.shin}" ${ROBOT}/>
      <body name="${P}_foot" pos="0 0 ${-G.shin}">
        ${joint(`${P}_ankle_pitch`, '0 1 0')}
        ${joint(`${P}_ankle_roll`, '1 0 0')}
        <geom name="${P}_foot_g" type="box" pos="${v3(footC)}" size="${fl / 2} ${fw / 2} 0.017" mass="${MASS.foot}" contype="2" conaffinity="1" group="3" friction="1.0 0.02 0.001" priority="1"/>
        <site name="${P}_sole" pos="${v3([footC[0], 0, -G.ankleZ])}" size="0.01"/>
      </body>
    </body>
  </body>`;
}

export function robotXML() {
  const torsoC = [0.0, 0, 0.2];
  return `<body name="pelvis" pos="0 0 ${G.pelvisZ}">
  <freejoint name="root"/>
  <geom name="pelvis_g" type="capsule" fromto="0 -0.07 -0.01 0 0.07 -0.01" size="0.105" mass="${MASS.pelvis}" ${TORSOC}/>
  <site name="imu" pos="0 0 0" size="0.01"/>
  <body name="abdomen" pos="0 0 ${G.waistZ - G.pelvisZ}">
    ${joint('waist_yaw', '0 0 1')}
    <geom type="sphere" size="0.05" mass="${MASS.abdomen}" ${ROBOT}/>
    <body name="torso">
      ${joint('waist_pitch', '0 1 0')}
      <geom name="torso_g" type="ellipsoid" pos="${v3(torsoC)}" size="0.115 0.165 0.2" mass="${MASS.torso}" ${TORSOC}/>
      <body name="neck" pos="0 0 ${G.neckZ - G.waistZ}">
        ${joint('neck_yaw', '0 0 1')}
        <inertial pos="0 0 0.02" mass="0.35" diaginertia="0.0004 0.0004 0.0003"/>
        <body name="head">
          ${joint('neck_pitch', '0 1 0')}
          <geom name="head_g" type="ellipsoid" pos="0.01 0 ${G.headC - G.neckZ}" size="0.1 0.083 0.118" mass="${MASS.head}" ${ROBOT}/>
          <camera name="eyes" pos="0.09 0 ${G.headC - G.neckZ + 0.02}" xyaxes="0 -1 0 0 0 1" fovy="70"/>
        </body>
      </body>
      ${arm('l')}
      ${arm('r')}
    </body>
  </body>
  ${leg('l')}
  ${leg('r')}
</body>`;
}

export function actuatorXML() {
  const body = JOINTS.map(j => `<motor name="${j.name}" joint="${j.name}" ctrlrange="${-j.torque} ${j.torque}" ctrllimited="true"/>`).join('\n');
  // אצבעות: מפעיל מיקום (סרוו), כמו מנוע באמה שמושך גיד עם בקר מיקום
  const fingers = ['l', 'r'].flatMap(s => FINGERS.flatMap(f => ['abd', 'mcp', 'pip', 'dip'].map(k => {
    const len = f.segs[k === 'mcp' ? 0 : k === 'pip' ? 1 : 2] ?? 0.03;
    const fr = Math.max(0.25, FINGER_FORCE * (k === 'abd' ? 0.03 : len + 0.02));
    const kp = FINGER_KP * (k === 'mcp' ? 1.6 : k === 'abd' ? 0.6 : 1);
    return `<position name="${s}_${f.key}_${k}" joint="${s}_${f.key}_${k}" kp="${n(kp)}" kv="${n(kp * 0.04)}" forcerange="${n(-fr)} ${n(fr)}" forcelimited="true"/>`;
  }))).join('\n');
  return body + '\n' + fingers;
}

// תנוחת התחלה: ברכיים כפופות מעט, הגובה מחושב כך שהרגליים על הרצפה
export function standingQpos(model, mj, x = 0, y = 0, yaw = 0) {
  const q = new Float64Array(model.nq);
  const set = (name, v) => { const id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT.value, name); if (id >= 0) q[model.jnt_qposadr[id]] = v; };
  // גובה האגן: הרגל מקוצרת ע"י כפיפת הברך
  const N = NOMINAL;
  const legH = G.thigh * Math.cos(N.hipPitch) + G.shin * Math.cos(N.knee + N.hipPitch);
  const pz = legH + G.ankleZ + (G.pelvisZ - G.hipZ) + 0.002;
  q[0] = x; q[1] = y; q[2] = pz;
  q[3] = Math.cos(yaw / 2); q[6] = Math.sin(yaw / 2);
  for (const s of ['l', 'r']) {
    const sg = s === 'l' ? 1 : -1;
    set(`${s}_hip_pitch`, N.hipPitch); set(`${s}_knee`, N.knee); set(`${s}_ankle_pitch`, -(N.knee + N.hipPitch));
    set(`${s}_shoulder_roll`, sg * N.shoulderRoll); set(`${s}_elbow`, N.elbow); set(`${s}_shoulder_pitch`, N.shoulderPitch);
  }
  return q;
}
