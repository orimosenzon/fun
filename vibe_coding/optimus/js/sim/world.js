// העולם: טוען את MuJoCo, בונה מודל מלא (רובוט + סצנה), ומחזיק את המצב.
// המודול לא מייבא את MuJoCo בעצמו: הדפדפן והבדיקות ב-node מעבירים אותו (mj), כדי שאותו קוד ירוץ בשניהם.
import { robotXML, actuatorXML, standingQpos } from '../robot/mjcf.js';

export const DT = 0.002;   // צעד סימולציה: 500 הרץ

export function buildXML(scene = {}) {
  const objs = scene.objects || [];
  const statics = scene.statics || [];
  return `<mujoco model="optimus3">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="${DT}" integrator="implicitfast" cone="elliptic" impratio="3" iterations="40" ls_iterations="12">
    <flag eulerdamp="disable"/>
  </option>
  <size memory="40M"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <default>
    <geom solref="0.005 1" solimp="0.92 0.97 0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1" friction="0.9 0.01 0.001" contype="1" conaffinity="3"/>
    ${statics.map(staticXML).join('\n    ')}
    ${robotXML()}
    ${objs.map(objectXML).join('\n    ')}
  </worldbody>
  <actuator>
${actuatorXML()}
  </actuator>
</mujoco>`;
}

// חפצים סטטיים (שולחנות, מדפים, מדרגות): רק גיאומטריה
function staticXML(s) {
  const p = (s.pos || [0, 0, 0]).join(' ');
  const q = s.yaw ? ` quat="${Math.cos(s.yaw / 2)} 0 0 ${Math.sin(s.yaw / 2)}"` : '';
  const fr = s.friction ? ` friction="${s.friction} 0.01 0.001"` : '';
  return `<geom name="${s.name}" type="${s.type || 'box'}" pos="${p}"${q} size="${s.size.join(' ')}"${fr} contype="1" conaffinity="3"/>`;
}

// חפצים חופשיים (קופסאות, תאי סוללה, כוסות): גוף עם freejoint
function objectXML(o) {
  const p = (o.pos || [0, 0, 1]).join(' ');
  const q = o.yaw ? ` quat="${Math.cos(o.yaw / 2)} 0 0 ${Math.sin(o.yaw / 2)}"` : (o.quat ? ` quat="${o.quat.join(' ')}"` : '');
  const fr = o.friction ? `${o.friction} 0.01 0.0005` : '0.9 0.01 0.0005';
  const geoms = (o.geoms || [{ type: o.type, size: o.size }]).map((g, i) =>
    `<geom name="${o.name}${i ? '_' + i : ''}" type="${g.type}" size="${g.size.join(' ')}"${g.pos ? ` pos="${g.pos.join(' ')}"` : ''} ${g.mass != null ? `mass="${g.mass}"` : ''} friction="${fr}" condim="4" contype="1" conaffinity="3"/>`).join('');
  const mass = o.mass != null && !o.geoms ? ` mass="${o.mass}"` : '';
  return `<body name="${o.name}" pos="${p}"${q}><freejoint name="${o.name}_j"/>${geoms.replace('/>', mass + '/>')}</body>`;
}

export class World {
  constructor(mj, scene = {}) {
    this.mj = mj;
    this.scene = scene;
    this.xml = buildXML(scene);
    this.model = mj.MjModel.from_xml_string(this.xml);
    this.data = new mj.MjData(this.model);
    this.t = 0;
    this.index();
    this.reset(scene.start);
  }
  id(type, name) { return this.mj.mj_name2id(this.model, this.mj.mjtObj[type].value, name); }
  bodyId(name) { return this.id('mjOBJ_BODY', name); }
  siteId(name) { return this.id('mjOBJ_SITE', name); }
  jointId(name) { return this.id('mjOBJ_JOINT', name); }
  index() {
    const m = this.model;
    this.names = { body: [], joint: [], actuator: [] };
    for (let i = 0; i < m.nbody; i++) this.names.body.push(this.mj.mj_id2name(m, this.mj.mjtObj.mjOBJ_BODY.value, i));
    for (let i = 0; i < m.njnt; i++) this.names.joint.push(this.mj.mj_id2name(m, this.mj.mjtObj.mjOBJ_JOINT.value, i));
    for (let i = 0; i < m.nu; i++) this.names.actuator.push(this.mj.mj_id2name(m, this.mj.mjtObj.mjOBJ_ACTUATOR.value, i));
  }
  reset(start = {}) {
    const { mj, model: m, data: d } = this;
    mj.mj_resetData(m, d);
    const q = standingQpos(m, mj, start.x || 0, start.y || 0, start.yaw || 0);
    // שומרים על מיקום החפצים כפי שהוגדרו במודל (qpos0)
    const robotNq = 7 + this.robotJointCount();
    for (let i = 0; i < robotNq; i++) d.qpos[i] = q[i];
    // אצבעות: מתחילות במיקום של היד הפתוחה
    mj.mj_forward(m, d);
    this.t = 0;
  }
  robotJointCount() {
    let c = 0;
    for (let j = 1; j < this.model.njnt; j++) { if (this.model.jnt_type[j] === 0) break; c++; }
    return c;
  }
  step() { this.mj.mj_step(this.model, this.data); this.t += DT; }
  dispose() { this.data.delete(); this.model.delete(); }
}
