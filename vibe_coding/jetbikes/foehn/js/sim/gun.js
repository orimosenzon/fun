// The gun: one barrel bolted to the nose. It fires where the bike points, so aiming is flying.
// A single round every second and a half; the round is slow enough to watch (and to have to
// lead with), falls under gravity, and starts with the bike's own velocity. Every shot kicks the
// bike back and its nose up, and heats the core that the overdrive and the side kick also heat.
import { surfaceAt, waterAt, groundAt } from './terrain.js';
import { hitScenery } from './scenery.js';
import { hitCastle } from './castle.js';
import { FEEL } from './fcs.js';
import { G } from './bike.js';
import { qrot, clamp } from './vmath.js';

export const MUZZLE = [1.22, 0.74, 0];     // design frame: under the nose
export const GUN = { period: 1.5, heat: 0.2, life: 4, missPenalty: 2, nose: 0.5 };

export class Gun {
  constructor() { this.reset(); }
  reset() { this.shots = []; this.cool = 0; this.trigger = false; this.fired = 0; this.hits = 0; this.id = 0; }

  // once per control tick (1/120 s)
  step(dt, game) {
    const b = game.bike;
    this.cool = Math.max(0, this.cool - dt);
    if (this.trigger) {
      this.trigger = false;
      if (this.cool <= 0 && game.state === 'fly' && !b.abLock) this.fire(game);
    }
    for (let i = this.shots.length - 1; i >= 0; i--) {
      const s = this.shots[i], p = s.p, v = s.v, a = [p[0], p[1], p[2]];
      v[1] -= G * dt; p[0] += v[0] * dt; p[1] += v[1] * dt; p[2] += v[2] * dt; s.life -= dt;
      // a balloon on the way?  (closest point of this tick's stretch of flight to its centre)
      let hit = null, off = 0;
      const dx = p[0] - a[0], dy = p[1] - a[1], dz = p[2] - a[2], l2 = dx * dx + dy * dy + dz * dz || 1;
      for (let k = game.cp; k < game.cps.length; k++) {
        const c = game.cps[k]; if (c.type !== 'target' || c.popped) continue;
        const u = clamp(((c.p[0] - a[0]) * dx + (c.p[1] - a[1]) * dy + (c.p[2] - a[2]) * dz) / l2, 0, 1);
        const d = Math.hypot(a[0] + dx * u - c.p[0], a[1] + dy * u - c.p[1], a[2] + dz * u - c.p[2]);
        if (d < c.r && (!hit || d / c.r < off)) { hit = c; off = d / c.r; }
      }
      let end = null;
      if (hit) end = 'pop';
      else if (p[1] < surfaceAt(p[0], p[2])) end = waterAt(p[0], p[2]) > groundAt(p[0], p[2]) ? 'water' : 'ground';
      else if (hitCastle(p, 0) || hitScenery(p, 0)) end = 'wall';
      else if (s.life <= 0) end = 'air';
      if (!end) continue;
      this.shots.splice(i, 1);
      game.emit('shotEnd', { id: s.id, end, p: p.slice() });
      if (hit) { this.hits++; game.popTarget(hit, off); }
      else if (game.ev.gun && game.state === 'fly') { game.penalty += GUN.missPenalty; game.emit('shotMiss', { penalty: GUN.missPenalty }); }
    }
  }

  fire(game) {
    const b = game.bike, f = qrot(b.q, [1, 0, 0]), m = b.toWorld(MUZZLE), V = FEEL.muzzle, k = FEEL.recoil;
    const s = { id: ++this.id, p: m, v: [b.v[0] + f[0] * V, b.v[1] + f[1] * V, b.v[2] + f[2] * V], life: GUN.life };
    this.shots.push(s); this.fired++; this.cool = GUN.period;
    // recoil: a shove straight back and the nose thrown up (the barrel sits below the centre of mass)
    b.v[0] -= f[0] * k; b.v[1] -= f[1] * k; b.v[2] -= f[2] * k;
    b.w[2] += GUN.nose * k;
    b.heat = Math.min(1, b.heat + GUN.heat);
    game.emit('shot', { id: s.id, p: m.slice(), v: s.v.slice() });
  }
}
