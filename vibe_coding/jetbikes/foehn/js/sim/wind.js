// Mountain wind: a mean valley wind with a logarithmic boundary layer, orographic lift and sink
// where it blows up or down a slope, and a frozen-turbulence gust field (Taylor hypothesis)
// whose intensity grows near steep ground. Deterministic, so runs can be replayed.
import { clamp } from './vmath.js';
import { normalAt } from './terrain.js';
import { mulberry32 } from './noise.js';

export function makeWind({ heading = 0.25, speed = 4, gust = 1, seed = 7 } = {}) {
  const dir = [Math.cos(heading), 0, Math.sin(heading)];
  const r = mulberry32(seed);
  const modes = [];
  for (let k = 0; k < 12; k++) {
    const f = 0.03 * Math.pow(1.5, k);                 // Hz
    const amp = Math.pow(f / 0.03, -1 / 3);            // Kolmogorov -5/3 energy slope for log-spaced modes
    const a = r() * 6.283;
    modes.push({ f, amp, ph: [r() * 6.28, r() * 6.28, r() * 6.28], lat: [Math.cos(a), Math.sin(a)] });
  }
  const nrm = Math.sqrt(modes.reduce((s, m) => s + m.amp * m.amp, 0) / 2);
  const U = Math.max(speed, 1.5);
  const wind = function (p, t, agl) {
    agl = Math.max(0.3, agl);
    const prof = clamp(Math.log(agl / 0.05) / Math.log(40 / 0.05), 0.25, 1.35);
    const n = normalAt(p[0], p[2]);
    const slope = -(n[0] * dir[0] + n[2] * dir[2]) / n[1];           // dh/ds along the wind
    const steep = Math.hypot(n[0], n[2]) / n[1];
    const decay = Math.exp(-agl / 280);
    const w = [dir[0] * speed * prof, clamp(speed * prof * slope * decay, -7, 7), dir[2] * speed * prof];
    const sig = Math.min(gust * (0.35 + 0.11 * speed) * (1 + 1.4 * Math.min(steep, 1.5) * decay), 10);
    const along = p[0] * dir[0] + p[2] * dir[2], across = -p[0] * dir[2] + p[2] * dir[0];
    for (const m of modes) {
      const k = 6.283 * m.f / U;
      const arg = 6.283 * m.f * t - k * along + 0.6 * k * across * m.lat[0] + 0.4 * k * p[1] * m.lat[1];
      const a = sig * m.amp / nrm;
      w[0] += a * Math.sin(arg + m.ph[0]);
      w[1] += 0.6 * a * Math.min(1, agl / 12) * Math.sin(arg * 1.07 + m.ph[1]);
      w[2] += a * Math.sin(arg * 0.93 + m.ph[2]);
    }
    return w;
  };
  wind.dir = dir; wind.speed = speed;
  return wind;
}
export const calm = Object.assign(() => [0, 0, 0], { dir: [1, 0, 0], speed: 0 });

// The rider chooses how hard it blows. Level 1 is the wind each event was designed with; above
// it both the mean wind and, more than that, the gusts grow, and the gusts are what makes the
// bike hard to hold: they come from every side and no two runs are the same (the seed).
export const WIND_LEVELS = ['calm', 'breeze', 'strong', 'storm'];
export function windFor(ev = {}, level = 1, seed = 7) {
  if (level <= 0) return calm;
  const k = [0, 1, 2.2, 3.4][level], g = [0, 1, 2.2, 3.4][level], s0 = [0, 0, 9, 15][level], g0 = [0, 0, 2.2, 3.4][level];
  return makeWind({ heading: ev.heading ?? 0.25, speed: Math.max((ev.speed ?? 4) * k, s0), gust: Math.max((ev.gust ?? 1) * g, g0), seed });
}
