// node tools/guide_shots.mjs: plays a scripted beginner's game (one human, Spheroid, blue) through
// the whole first round and saves the screenshots the guide (how.html) uses, into guide/.
// The play is deterministic, so the pictures come out the same every time.
import fs from 'node:fs';
import { Session, grab, owner } from './lab.mjs';
import { shot } from './node64.mjs';

const out = new URL('../guide/', import.meta.url).pathname;
fs.mkdirSync(out, { recursive: true });
const s = new Session('guide', { quiet: true });
const texts = {};
const snap = (name) => { shot(s.m, `${out}${name}.png`, 2); texts[name] = s.text(); console.log(name.padEnd(18), s.t.toFixed(1)); return s; };
const me = () => s.sprites()[0];
const walkX = (x, max = 6) => { const d = me().x < x ? 'R' : 'L'; s.hold(1, d).until(() => (d === 'R' ? me().x >= x : me().x <= x), max); s.hold(1, '').wait(0.05); };
const walkY = (y, max = 8) => { const d = me().y < y ? 'D' : 'U'; s.hold(1, d).until(() => (d === 'D' ? me().y >= y : me().y <= y), max); s.hold(1, '').wait(0.05); };
const fire = () => s.joy(1, 'F', 0.3);

// Title and options: one human planeteer, beginner's game.
s.wait(3); fire(); s.wait(7); snap('title'); s.wait(6);
for (let i = 0; i < 3; i++) s.key('F5', 0.15, 0.6);
s.wait(0.5); snap('options'); s.key('F7').wait(2);
// Colour, species, colony.
fire(); s.wait(2); snap('color'); s.wait(2); fire();
s.wait(3); snap('species'); s.wait(5).joy(1, 'R', 0.5).wait(2); snap('spheroid'); s.wait(1); fire();
s.wait(3); snap('colony'); s.wait(1); fire();
s.wait(3); snap('landing'); s.wait(13); snap('summary0'); fire();
// Land grant: take the river plot just above the town.
s.wait(5);
s.until(() => { const p = s.sprites()[1]; return p.on && p.x === 40 + 32 * 2 && p.y === 66 + 32; }, 40); s.wait(0.15); snap('grant');
let mine = null;
for (const [c, r] of [[4, 1], [4, 3], [4, 4]]) { const before = owner(s, c, r); grab(s, c, r); s.wait(0.6); console.log('plot', c, r, 'colour', before, '->', owner(s, c, r)); if (owner(s, c, r) === 6) { mine = [c, r]; break; } }
if (!mine) throw new Error('no plot');
const plotAt = { x: 178 + 32 * (mine[0] - 4), y: 108 + 32 * (mine[1] - 1) };
snap('granted');
// Development: corral, food shop, out to the plot, install, back for the pub.
s.untilSees('PRESS YOUR STICK BUTTON'); s.wait(0.5); snap('yourturn'); fire(); s.wait(1); snap('town');
walkX(240); walkY(170, 2); snap('corral'); walkY(114, 3);
walkX(247); walkY(60, 2); s.wait(1); snap('outfit'); s.wait(1); walkY(114, 3); snap('outfitted');
walkX(300, 2); s.wait(0.5); snap('leading'); walkY(plotAt.y, 8); walkX(plotAt.x, 8); snap('onplot');
s.joy(1, 'F', 0.2).wait(1); if (!s.sees('MULE INSTALLED')) throw new Error('not installed: ' + s.text()); snap('installed');
walkX(215, 4); walkY(142, 4); walkX(190, 3); s.wait(0.5); walkX(184, 3); walkY(170, 2);
s.untilSees('GAMBLING', 5); s.wait(1); snap('pub');
s.untilSees('COMPUTER IS TAKING'); s.wait(1.5); snap('computer');
// A random event, if one comes before production.
let event = null;
s.until(() => { if (s.frames % 10) return false; const t = s.text(); if (!event && t && !/^(DEVELOPMENT|PRODUCTION)/.test(t) && t.split(' | ').length > 6 && /[A-Z]{4}/.test(t.split(' | ')[0])) { event = t.split(' | ')[0]; s.wait(1); snap('event'); } return t.startsWith('PRODUCTION'); }, 120);
console.log('event:', event);
s.untilSees('BUTTONS TO GO ON'); s.wait(0.5); snap('production'); fire();
// The three auctions.
s.untilSees(['STATUS', 'SMITHORE']);
try { s.untilSees(['AUCTION', 'SMITHORE', 'UNITS TRADED'], 60); s.wait(0.3); snap('auction_smithore'); } catch (e) { console.log('no smithore trade'); }
s.untilSees(['STATUS', 'FOOD', 'USAGE']); s.wait(0.6); snap('status_food_usage');
s.untilSees(['FOOD', 'SURPLUS']); s.wait(1); snap('status_food');
s.untilSees(['FOOD', 'DECLARE']); s.wait(2); snap('declare');
s.untilSees(['AUCTION', 'FOOD']); s.wait(0.5); s.hold(1, 'D'); s.wait(2.5); snap('auction_down');
s.untilSees('UNITS TRADED', 30); s.wait(1.2); snap('auction_trade');
s.untilSees('CRITICAL', 30); s.wait(0.3); snap('auction_critical'); s.wait(2.2); snap('auction_mine'); s.wait(3); snap('auction_mine2'); s.hold(1, '');
s.untilSees(['STATUS', 'ENERGY', 'SURPLUS']); s.wait(1); snap('status_energy');
s.untilSees('STATUS SUMMARY #1'); s.wait(3); snap('summary1');
fs.writeFileSync(new URL('../reference/guide_texts.json', import.meta.url).pathname, JSON.stringify(texts, null, 1));
