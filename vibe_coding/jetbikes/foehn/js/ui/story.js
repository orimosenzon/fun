// The opening film: how Yakir found his father's drawing and built the bike.
// Everything is drawn here as SVG line art that draws itself (no images, no video): sepia paper for
// the past, a blueprint for the plan, and at the end the drawing dissolves into the live valley.
// playStory({ lang, audio, from, onEnd }) -> { stop() }.  ?story=N in the URL starts at scene N.

import { VOICE_LEN } from './story_voice.js';

const NS = 'http://www.w3.org/2000/svg';
const INK = '#2b2118', PAPER = '#e9dcc0', BLUE = '#0c3156', LINE = '#dff1ff', CY = '#6fe3ff', ACC = '#ff7a2a';

// ---- the words (subtitles), per scene: [seconds into the scene, he, en] ----
export const SCENES = [
  { id: 'lake', dur: 10, subs: [
    [0.8, 'שווייץ, 1969. הווארד יוז, טייס, ממציא ואחד האנשים העשירים בעולם, מתאהב בפיזיקאית ישראלית.', 'Switzerland, 1969. Howard Hughes, pilot, inventor and one of the richest men alive, falls in love with an Israeli physicist.'],
    [6.4, 'איש לא יודע על זה.', 'Nobody knows.']] },
  { id: 'born', dur: 10.5, subs: [
    [0.8, 'ב-1970 נולד להם בן. הם קוראים לו יקיר.', 'In 1970 they have a son. They name him Yakir.'],
    [5.6, 'יוז נעלם מחייהם, כמו שנעלם מהעולם כולו. הסוד נשאר בשווייץ.', 'Hughes vanishes from their lives, as he vanished from the whole world. The secret stays in Switzerland.']] },
  { id: 'attic', dur: 10.5, subs: [
    [0.8, 'עשרות שנים אחר כך, יקיר עולה לעליית הגג של בית ילדותו.', 'Decades later, Yakir climbs into the attic of the house he grew up in.'],
    [5.8, 'בתוך ארגז ישן מחכה לו גליל נייר.', 'In an old trunk, a roll of paper is waiting for him.']] },
  { id: 'plan', dur: 13, subs: [
    [0.8, 'תרשים, בכתב ידו של אביו.', 'A drawing, in his father\'s hand.'],
    [4.8, 'בהתחלה יקיר לא מבין על מה הוא מסתכל.', 'At first Yakir has no idea what he is looking at.'],
    [9, 'אבל הוא לא מסוגל להניח לזה.', 'But he cannot let it go.']] },
  { id: 'study', dur: 13, subs: [
    [0.8, 'הוא חוזר ללמוד. פיזיקה, כימיה, מתמטיקה. מההתחלה, לילה אחרי לילה.', 'He goes back to school. Physics, chemistry, mathematics. From the beginning, night after night.'],
    [7, 'עד שהוא מבין מה אבא שלו השאיר לו: דרך חדשה להיתוך קר.', 'Until he understands what his father left him: a new way to cold fusion.']] },
  { id: 'core', dur: 11.5, subs: [
    [0.8, 'עוד שנים של עבודה בסדנה, ויום אחד הליבה נדלקת.', 'More years in the workshop, and one day the core lights up.'],
    [6, 'ליקיר יש עוד אהבה ישנה: אופנועים. הוא יודע בדיוק לאן המנוע הזה הולך.', 'Yakir has one more old love: motorcycles. He knows exactly where this engine belongs.']] },
  { id: 'crash', dur: 14, subs: [
    [0.8, 'אופנוע בלי גלגלים. כל מה שמחזיק אותו באוויר הוא דחף.', 'A motorcycle with no wheels. Nothing holds it in the air but thrust.'],
    [5.2, 'הטיסות הראשונות נגמרות רע. באחת מהן יקיר נפצע קשה.', 'The first flights end badly. In one of them Yakir is badly hurt.'],
    [10, 'הוא חוזר לסדנה ומנסה שוב.', 'He goes back to the workshop and tries again.']] },
  { id: 'fly', dur: 14, subs: [
    [0.8, 'ואז, בוקר אחד מעל העמק, האופנוע נשאר באוויר.', 'And then, one morning above the valley, the bike stays in the air.'],
    [8.6, 'אבל בשולי התרשים יש שורה אחת שיקיר עדיין לא פיענח.', 'But in the margin of the drawing there is one line Yakir has not yet deciphered.']] },
];

const CSS = `
#story { position: fixed; inset: 0; z-index: 9; opacity: 0; transition: opacity .9s; background: #000; overflow: hidden; }
#story.on { opacity: 1; } #story.reveal { background: transparent; transition: opacity .9s, background 2.2s; }
#story svg { position: absolute; inset: 0; width: 100%; height: 100%; display: block; direction: ltr; }
#story .sc { opacity: 0; transition: opacity 1.4s; }
#story .sc.on { opacity: 1; }
#story .sc > .kb { transform-origin: 800px 450px; }
#story .sc.on > .kb { animation: st-kb var(--dur, 12s) linear forwards; }
@keyframes st-kb { from { transform: scale(1.0) translate(0, 0); } to { transform: scale(var(--z, 1.07)) translate(var(--px, 0px), var(--py, 0px)); } }
#story .dr { fill: none; stroke-dasharray: 1; stroke-dashoffset: 1; }
#story .sc.on .dr { animation: st-draw var(--t, 2s) var(--d, 0s) ease-in-out forwards; }
@keyframes st-draw { to { stroke-dashoffset: 0; } }
#story .fi { opacity: 0; }
#story .sc.on .fi { animation: st-fin var(--t, 1.2s) var(--d, 0s) ease forwards; }
@keyframes st-fin { to { opacity: var(--o, 1); } }
#story .fo { opacity: var(--o, 1); }
#story .sc.on .fo { animation: st-fout var(--t, 3s) var(--d, 0s) ease forwards; }
@keyframes st-fout { to { opacity: 0; } }
#story .sc.on .plane { animation: st-plane 16s linear forwards; }
@keyframes st-plane { from { transform: translate(-160px, 230px); } to { transform: translate(1750px, 120px); } }
#story .sc.on .rock { animation: st-rock 3.4s ease-in-out infinite alternate; transform-origin: 1010px 700px; }
@keyframes st-rock { from { transform: rotate(-3.5deg); } to { transform: rotate(3.5deg); } }
#story .sc.on .swing { animation: st-rock 4.6s ease-in-out infinite alternate; transform-origin: 1010px 0px; }
#story .sc.on .rise { animation: st-rise 4.5s .4s cubic-bezier(.3,.7,.3,1) both; }
@keyframes st-rise { from { transform: translateY(170px); } to { transform: translateY(0); } }
#story .mote { animation: st-mote var(--t, 9s) var(--d, 0s) linear infinite; }
@keyframes st-mote { 0% { transform: translate(0, 0); opacity: 0; } 15% { opacity: var(--o, .7); } 85% { opacity: var(--o, .7); } 100% { transform: translate(var(--mx, 30px), var(--my, 60px)); opacity: 0; } }
#story .sc.on .unroll { animation: st-unroll 1.6s .2s cubic-bezier(.5,0,.2,1) both; transform-origin: 800px 450px; }
@keyframes st-unroll { from { transform: scaleX(0.02); } to { transform: scaleX(1); } }
#story .sc.on .moon { animation: st-moon 12s linear forwards; }
@keyframes st-moon { from { transform: translate(-70px, 150px); } to { transform: translate(210px, -40px); } }
#story .sc.on .dawn { animation: st-fin 5s 7s ease forwards; }
#story .sc.on .glow { animation: st-glow 9s 2.2s linear both; transform-origin: 620px 470px; }
@keyframes st-glow { 0% { opacity: 0; transform: scale(.3); } 5% { opacity: .9; transform: scale(.5); } 8% { opacity: .1; } 13% { opacity: .8; } 16% { opacity: .2; } 24% { opacity: 1; transform: scale(1); } 60% { opacity: .82; transform: scale(.96); } 100% { opacity: 1; transform: scale(1.03); } }
#story .sc.on .needle { animation: st-needle 5s 2.4s cubic-bezier(.3,1.5,.5,1) both; transform-origin: 1010px 560px; }
@keyframes st-needle { from { transform: rotate(-60deg); } to { transform: rotate(48deg); } }
#story .sc.on .bike { animation: st-bike 13s .6s linear both; }
@keyframes st-bike {
  0% { transform: translate(560px, 700px) rotate(0deg); } 5% { transform: translate(560px, 700px) rotate(0deg); }
  14% { transform: translate(580px, 590px) rotate(-5deg); } 18% { transform: translate(610px, 580px) rotate(14deg); }
  23% { transform: translate(690px, 694px) rotate(38deg); } 31% { transform: translate(690px, 694px) rotate(38deg); opacity: 1; }
  33% { transform: translate(690px, 694px) rotate(38deg); opacity: 0; } 34% { transform: translate(560px, 700px) rotate(0deg); opacity: 0; } 37% { transform: translate(560px, 700px) rotate(0deg); opacity: 1; }
  48% { transform: translate(600px, 470px) rotate(-4deg); } 53% { transform: translate(700px, 440px) rotate(-30deg); }
  57% { transform: translate(860px, 520px) rotate(-100deg); } 61% { transform: translate(960px, 690px) rotate(-170deg); } 69% { transform: translate(960px, 690px) rotate(-170deg); opacity: 1; }
  71% { transform: translate(960px, 690px) rotate(-170deg); opacity: 0; } 72% { transform: translate(560px, 700px) rotate(0deg); opacity: 0; } 75% { transform: translate(560px, 700px) rotate(0deg); opacity: 1; }
  86% { transform: translate(600px, 520px) rotate(-3deg); } 90% { transform: translate(640px, 500px) rotate(3deg); } 94% { transform: translate(690px, 490px) rotate(-2deg); } 100% { transform: translate(760px, 470px) rotate(-1deg); }
}
#story .sc.on .jet { animation: st-jet .09s linear infinite alternate; }
@keyframes st-jet { from { opacity: .95; } to { opacity: .5; } }
#story .sc.on .puff { animation: st-puff 1.6s var(--d, 0s) ease-out both; transform-origin: var(--ox, 0px) 700px; }
@keyframes st-puff { 0% { opacity: 0; transform: scale(.2); } 12% { opacity: .8; } 100% { opacity: 0; transform: scale(1.6); } }
#story .note { opacity: 0; }
#story .sc.on .note { animation: st-note var(--t, 3.6s) var(--d, 0s) ease both; }
@keyframes st-note { 0% { opacity: 0; } 10% { opacity: 1; } 85% { opacity: 1; } 100% { opacity: var(--end, 0); } }
#story .sub { position: absolute; left: 0; right: 0; bottom: 9%; display: flex; justify-content: center; padding: 0 5%; pointer-events: none; }
#story .sub span { max-width: 980px; text-align: center; font-size: clamp(17px, 2.5vw, 30px); line-height: 1.45; font-weight: 500; color: #fff; text-shadow: 0 2px 10px rgba(0,0,0,.95), 0 0 3px rgba(0,0,0,.9); background: rgba(6,10,16,.42); padding: 8px 22px; border-radius: 12px; opacity: 0; transition: opacity .6s; }
#story .sub span.on { opacity: 1; }
#story .ttl { position: absolute; left: 0; right: 0; top: 7%; text-align: center; font-size: clamp(48px, 9vw, 120px); font-weight: 900; letter-spacing: .12em; direction: ltr; color: #fff; text-shadow: 0 6px 40px rgba(0,0,0,.7); opacity: 0; transform: scale(.94); transition: opacity 2s, transform 6s ease-out; pointer-events: none; }
#story .ttl b { color: ${ACC}; font-weight: 900; } #story .ttl.on { opacity: 1; transform: scale(1); }
#story .skip { position: absolute; top: 16px; inset-inline-end: 18px; font: inherit; font-size: 14px; font-weight: 600; color: #fff; background: rgba(10,16,24,.5); border: 1px solid rgba(255,255,255,.22); border-radius: 10px; padding: 7px 16px; cursor: pointer; opacity: .75; }
#story .skip:hover { opacity: 1; border-color: #ffc440; }
#story .gate { position: absolute; inset: 0; display: none; flex-direction: column; align-items: center; justify-content: center; gap: 14px; padding: 20px; text-align: center; background: rgba(6,10,16,.72); }
#story.gated .gate { display: flex; } #story.gated .skip, #story.gated .bar { display: none; }
#story .gate .gt { font-size: clamp(54px, 11vw, 130px); font-weight: 900; letter-spacing: .12em; direction: ltr; line-height: 1; } #story .gate .gt b { color: ${ACC}; font-weight: 900; }
#story .gate p { margin: 0 0 8px; font-size: clamp(16px, 2.2vw, 22px); opacity: .9; }
#story .gate > div { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }
#story .gate button { font: inherit; font-weight: 700; font-size: 17px; padding: 12px 26px; border-radius: 12px; cursor: pointer; color: #fff; background: rgba(10,16,24,.6); border: 1px solid rgba(255,255,255,.25); }
#story .gate button.gp { background: ${ACC}; border-color: transparent; color: #1a0d04; } #story .gate button:hover { border-color: #ffc440; }
#story .gate button.gd { margin-top: 26px; font-size: 13.5px; font-weight: 500; padding: 7px 14px; background: none; border: 1px dashed rgba(255,255,255,.3); opacity: .7; }
#story .gate button.gd:hover { opacity: 1; }
#story i { font-style: normal; font-weight: 800; font-size: .82em; letter-spacing: .04em; color: #1a0d04; background: #ffc440; border-radius: 5px; padding: 1px 7px; margin-inline-end: 8px; }
#story .dv { position: absolute; inset: 0; display: none; background: #000; }
#story.draft .dv { display: block; } #story.draft .skip, #story.draft .bar, #story.draft .sub, #story.draft svg { display: none; }
#story .dv video { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; background: #000; }
#story .dtag { position: absolute; top: 16px; inset-inline-start: 18px; font-size: 14px; color: #fff; background: rgba(10,16,24,.6); border-radius: 9px; padding: 6px 12px; pointer-events: none; }
#story .dx { position: absolute; top: 14px; inset-inline-end: 16px; font: inherit; font-size: 16px; color: #fff; background: rgba(10,16,24,.6); border: 1px solid rgba(255,255,255,.25); border-radius: 10px; padding: 6px 13px; cursor: pointer; }
#story .bar { position: absolute; left: 0; bottom: 0; height: 3px; width: 0; background: ${ACC}; opacity: .8; }
body.story-on #menu, body.story-on #mute { opacity: 0 !important; pointer-events: none; }
#menu { transition: opacity 1s; }
`;

// ---- small drawing helpers ----
const dr = (d, o = {}) => `<path class="dr" pathLength="1" d="${d}" stroke="${o.c || INK}" stroke-width="${o.w || 3}" stroke-linecap="round" stroke-linejoin="round" style="--d:${o.d || 0}s;--t:${o.t || 2}s" ${o.x || ''}/>`;
const fi = (inner, d = 0, t = 1.2, o = 1) => `<g class="fi" style="--d:${d}s;--t:${t}s;--o:${o}">${inner}</g>`;
const hand = 'font-family="\'Segoe Script\',\'Bradley Hand\',\'Comic Sans MS\',cursive"';
const mono = 'font-family="\'Courier New\',monospace"';

// a man seen from behind, feet at the origin, about 290 high. hat: the fedora
const man = (x, y, s = 1, hat = true, fill = INK) => `<g transform="translate(${x},${y}) scale(${s})" fill="${fill}">
  <path d="M-23,0 L-19,-125 L19,-125 L23,0 L6,0 L0,-98 L-6,0 Z"/>
  <path d="M-42,-112 C-50,-175 -48,-222 -31,-240 L31,-240 C48,-222 50,-175 42,-112 Z"/>
  <rect x="-8" y="-250" width="16" height="14"/><circle cx="0" cy="-262" r="19"/>
  ${hat ? '<ellipse cx="0" cy="-274" rx="35" ry="6.5"/><path d="M-21,-275 C-22,-303 22,-303 21,-275 Z"/>' : ''}</g>`;
const woman = (x, y, s = 1, fill = INK) => `<g transform="translate(${x},${y}) scale(${s})" fill="${fill}">
  <rect x="-15" y="-30" width="9" height="30"/><rect x="6" y="-30" width="9" height="30"/>
  <path d="M-29,-228 C-40,-170 -54,-85 -62,-24 L62,-24 C54,-85 40,-170 29,-228 Z"/>
  <rect x="-6" y="-240" width="12" height="14"/><circle cx="0" cy="-252" r="17"/><circle cx="9" cy="-270" r="10"/></g>`;
// the F-5 from the side: no wheels, lift nozzles where they would be, one nozzle behind, and its rider
const bike = (stroke, fill, jets = true) => `<g stroke="${stroke}" stroke-width="3" stroke-linejoin="round" fill="${fill}">
  ${jets ? `<g class="jet"><path d="M-64,-8 L-42,-8 L-47,46 L-59,46 Z" fill="${CY}" stroke="none" opacity=".75"/><path d="M42,-10 L64,-10 L59,44 L47,44 Z" fill="${CY}" stroke="none" opacity=".75"/></g>` : ''}
  <path d="M-98,-36 C-82,-58 -42,-66 0,-64 C32,-63 62,-58 102,-42 L106,-34 C72,-26 22,-22 -28,-24 L-88,-28 Z"/>
  <rect x="-66" y="-26" width="26" height="16" rx="3"/><rect x="40" y="-28" width="26" height="16" rx="3"/><rect x="-118" y="-46" width="22" height="16" rx="3"/>
  <path d="M-34,-64 C-30,-92 -4,-106 20,-100 L44,-76 L22,-64 Z"/><circle cx="32" cy="-108" r="13"/>
  <path d="M20,-96 L62,-72 L60,-64" fill="none"/></g>`;

function sceneLake() {
  let bal = ''; for (let x = 20; x < 1600; x += 58) bal += `<rect x="${x}" y="742" width="12" height="90" rx="5"/>`;
  return `<rect width="1600" height="900" fill="url(#st-paper)"/><g class="kb" style="--z:1.08;--px:-14px;--py:-8px">
    ${fi(`<text x="130" y="210" font-family="Georgia,serif" font-size="150" fill="${INK}" opacity=".16" font-style="italic">1969</text>`, .3, 2)}
    ${fi(`<circle cx="1240" cy="190" r="62" fill="#e2a15a" opacity=".45"/>`, .2, 3)}
    ${dr('M0,470 L130,395 L215,440 L335,318 L425,388 L525,292 L605,358 L705,262 L795,348 L905,286 L1015,378 L1125,316 L1235,398 L1355,338 L1475,418 L1600,372', { t: 3.4, d: .2, w: 3.5 })}
    ${dr('M0,520 L160,470 L300,505 L470,452 L640,500 L830,455 L1000,505 L1180,462 L1380,508 L1600,470', { t: 3, d: 1, w: 2, c: '#6b5a48' })}
    ${dr('M318,340 L335,318 L352,338 M508,312 L525,292 L545,314 M688,284 L705,262 L724,286 M1108,336 L1125,316 L1142,334', { t: 1.5, d: 3, w: 2 })}
    ${[0, 1, 2, 3, 4, 5].map((i) => dr(`M${80 + i * 210 + (i % 2) * 90},${560 + (i % 3) * 34} h${150 + (i % 3) * 60}`, { t: 1.2, d: 2 + i * .25, w: 1.6, c: '#7d6b57' })).join('')}
    <g class="plane"><g fill="${INK}"><ellipse cx="0" cy="0" rx="46" ry="6.5"/><rect x="-10" y="-2" width="26" height="4"/><path d="M-8,-3 L14,-3 L22,-30 L8,-30 Z M-8,3 L14,3 L22,30 L8,30 Z" transform="scale(1,.35)"/><path d="M-46,0 L-58,-16 L-48,-16 L-34,-2 Z"/><circle cx="48" cy="0" r="3"/></g></g>
    ${fi(man(985, 852, 1.08) + woman(1108, 852, 1.02), 1.6, 2.2)}
    <g fill="${INK}"><rect x="0" y="726" width="1600" height="18"/>${bal}<rect x="0" y="826" width="1600" height="80"/></g></g>`;
}
function sceneBorn() {
  return `<rect width="1600" height="900" fill="url(#st-paper)"/><g class="kb" style="--z:1.09;--px:-22px;--py:6px">
    ${fi(`<text x="1130" y="230" font-family="Georgia,serif" font-size="150" fill="${INK}" opacity=".16" font-style="italic">1970</text>`, .3, 2)}
    ${dr('M250,150 h520 v470 h-520 Z M510,150 v470 M250,385 h520', { t: 2.2, w: 5 })}
    <g clip-path="url(#st-win)">${dr('M250,470 L330,400 L410,450 L500,350 L590,430 L680,370 L770,440', { t: 2.4, d: 1, w: 2.5 })}${fi(`<circle cx="640" cy="240" r="34" fill="#e2a15a" opacity=".5"/>`, 1, 2)}</g>
    ${dr('M225,140 C215,300 240,470 220,640 M795,140 C805,300 780,470 800,640', { t: 2, d: .8, w: 2.5, c: '#6b5a48' })}
    ${dr('M0,790 H1600', { t: 1.6, w: 3 })}
    <g class="swing">${dr('M1010,0 V300 M930,300 H1090 M940,300 V350 M1010,300 V372 M1080,300 V342', { t: 1.6, d: 1.2, w: 2 })}
      ${fi(`<g fill="${INK}"><g transform="translate(940,362) scale(.5)"><ellipse rx="46" ry="6.5"/><path d="M-8,-10 h22 l6,-12 h-12 Z M-8,10 h22 l6,12 h-12 Z"/><path d="M-46,0 L-58,-16 L-48,-16 L-34,-2 Z"/></g><circle cx="1010" cy="386" r="13"/><path d="M1080,342 l5,12 13,1 -10,8 3,13 -11,-7 -11,7 3,-13 -10,-8 13,-1 Z"/></g>`, 2.2, 1.2)}</g>
    <g class="rock">${fi(`<g fill="${INK}"><path d="M880,600 Q1010,740 1140,600 Z"/><path d="M880,602 C868,510 900,470 968,470 L968,602 Z"/><path d="M850,760 Q1010,806 1170,760 L1170,772 Q1010,820 850,772 Z"/><rect x="925" y="676" width="10" height="92"/><rect x="1085" y="676" width="10" height="92"/></g>`, 1.4, 1.6)}</g>
    ${dr('M1330,260 h200 v530 M1330,260 v530', { t: 2, d: .4, w: 5 })}
    <g class="fo" style="--d:5.4s;--t:3.6s;--o:.92">${fi(man(1432, 790, .98), 2.4, 1.6)}</g></g>`;
}
function sceneAttic() {
  let motes = '';
  for (let i = 0; i < 34; i++) { const u = (i * 0.618) % 1, v = (i * 0.377) % 1; motes += `<circle class="mote" cx="${930 + u * 300 - v * 200}" cy="${250 + v * 480}" r="${1.2 + (i % 3)}" fill="#ffe9b8" style="--t:${7 + (i % 5) * 1.7}s;--d:${-(i * 0.9) % 9}s;--mx:${-30 + (i % 7) * 12}px;--my:${30 + (i % 4) * 22}px;--o:${.35 + (i % 4) * .15}"/>`; }
  return `<rect width="1600" height="900" fill="#120e0b"/><g class="kb" style="--z:1.1;--px:-30px;--py:-16px">
    ${dr('M800,40 L40,900 M800,40 L1560,900 M430,460 H1170 M800,40 V460 M250,662 H1350', { t: 3, w: 9, c: '#5b4329' })}
    ${dr('M615,250 L330,900 M985,250 L1270,900 M505,372 L140,900 M1095,372 L1460,900', { t: 3, d: .6, w: 4, c: '#3d2d1c' })}
    ${fi(`<path d="M1010,180 L1110,292 L1010,860 L640,860 Z" fill="url(#st-beam)"/><path d="M1010,180 L1110,292 L1060,292 L975,200 Z" fill="#fff3cf" opacity=".85"/>`, .8, 2.6)}
    ${fi(motes, 1.6, 2)}
    ${fi(`<g fill="#241a12" stroke="#4a3723" stroke-width="3"><rect x="1180" y="640" width="150" height="130"/><rect x="1210" y="540" width="120" height="100"/><rect x="330" y="690" width="170" height="90"/><rect x="360" y="610" width="110" height="80"/></g>`, 1, 2)}
    ${fi(`<g stroke="#8a6a44" stroke-width="4" fill="#3a2917"><path d="M720,640 h260 v140 h-260 Z"/><path d="M712,640 Q850,520 988,640 Z" transform="rotate(-24 712 640)"/><path d="M770,640 v140 M930,640 v140" fill="none"/><rect x="838" y="690" width="24" height="30" fill="#8a6a44"/></g>
      <g transform="rotate(-18 900 610)"><rect x="868" y="520" width="34" height="150" rx="16" fill="#f1e6c8" stroke="#b9a678" stroke-width="3"/><ellipse cx="885" cy="522" rx="17" ry="7" fill="#fff8e0" stroke="#b9a678" stroke-width="2"/></g>`, 1.4, 2.4)}
    ${fi(`<circle cx="888" cy="560" r="120" fill="url(#st-warm)"/>`, 5.6, 2.5)}
    <rect x="190" y="800" width="230" height="110" fill="#050403"/>
    <g class="rise">${fi(`<path d="M392,726 L770,560 L800,700 Z" fill="url(#st-torch)"/>`, 3, 2)}<g fill="#050403" stroke="#6a5234" stroke-width="2.5"><circle cx="305" cy="700" r="34"/><path d="M228,900 C226,790 258,748 305,748 C352,748 384,790 382,900 Z"/><path d="M352,790 L402,728 L416,740 L378,808 Z"/></g></g>
    ${dr('M190,800 H420', { t: 1, w: 6, c: '#5b4329' })}</g>`;
}
function scenePlan(L) {
  let coils = ''; for (let i = 0; i < 8; i++) coils += `<rect x="-17" y="-213" width="34" height="50" transform="rotate(${i * 45 + 22.5})" class="dr" pathLength="1" stroke="${LINE}" stroke-width="2.5" style="--d:${3 + i * .16}s;--t:.8s"/>`;
  let hex = ''; for (let i = 0; i < 7; i++) { const a = i * Math.PI / 3, r = i === 6 ? 0 : 30; hex += `<circle cx="${(r * Math.cos(a)).toFixed(1)}" cy="${(r * Math.sin(a)).toFixed(1)}" r="5" fill="${LINE}"/>`; }
  const o = { c: LINE };
  return `<rect width="1600" height="900" fill="#081f38"/><g class="kb" style="--z:1.1;--px:10px;--py:-10px"><g class="unroll">
    <rect x="90" y="60" width="1420" height="760" fill="${BLUE}"/><rect x="90" y="60" width="1420" height="760" fill="url(#st-grid)"/>
    <rect x="110" y="80" width="1380" height="720" fill="none" stroke="${LINE}" stroke-width="2" opacity=".6"/>
    <g transform="translate(760,395) scale(.9)">
      ${dr('M-300,0 H300 M0,-300 V300', { ...o, t: 1.4, d: 1.4, w: 1.2 })}
      <circle r="232" class="dr" pathLength="1" stroke="${LINE}" stroke-width="3.5" style="--d:1.6s;--t:1.8s"/>
      <circle r="150" class="dr" pathLength="1" stroke="${LINE}" stroke-width="2.5" style="--d:2.2s;--t:1.6s"/>
      <circle r="62" class="dr" pathLength="1" stroke="${LINE}" stroke-width="3" style="--d:2.8s;--t:1.2s"/>
      ${coils}
      ${dr('M53.7,-31 L0,-62 L-53.7,-31 L-53.7,31 L0,62 L53.7,31 Z', { ...o, t: 1.2, d: 4.2, w: 2 })}
      ${fi(hex, 4.8, 1)}
      ${dr('M-150,0 C-120,-50 -90,50 -62,0 M62,0 C90,-50 120,50 150,0', { ...o, t: 1.4, d: 5, w: 2 })}
      ${dr('M-232,300 H232 M-232,286 V314 M232,286 V314', { ...o, t: 1, d: 5.6, w: 1.6 })}
      ${fi(`<text y="336" text-anchor="middle" ${mono} font-size="22" fill="${LINE}">Ø 46 cm</text>`, 6.2)}
      ${dr('M40,-40 L250,-250 H330', { ...o, t: .8, d: 5.8, w: 1.4 })}${fi(`<text x="338" y="-243" ${mono} font-size="21" fill="${LINE}">Pd LATTICE</text>`, 6.4)}
      ${dr('M140,60 L290,150 H340', { ...o, t: .8, d: 6.2, w: 1.4 })}${fi(`<text x="348" y="157" ${mono} font-size="21" fill="${LINE}">D₂ · 1 atm · 20°C</text>`, 6.8)}
      ${dr('M-160,-120 L-330,-230 H-380', { ...o, t: .8, d: 6.6, w: 1.4 })}${fi(`<text x="-388" y="-223" text-anchor="end" ${mono} font-size="21" fill="${LINE}">RESONANT COILS ×8</text>`, 7.2)}
    </g>
    ${dr('M1180,560 V700 H1400', { ...o, t: 1, d: 7, w: 2 })}${dr('M1185,694 C1260,690 1300,676 1330,640 C1350,612 1368,584 1392,566', { c: CY, t: 1.4, d: 7.6, w: 3 })}
    ${fi(`<text x="1290" y="730" text-anchor="middle" ${mono} font-size="18" fill="${LINE}">OUT / IN</text><text x="1398" y="556" text-anchor="end" ${mono} font-size="20" fill="${CY}">× 40 ?!</text>`, 8.4)}
    ${fi(`<g transform="rotate(-5 300 640)" fill="#ffe9a8" ${hand} font-size="30"><text x="150" y="620">it runs cold.</text><text x="150" y="664">it should not work.</text><text x="150" y="708">it does.   — H.</text></g>`, 8.8, 1.6)}
    <g>${dr('M1130,740 H1470 V790 H1130 Z M1300,740 V790', { ...o, t: 1, d: 1.2, w: 2 })}${fi(`<text x="1215" y="772" text-anchor="middle" ${mono} font-size="19" fill="${LINE}">PROJECT FÖHN</text><text x="1385" y="772" text-anchor="middle" ${mono} font-size="19" fill="${LINE}">H.H. · 1969</text>`, 2)}</g>
  </g></g>`;
}
function sceneStudy() {
  const eq = (x, y, s, txt, d, c = '#e8efe9', r = 0) => fi(`<text x="${x}" y="${y}" ${hand} font-size="${s}" fill="${c}" transform="rotate(${r} ${x} ${y})">${txt}</text>`, d, 1.4, .9);
  let stars = ''; for (let i = 0; i < 14; i++) stars += `<circle cx="${1290 + (i * 61) % 200}" cy="${132 + (i * 37) % 230}" r="1.6" fill="#fff" opacity=".8"/>`;
  return `<rect width="1600" height="900" fill="#0a121c"/><g class="kb" style="--z:1.08;--px:-18px;--py:-6px">
    <rect x="110" y="100" width="960" height="520" fill="#12241f" stroke="#5b4329" stroke-width="14"/>
    ${eq(170, 190, 44, 'F = ma', .6)}${eq(430, 200, 44, 'E = mc²', 1.5, '#e8efe9', -2)}${eq(720, 190, 38, '∇·E = ρ/ε₀', 2.4, '#e8efe9', 2)}
    ${eq(170, 300, 40, 'iħ ∂ψ/∂t = Ĥψ', 3.3)}${eq(600, 310, 36, 'PV = nRT', 4.1, '#e8efe9', -3)}${eq(180, 400, 34, 'Pd + D₂ ⇌ PdDₓ', 5, '#e8efe9', 1)}${eq(660, 410, 34, 'Γ ∝ e^(−2πη)  ← ?', 5.8, '#e8efe9', -1)}
    ${fi(`<text x="250" y="530" ${hand} font-size="52" fill="#ffe9a8">D + D → ⁴He + 23.8 MeV</text>`, 7.4, 1.6)}
    ${dr('M215,545 C200,470 420,455 620,462 C860,468 960,490 950,530 C940,575 640,580 440,576 C300,574 222,570 232,520', { c: CY, t: 1.6, d: 8.8, w: 3.5 })}
    <g clip-path="url(#st-win2)"><rect x="1270" y="110" width="240" height="280" fill="#0d1a33"/>${stars}<g class="moon"><circle cx="1330" cy="250" r="22" fill="#f4efd8"/></g><rect class="dawn fi" x="1270" y="110" width="240" height="280" fill="#e79a5a" style="--o:.75"/></g>
    <path d="M1270,110 h240 v280 h-240 Z M1390,110 v280" fill="none" stroke="#3a2c1c" stroke-width="8"/>
    ${fi(`<path d="M1210,520 L1010,780 L1420,780 Z" fill="url(#st-lamp)"/>`, .6, 2)}
    <g stroke="#c9a45a" stroke-width="5" fill="none"><path d="M1330,780 L1300,600 L1230,520"/></g><path d="M1180,538 L1218,494 L1262,530 Z" fill="#c9a45a"/>
    <rect x="930" y="780" width="670" height="16" fill="#3a2c1c"/><rect x="960" y="796" width="20" height="110" fill="#2a2016"/>
    <g stroke="#0a121c" stroke-width="2"><rect x="1010" y="744" width="170" height="36" fill="#7a3b2e"/><rect x="1024" y="712" width="150" height="32" fill="#2f5d62"/><rect x="1004" y="682" width="164" height="30" fill="#8a7a3a"/><rect x="1030" y="654" width="130" height="28" fill="#4a3d6b"/></g>
    <g fill="#04070b" stroke="#c9a45a" stroke-width="2" stroke-opacity=".5"><circle cx="1390" cy="610" r="40"/><path d="M1290,900 C1288,760 1320,668 1390,668 C1460,668 1492,760 1490,900 Z"/></g></g>`;
}
function sceneCore() {
  let coils = ''; for (let i = 0; i < 8; i++) coils += `<rect x="-15" y="-176" width="30" height="44" rx="4" transform="rotate(${i * 45 + 22.5})" fill="#b87333" stroke="#5c391a" stroke-width="2"/>`;
  let peg = ''; for (let i = 0; i < 9; i++) peg += `<rect x="${130 + i * 36}" y="${130 + (i % 3) * 14}" width="${8 + (i % 2) * 8}" height="${90 + (i % 4) * 24}" rx="3"/>`;
  return `<rect width="1600" height="900" fill="#07090d"/><g class="kb" style="--z:1.09;--px:14px;--py:-8px">
    <g fill="#141a22">${peg}</g>
    <circle class="glow" cx="620" cy="470" r="460" fill="url(#st-core)"/>
    <rect x="0" y="690" width="1600" height="14" fill="#2a323d"/><rect x="0" y="704" width="1600" height="200" fill="#0c1016"/>
    <rect x="520" y="640" width="200" height="50" fill="#1b222b" stroke="#3b4653" stroke-width="3"/>
    <g transform="translate(620,470)"><circle r="190" fill="none" stroke="#8fa3b5" stroke-width="10"/><circle r="124" fill="#0b0f14" stroke="#8fa3b5" stroke-width="6"/>${coils}<circle r="52" fill="#0b0f14" stroke="#c9d6e2" stroke-width="5"/>
      <circle class="glow" r="46" fill="#dffaff" style="transform-origin:0 0"/></g>
    <path d="M790,560 C880,640 900,560 960,580" fill="none" stroke="#a23b2a" stroke-width="6"/><path d="M800,520 C870,480 910,520 960,540" fill="none" stroke="#2a3b55" stroke-width="6"/>
    <rect x="950" y="490" width="120" height="200" rx="8" fill="#1b222b" stroke="#3b4653" stroke-width="3"/><circle cx="1010" cy="560" r="40" fill="#e9e4d4"/><path class="needle" d="M1010,560 L1010,526" stroke="#c0271c" stroke-width="4" stroke-linecap="round"/><circle cx="1010" cy="560" r="5" fill="#222"/>
    <g class="fi" style="--d:5.4s;--t:2.6s;--o:.9" stroke="${CY}" stroke-opacity=".7" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" fill="#0d1218">
      <circle cx="1215" cy="632" r="56"/><circle cx="1215" cy="632" r="20"/><circle cx="1465" cy="632" r="56"/><circle cx="1465" cy="632" r="20"/>
      <path d="M1215,632 L1290,560 L1400,560 L1340,640 L1280,640 Z"/><path d="M1465,632 L1412,508 M1392,500 L1436,490" fill="none" stroke-width="5"/>
      <path d="M1296,548 C1310,518 1380,514 1404,548 Z"/><path d="M1180,548 C1210,530 1262,532 1296,548 L1290,560 L1196,560 Z"/>
      <rect x="1296" y="574" width="62" height="56" rx="8"/><path d="M1300,640 L1170,654" fill="none" stroke-width="7"/><circle cx="1412" cy="512" r="13"/></g>
    ${fi(man(250, 700, 1.02, false, '#04070b'), .2, 1)}</g>`;
}
function sceneCrash(L) {
  let hatch = ''; for (let x = 20; x < 1600; x += 36) hatch += `M${x},700 l-18,22 `;
  const note = (y, txt, d, t, end = 0, c = '#ffe9a8') => `<text class="note" x="1130" y="${y}" text-anchor="middle" direction="${L('rtl', 'ltr')}" ${hand} font-size="32" fill="${c}" style="--d:${d}s;--t:${t}s;--end:${end}">${txt}</text>`;
  return `<rect width="1600" height="900" fill="#0a2440"/><rect width="1600" height="900" fill="url(#st-grid)"/><g class="kb" style="--z:1.05;--px:-10px;--py:0px">
    ${dr('M0,700 H1600', { c: LINE, t: 1.4, w: 3 })}${dr(hatch, { c: LINE, t: 2, d: .4, w: 1.2 })}
    ${dr('M90,700 V520 Q240,430 390,520 V700 M90,520 H390 M170,700 V590 H310 V700', { c: LINE, t: 2.4, d: .3, w: 2.5 })}
    ${dr('M1440,700 V470 M1440,480 L1520,492 L1440,512', { c: LINE, t: 1.4, d: .8, w: 2.5 })}
    <circle class="puff" cx="690" cy="690" r="70" fill="${LINE}" style="--d:3.5s;--ox:690px"/><circle class="puff" cx="960" cy="690" r="95" fill="${LINE}" style="--d:8.4s;--ox:960px"/>
    <g class="bike">${bike(LINE, '#0f3357')}</g>
    ${note(200, L('ניסוי 4: נכשל', 'Test 4: failed'), 3.6, 3.4, .35)}
    ${note(256, L('ניסוי 17: נכשל. שלוש צלעות ויד', 'Test 17: failed. Three ribs and an arm'), 8.4, 3.6, .35, '#ffb19a')}
    ${note(312, L('ניסוי 43:', 'Test 43:'), 11.2, 3, 1)}</g>`;
}
function sceneFly() {
  return `<rect width="1600" height="900" fill="url(#st-vig)"/>`;
}
const DEFS = `<defs>
  <radialGradient id="st-paper" cx="50%" cy="45%" r="75%"><stop offset="0" stop-color="#f1e6cc"/><stop offset=".7" stop-color="${PAPER}"/><stop offset="1" stop-color="#c8b48f"/></radialGradient>
  <linearGradient id="st-beam" x1="1" y1="0" x2=".3" y2="1"><stop offset="0" stop-color="#ffe9b0" stop-opacity=".5"/><stop offset="1" stop-color="#ffe9b0" stop-opacity=".03"/></linearGradient>
  <linearGradient id="st-torch" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#fff2c4" stop-opacity=".55"/><stop offset="1" stop-color="#fff2c4" stop-opacity="0"/></linearGradient>
  <linearGradient id="st-lamp" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffd98a" stop-opacity=".5"/><stop offset="1" stop-color="#ffd98a" stop-opacity=".04"/></linearGradient>
  <radialGradient id="st-warm"><stop offset="0" stop-color="#fff2c4" stop-opacity=".55"/><stop offset="1" stop-color="#fff2c4" stop-opacity="0"/></radialGradient>
  <radialGradient id="st-core"><stop offset="0" stop-color="#dffaff" stop-opacity=".95"/><stop offset=".18" stop-color="${CY}" stop-opacity=".6"/><stop offset=".55" stop-color="#1c7fb8" stop-opacity=".18"/><stop offset="1" stop-color="#1c7fb8" stop-opacity="0"/></radialGradient>
  <linearGradient id="st-vig" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity=".4"/><stop offset=".25" stop-color="#000" stop-opacity="0"/><stop offset=".7" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".55"/></linearGradient>
  <pattern id="st-grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40,0 H0 V40" fill="none" stroke="#fff" stroke-opacity=".07" stroke-width="1"/></pattern>
  <clipPath id="st-win"><rect x="250" y="150" width="520" height="470"/></clipPath><clipPath id="st-win2"><rect x="1270" y="110" width="240" height="280"/></clipPath>
</defs>`;

export const STORY_SCENES = SCENES.length;
export function playStory({ lang = 'he', audio = null, from = 0, onEnd = null, onScene = null, gate = false, wake = null, first = true, draft = false } = {}) {
  const L = (he, en) => (lang === 'en' ? en : he);
  if (!document.getElementById('story-css')) { const s = document.createElement('style'); s.id = 'story-css'; s.textContent = CSS; document.head.appendChild(s); }
  const root = document.createElement('div'); root.id = 'story';
  const draw = { lake: sceneLake, born: sceneBorn, attic: sceneAttic, plan: scenePlan, study: sceneStudy, core: sceneCore, crash: sceneCrash, fly: sceneFly };
  root.innerHTML = `<svg xmlns="${NS}" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice">${DEFS}${SCENES.map((s) => `<g class="sc" data-id="${s.id}" style="--dur:${s.dur + 4}s"></g>`).join('')}</svg>
    <div class="ttl">F<b>Ö</b>HN</div><div class="sub"><span></span></div><button class="skip" type="button">${L('דילוג', 'Skip')} ›</button><div class="bar"></div>
    <div class="gate"><div class="gt">F<b>Ö</b>HN</div><p>${first ? L('לפני שטסים: הסיפור של יקיר, בפחות משתי דקות', 'Before you fly: Yakir\'s story, in under two minutes') : L('הסיפור של יקיר, בפחות משתי דקות', 'Yakir\'s story, in under two minutes')}</p>
      <div><button class="gp" type="button">▶ ${L('לצפות בסיפור', 'Watch the story')}</button><button class="gs" type="button">${first ? L('ישר למשחק', 'Straight to the game') : L('חזרה', 'Back')}</button></div>
      ${draft ? `<button class="gd" type="button"><i>${L('טיוטה', 'Draft')}</i>${L('ניסוי ראשון בגרסה מצולמת, נוצרה בבינה מלאכותית', 'A first try at a live-action version, made with AI')} ›</button>` : ''}</div>
    <div class="dv"><video playsinline controls preload="none" poster="${new URL('../../assets/story/draft.jpg', import.meta.url).href}"></video>
      <div class="dtag"><i>${L('טיוטה', 'Draft')}</i>${L('ניסוי ראשון. הסרט הזה עוד ישתנה', 'A first try. This film will change')}</div><button class="dx" type="button">✕</button></div>`;
  document.body.appendChild(root); document.body.classList.add('story-on');
  const groups = [...root.querySelectorAll('.sc')], sub = root.querySelector('.sub span'), ttl = root.querySelector('.ttl'), bar = root.querySelector('.bar');
  // The timing follows the narration: a line never starts before the one before it has been said,
  // and a scene lasts until its last line is over.
  const VL = VOICE_LEN[lang] || [];
  const TL = SCENES.map((s, i) => {
    let free = 0;
    const subs = s.subs.map((u, k) => { const at = Math.max(u[0], free), d = (VL[i] && VL[i][k]) || 0; free = at + d + 0.45; return { at, d, text: L(u[1], u[2]), url: d ? new URL(`../../assets/story/${lang}/${s.id}_${k}.mp3`, import.meta.url).href : null }; });
    return { id: s.id, subs, dur: Math.max(s.dur, free + 1.1) };
  });
  const total = TL.reduce((a, s) => a + s.dur, 0), starts = []; { let a = 0; for (const s of TL) { starts.push(a); a += s.dur; } }
  let cur = -1, subK = '', done = false, running = false, raf = 0, t0 = 0, pad = { scene() {}, stop() {} }, voice = null;
  const clips = new Map();
  const clip = (url) => { if (!clips.has(url)) { const a = new window.Audio(url); a.preload = 'auto'; clips.set(url, a); } return clips.get(url); };
  const say = (u) => {
    if (voice) { voice.pause(); voice = null; }
    if (!u || !u.url || (audio && audio.muted)) return;
    voice = clip(u.url); try { voice.currentTime = 0; } catch { /* not loaded yet */ }
    voice.volume = 1; const p = voice.play(); if (p && p.catch) p.catch(() => {});      // (no click yet: the browser keeps it silent, the subtitles carry the film)
  };

  function show(i) {
    if (cur >= 0) { const g = groups[cur]; g.classList.remove('on'); setTimeout(() => { if (cur !== groups.indexOf(g)) g.innerHTML = ''; }, 1600); }
    cur = i; const s = TL[i], g = groups[i];
    g.innerHTML = draw[s.id](L);                    // (built when shown, so that every animation starts from its beginning)
    void g.getBoundingClientRect(); g.classList.add('on');
    root.classList.toggle('reveal', s.id === 'fly');
    for (const u of s.subs) if (u.url) clip(u.url);                                    // start fetching this scene's lines, and the next one's
    if (TL[i + 1]) for (const u of TL[i + 1].subs) if (u.url) clip(u.url);
    pad.scene(i); if (onScene) onScene(s.id);
  }
  function tick(now) {
    if (done) return;
    const T = (now - t0) / 1000;
    if (T >= total) { stop(); return; }
    let i = TL.length - 1; while (i > 0 && T < starts[i]) i--;
    if (i !== cur) show(i);
    const ts = T - starts[i], s = TL[i];
    let line = null; for (const u of s.subs) if (ts >= u.at) line = u;
    const hide = ts > s.dur - 0.7 || !line, key = hide ? '' : i + ':' + line.at;
    if (key !== subK) {
      subK = key; sub.classList.remove('on');
      if (!hide) { say(line); setTimeout(() => { if (subK === key) { sub.textContent = line.text; sub.classList.add('on'); } }, 350); }
    }
    ttl.classList.toggle('on', s.id === 'fly' && ts > 5.5 && ts < s.dur - 0.8);
    bar.style.width = (100 * T / total).toFixed(2) + '%';
    raf = requestAnimationFrame(tick);
  }
  // the draft of a live-action film (made with Google Flow): a side door, clearly marked as a draft
  function showDraft() {
    if (running || done) return; running = true;
    root.classList.remove('gated', 'reveal'); root.classList.add('draft');
    const v = root.querySelector('video'); v.src = new URL('../../assets/story/draft.mp4', import.meta.url).href;
    v.muted = !!(audio && audio.muted); v.onended = stop; const p = v.play(); if (p && p.catch) p.catch(() => {});
  }
  function begin() {
    if (running || done) return; running = true;
    root.classList.remove('gated', 'reveal');
    if (wake) wake();                                 // (we are inside a click or a key here: the sound may start)
    pad = makePad(audio);
    t0 = performance.now() - 1000 * (starts[Math.max(0, Math.min(TL.length - 1, from))] || 0);
    raf = requestAnimationFrame(tick);
  }
  function stop() {
    if (done) return; done = true; cancelAnimationFrame(raf); removeEventListener('keydown', onKey, true);
    if (voice) voice.pause(); for (const a of clips.values()) a.src = '';
    const dv = root.querySelector('video'); dv.pause(); dv.removeAttribute('src'); dv.load();
    pad.stop(); root.classList.remove('on'); document.body.classList.remove('story-on');
    setTimeout(() => root.remove(), 950);
    if (onEnd) onEnd();
  }
  function onKey(e) {
    e.stopImmediatePropagation();                     // the game below must not see the keys
    if (!running) { if (e.code === 'Enter' || e.code === 'Space') { e.preventDefault(); begin(); } else if (e.code === 'Escape') stop(); return; }
    if (root.classList.contains('draft')) { if (e.code === 'Escape') stop(); return; }   // (space and the arrows belong to the video player)
    if (e.code === 'Escape' || e.code === 'Enter' || e.code === 'Space') { e.preventDefault(); stop(); }
  }
  addEventListener('keydown', onKey, true);
  root.querySelector('.skip').onclick = stop; root.querySelector('.gs').onclick = stop; root.querySelector('.gp').onclick = begin; root.querySelector('.dx').onclick = stop;
  if (draft) root.querySelector('.gd').onclick = showDraft;
  if (gate) root.classList.add('gated', 'reveal');
  requestAnimationFrame(() => { root.classList.add('on'); if (!gate) begin(); });
  return { stop, get scene() { return cur; }, get total() { return total; }, get speaking() { return !!voice && !voice.paused && voice.currentTime > 0; } };
}

// A quiet pad under the film: two detuned voices per note through a slow low-pass, one chord per scene.
// Only if the game's audio is already running (the browser lets no sound out before the first key or click).
function makePad(audio) {
  const none = { scene() {}, stop() {} };
  const c = audio && audio.ctx; if (!c || audio.muted) return none;
  if (c.state !== 'running') c.resume().catch(() => {});   // (right after the click that made it, it may still be waking up)
  const out = c.createGain(); out.gain.value = 0; const lp = c.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 700; lp.Q.value = 0.4;
  lp.connect(out); out.connect(audio.master || c.destination);
  out.gain.linearRampToValueAtTime(0.2, c.currentTime + 3);     // (quiet: it sits under the narrator)
  // D minor, wandering, and opening to D major when the bike finally flies
  const CH = [[146.8, 220, 349.2], [130.8, 196, 329.6], [116.5, 174.6, 293.7], [110, 164.8, 277.2], [146.8, 220, 349.2], [174.6, 261.6, 392], [98, 146.8, 233.1], [146.8, 220, 370, 440]];
  let voices = [];
  const scene = (i) => {
    const now = c.currentTime;
    for (const v of voices) { v.g.gain.cancelScheduledValues(now); v.g.gain.setTargetAtTime(0, now, 0.9); v.o.forEach((o) => o.stop(now + 5)); }
    voices = CH[i % CH.length].map((f, k) => {
      const g = c.createGain(); g.gain.value = 0; g.gain.setTargetAtTime(k ? 0.22 : 0.3, now, 1.2); g.connect(lp);
      const o = [-4, 4].map((det) => { const os = c.createOscillator(); os.type = k === 0 ? 'triangle' : 'sawtooth'; os.frequency.value = f; os.detune.value = det + k * 2; os.connect(g); os.start(now); return os; });
      return { g, o };
    });
    lp.frequency.setTargetAtTime(i === CH.length - 1 ? 2200 : 600 + 120 * (i % 3), now, 2);
  };
  const stop = () => { const now = c.currentTime; out.gain.cancelScheduledValues(now); out.gain.setTargetAtTime(0, now, 0.4); for (const v of voices) v.o.forEach((o) => o.stop(now + 3)); };
  return { scene, stop };
}
