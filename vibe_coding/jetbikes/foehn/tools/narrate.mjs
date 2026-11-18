// Narration for the opening film: one mp3 per subtitle line and language, spoken by Azure neural voices.
//   node tools/narrate.mjs [he|en] [--force]
// Reads the lines from js/ui/story.js (SCENES), writes assets/story/<lang>/<scene>_<n>.mp3 and
// js/ui/story_voice.js (the length of every line, which the film uses for its timing).
// The key is the shared one in ~/.config/ai/azure.env (never in the repo). About 3,000 characters in all.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { SCENES } from '../js/ui/story.js';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const env = Object.fromEntries(fs.readFileSync(path.join(os.homedir(), '.config/ai/azure.env'), 'utf8').split('\n').filter((l) => /^[A-Z_]+=/.test(l)).map((l) => { const i = l.indexOf('='); return [l.slice(0, i), l.slice(i + 1).trim().replace(/^["']|["']$/g, '')]; }));
const VOICE = { he: ['he-IL', 'he-IL-Hila:DragonLatestNeural'], en: ['en-US', process.env.EN_VOICE || 'en-US-Ava:DragonHDLatestNeural'] };
// spelling for the voice only, where it says a word wrong (the subtitle keeps the normal spelling)
// (checked by letting speech recognition listen in English: the plain spelling comes out as "haverd", this one as "Howard")
const SPOKEN = { he: { 'הווארד יוז': 'הַאוַּרְד יוּז' }, en: {} };
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const args = process.argv.slice(2), force = args.includes('--force'), langs = args.filter((a) => a === 'he' || a === 'en');
const out = {};
for (const lang of ['he', 'en']) {
  const dir = path.join(ROOT, 'assets/story', lang); fs.mkdirSync(dir, { recursive: true });
  out[lang] = [];
  for (const sc of SCENES) {
    const row = [];
    for (let k = 0; k < sc.subs.length; k++) {
      const file = path.join(dir, `${sc.id}_${k}.mp3`);
      if ((!langs.length || langs.includes(lang)) && (force || !fs.existsSync(file))) {
        let text = sc.subs[k][lang === 'he' ? 1 : 2];
        for (const [w, s] of Object.entries(SPOKEN[lang])) text = text.split(w).join(s);
        const [xl, voice] = VOICE[lang];
        const ssml = `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="${xl}"><voice name="${voice}">${esc(text)}</voice></speak>`;
        const r = await fetch(`https://${env.AZURE_AI_REGION}.tts.speech.microsoft.com/cognitiveservices/v1`, { method: 'POST', body: ssml,
          headers: { 'Ocp-Apim-Subscription-Key': env.AZURE_OPENAI_API_KEY, 'Content-Type': 'application/ssml+xml', 'User-Agent': 'narrate', 'X-Microsoft-OutputFormat': 'audio-24khz-96kbitrate-mono-mp3' } });
        if (!r.ok) { console.error(`${lang} ${sc.id}_${k}: HTTP ${r.status} ${await r.text()}`); process.exit(1); }
        fs.writeFileSync(file, Buffer.from(await r.arrayBuffer()));
        console.log(`made ${lang}/${sc.id}_${k}.mp3`);
      }
      row.push(fs.existsSync(file) ? +(+execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', file]).toString()).toFixed(2) : 0);
    }
    out[lang].push(row);
  }
}
fs.writeFileSync(path.join(ROOT, 'js/ui/story_voice.js'), `// Written by tools/narrate.mjs: seconds of narration for every subtitle line, per language and scene.\nexport const VOICE_LEN = ${JSON.stringify(out)};\n`);
for (const lang of ['he', 'en']) console.log(lang, JSON.stringify(out[lang]), 'total', out[lang].flat().reduce((a, b) => a + b, 0).toFixed(1), 's');
