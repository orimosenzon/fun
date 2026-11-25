// Builds public/emoji.json: every standard emoji, in the usual order and groups, each as
// [emoji, Hebrew name, more search words]. The browser loads it only when someone opens the full emoji list.
// Order and groups come from emojibase-data, the Hebrew from Unicode's CLDR.
// Run again only after upgrading those packages: node tools/build_emoji.mjs
import { readFileSync, writeFileSync } from 'node:fs';

const read = (path) => JSON.parse(readFileSync(`node_modules/${path}`, 'utf8'));
const all = read('emojibase-data/en/data.json');
const hebrew = {
  ...read('cldr-annotations-derived-full/annotationsDerived/he/annotations.json').annotationsDerived.annotations,
  ...read('cldr-annotations-full/annotations/he/annotations.json').annotations.annotations,
};

// Newer emoji show as empty boxes on phones that are a few years old.
const NEWEST = 15;
// Group 2 holds skin tones and hair parts, which are not emoji on their own.
const GROUPS = [
  { ids: [0], name: 'פרצופים ורגשות', icon: '😀' },
  { ids: [1], name: 'אנשים וגוף', icon: '👋' },
  { ids: [3], name: 'חיות וטבע', icon: '🌳' },
  { ids: [4], name: 'אוכל ושתייה', icon: '🍊' },
  { ids: [5], name: 'מקומות ונסיעות', icon: '🏡' },
  { ids: [6], name: 'פעילויות', icon: '⚽' },
  { ids: [7], name: 'חפצים', icon: '💡' },
  { ids: [8], name: 'סמלים', icon: '💟' },
  { ids: [9], name: 'דגלים', icon: '🏁' },
];

// The name of an emoji in Hebrew, and the other words it is searched by.
const words = (emoji) => {
  const entry = hebrew[emoji] ?? hebrew[emoji.replaceAll('\uFE0F', '')];
  const name = entry?.tts?.[0];
  return name ? [name, (entry.default ?? []).filter((word) => !name.includes(word)).join(' ')] : null;
};

let missing = 0;
const groups = GROUPS.map(({ ids, name, icon }) => ({
  name,
  icon,
  emoji: all
    .filter((item) => ids.includes(item.group) && item.version <= NEWEST)
    .sort((a, b) => a.order - b.order)
    .map((item) => {
      const found = words(item.emoji);
      if (!found) missing++;
      return [item.emoji, ...(found ?? [item.label, ''])];
    }),
}));

writeFileSync('public/emoji.json', JSON.stringify({ groups }));
console.log(groups.map((group) => `${group.name}: ${group.emoji.length}`).join('\n'));
console.log(`without a Hebrew name: ${missing}`);
