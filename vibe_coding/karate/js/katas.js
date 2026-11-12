// katas.js: הקאטות של שוטוקאן (לפי קראטה־דו קיוהאן, כפי שנלמד אצל אושימה).
// כל תנועה: st עמידה, lead הרגל המובילה, dir הכיוון (F חזית, B אחור, L שמאל, R ימין ואלכסונים),
// mv הרגל שזזה (השנייה נשארת נטועה; בלי mv שתי הרגליים מסתובבות במקום), L/R מה עושה כל יד,
// t מקצב (n רגיל, s איטי, f מהיר ומחובר, x מהיר מאוד, m בינוני), turn: '-' סיבוב שמאלה גם כשהוא הדרך הארוכה, '+' ימינה.
// kiai: קיאי. via: תנועת מעבר שהיא חלק מהתנועה הבאה. he/ja: שם מפורש במקום השם האוטומטי. note: הערה.

const S = (st, lead, dir, mv, o = {}) => ({ st, lead, dir, mv: mv || undefined, ...o });
const Z = (lead, dir, mv, o) => S('zen', lead, dir, mv, o);
const KO = (lead, dir, mv, o) => S('kok', lead, dir, mv, o);
const KB = (lead, dir, mv, o) => S('kiba', lead, dir, mv, o);
// בעיטה: סוג, רגל, כיוון הבעיטה, הכיוון שהגוף פונה אליו
const KK = (kick, leg, to, face, o = {}) => ({ st: 'shizen', dir: face, kick, leg, to, t: 'f', hold: 0.02, ...o });
const MOROTE = { he: 'מורוטה־אוקה', ja: 'morote-uke' };

export const KATAS = [];
const kata = (id, group, he, ja, mean, o) => KATAS.push({ id, group, he, ja, mean, ...o });

// ---------- טאיקיוקו ----------
// שלוש הקאטות באותו תוואי: לצדדים חסימה ואוי־צוקי, ובקו האמצע שלושה אוי־צוקי.
function taikyoku(side, sideZ, mid) {
  const blk = (lead, dir, o) => (sideZ === 'kok' ? KO(lead, dir, lead, { [lead]: side, ...o }) : Z(lead, dir, lead, { [lead]: side, ...o }));
  const pair = (l1, d, o) => { const l2 = l1 === 'L' ? 'R' : 'L'; return [blk(l1, d, o), Z(l2, d, l2, { [l2]: 'tsuki_c' })]; };
  const line = (d, o) => [Z('L', d, 'L', { L: 'gedan', ...o }), Z('R', d, 'R', { R: mid }), Z('L', d, 'L', { L: mid }), Z('R', d, 'R', { R: mid, kiai: true })];
  return [...pair('L', 'L'), ...pair('R', 'R'), ...line('F'), ...pair('L', 'R', { turn: '-' }), ...pair('R', 'L'), ...line('B'), ...pair('L', 'L', { turn: '-' }), ...pair('R', 'R')];
}
kata('taikyoku1', 'basic', 'טאיקיוקו שודאן', 'Taikyoku Shodan', 'הסיבה הראשונה', { moves: taikyoku('gedan', 'zen', 'tsuki_c') });
kata('taikyoku2', 'basic', 'טאיקיוקו נידאן', 'Taikyoku Nidan', 'כמו שודאן, האוי־צוקי לג׳ודאן', { moves: taikyoku('gedan', 'zen', 'tsuki_j').map((m) => (m.L === 'tsuki_c' ? { ...m, L: 'tsuki_j' } : m.R === 'tsuki_c' ? { ...m, R: 'tsuki_j' } : m)) });
kata('taikyoku3', 'basic', 'טאיקיוקו סנדאן', 'Taikyoku Sandan', 'לצדדים אודה־אוקה בקוקוטסו', { moves: taikyoku('uchi', 'kok', 'tsuki_j') });

// ---------- הייאן ----------
kata('heian1', 'heian', 'הייאן שודאן', 'Heian Shodan', 'שלווה, ראשונה', {
  moves: [
    Z('L', 'L', 'L', { L: 'gedan' }),
    Z('R', 'L', 'R', { R: 'tsuki_c' }),
    Z('R', 'R', 'R', { R: 'gedan' }),
    Z('R', 'R', 'R', { R: 'tettsui', wide: 0.72, hip: 20, note: 'הרגל הקדמית נמשכת לאחור, היד משתחררת במעגל גדול ומכה מלמעלה' }),
    Z('L', 'R', 'L', { L: 'tsuki_c' }),
    Z('L', 'F', 'L', { L: 'gedan' }),
    Z('R', 'F', 'R', { R: 'age' }),
    Z('L', 'F', 'L', { L: 'age' }),
    Z('R', 'F', 'R', { R: 'age', kiai: true }),
    Z('L', 'R', 'L', { L: 'gedan', turn: '-' }),
    Z('R', 'R', 'R', { R: 'tsuki_c' }),
    Z('R', 'L', 'R', { R: 'gedan' }),
    Z('L', 'L', 'L', { L: 'tsuki_c' }),
    Z('L', 'B', 'L', { L: 'gedan' }),
    Z('R', 'B', 'R', { R: 'tsuki_c' }),
    Z('L', 'B', 'L', { L: 'tsuki_c' }),
    Z('R', 'B', 'R', { R: 'tsuki_c', kiai: true }),
    KO('L', 'L', 'L', { L: 'shuto', turn: '-' }),
    KO('R', 'FL', 'R', { R: 'shuto' }),
    KO('R', 'R', 'R', { R: 'shuto' }),
    KO('L', 'FR', 'L', { L: 'shuto' }),
  ],
});

kata('heian2', 'heian', 'הייאן נידאן', 'Heian Nidan', 'שלווה, שנייה', {
  moves: [
    KO('L', 'L', 'L', { L: 'haiwan', R: 'hitaiF' }),
    KO('L', 'L', 0, { R: 'uraIn', L: 'shoulder', t: 'f', he: 'משיכה ומכה', ja: '', note: 'יד שמאל נמשכת אל הכתף, אגרוף ימין מכה פנימה' }),
    KO('L', 'L', 0, { L: 'tsuki_c', he: 'צוקי צ׳ודאן', ja: 'chudan-zuki' }),
    KO('R', 'R', 0, { R: 'haiwan', L: 'hitaiF' }),
    KO('R', 'R', 0, { L: 'uraIn', R: 'shoulder', t: 'f', he: 'משיכה ומכה', ja: '', note: 'יד ימין נמשכת אל הכתף, אגרוף שמאל מכה פנימה' }),
    KO('R', 'R', 0, { R: 'tsuki_c', he: 'צוקי צ׳ודאן', ja: 'chudan-zuki' }),
    S('heiko', 'L', 'R', 'L', { L: 'kLbot', R: 'kLtop', look: 'B', via: true, t: 'n' }),
    KK('yoko', 'R', 'B', 'R', { R: 'uraken@B', look: 'B' }),
    S('heisoku', 'L', 'F', 'R', { via: true, t: 'x' }),
    KO('L', 'F', 'L', { L: 'shuto' }),
    KO('R', 'F', 'R', { R: 'shuto' }),
    KO('L', 'F', 'L', { L: 'shuto' }),
    Z('R', 'F', 'R', { R: 'nukite', kiai: true }),
    KO('L', 'R', 'L', { L: 'shuto', turn: '-' }),
    KO('R', 'BR', 'R', { R: 'shuto' }),
    KO('R', 'L', 'R', { R: 'shuto' }),
    KO('L', 'BL', 'L', { L: 'shuto' }),
    Z('L', 'B', 'L', { R: 'uchi', hip: -25, note: 'גיאקו־האנמי: המותן מסובבת אל היד החוסמת' }),
    KK('mae', 'R', 'B', 'B'),
    Z('R', 'B', 'R', { L: 'tsuki_c' }),
    Z('R', 'B', 0, { L: 'uchi', hip: -25 }),
    KK('mae', 'L', 'B', 'B'),
    Z('L', 'B', 'L', { R: 'tsuki_c' }),
    Z('R', 'B', 'R', { R: 'uchi', L: 'morote2', ...MOROTE }),
    Z('L', 'L', 'L', { L: 'gedan', turn: '-' }),
    Z('R', 'FL', 'R', { R: 'age' }),
    Z('R', 'R', 'R', { R: 'gedan' }),
    Z('L', 'FR', 'L', { L: 'age', kiai: true }),
  ],
});

kata('heian3', 'heian', 'הייאן סנדאן', 'Heian Sandan', 'שלווה, שלישית', {
  moves: [
    KO('L', 'L', 'L', { L: 'uchi' }),
    S('heisoku', 'L', 'L', 'R', { R: 'uchi', L: 'gedan' }),
    S('heisoku', 'L', 'L', 0, { L: 'uchi', R: 'gedan', t: 'f' }),
    KO('R', 'R', 'R', { R: 'uchi' }),
    S('heisoku', 'R', 'R', 'L', { L: 'uchi', R: 'gedan' }),
    S('heisoku', 'R', 'R', 0, { R: 'uchi', L: 'gedan', t: 'f' }),
    KO('L', 'F', 'L', { L: 'uchi', R: 'morote2', ...MOROTE }),
    Z('R', 'F', 'R', { R: 'nukite' }),
    KB('L', 'R', 'L', { L: 'tettsui@F', R: 'hiki', look: 'F', turn: '-', note: 'היד מסתובבת מאחורי הגב, סיבוב שמאלה על רגל ימין' }),
    Z('R', 'F', 'R', { R: 'tsuki_c', kiai: true }),
    S('heisoku', 'L', 'B', 'L', { L: 'hipFist', R: 'hipFist', t: 's', turn: '-', he: 'אגרופים על המותניים', ja: 'ryoken koshi-gamae' }),
    KB('R', 'R', 'R', { L: 'hipFist', R: 'hipFist', tw: -28, look: 'B', stomp: true, turn: '-', he: 'פומיקומי + אמפי־אוקה', ja: 'fumikomi, empi-uke' }),
    KB('R', 'R', 0, { L: 'hipFist', R: 'uraken@B', look: 'B', t: 'f' }),
    KB('L', 'L', 'L', { L: 'hipFist', R: 'hipFist', tw: 28, look: 'B', stomp: true, turn: '+', he: 'פומיקומי + אמפי־אוקה', ja: 'fumikomi, empi-uke' }),
    KB('L', 'L', 0, { R: 'hipFist', L: 'uraken@B', look: 'B', t: 'f' }),
    KB('R', 'R', 'R', { L: 'hipFist', R: 'hipFist', tw: -28, look: 'B', stomp: true, turn: '-', he: 'פומיקומי + אמפי־אוקה', ja: 'fumikomi, empi-uke' }),
    KB('R', 'R', 0, { L: 'hipFist', R: 'uraken@B', look: 'B', t: 'f' }),
    KB('R', 'R', 0, { L: 'hipFist', R: 'tateShuto@B', look: 'B', t: 's' }),
    Z('L', 'B', 'L', { L: 'tsuki_c' }),
    KB('L', 'B', 'R', { via: true, t: 'n', L: 'hold', R: 'hold' }),
    KB('L', 'F', 'L', { R: 'overShoulder', L: 'ushiroEmpi', turn: '-', he: 'צוקי מעל הכתף + אושירו־אמפי', ja: 'ushiro-zuki, ushiro-empi' }),
    KB('L', 'F', 0, { L: 'overShoulder', R: 'ushiroEmpi', byz: 0.35, kiai: true, he: 'צוקי מעל הכתף + אושירו־אמפי', ja: 'ushiro-zuki, ushiro-empi', note: 'יורי־אשי ימינה' }),
  ],
});

kata('heian4', 'heian', 'הייאן יודאן', 'Heian Yodan', 'שלווה, רביעית', {
  moves: [
    KO('L', 'L', 'L', { L: 'haiwanO', R: 'hitai', t: 's' }),
    KO('R', 'R', 0, { R: 'haiwanO', L: 'hitai', t: 's' }),
    Z('L', 'F', 'L', { L: 'jujiG' }),
    KO('R', 'F', 'R', { R: 'uchi', L: 'morote2', ...MOROTE }),
    S('heiko', 'R', 'F', 'L', { L: 'kRtop', R: 'kRbot', look: 'L', via: true, t: 'n' }),
    KK('yoko', 'L', 'L', 'F', { L: 'uraken@L', look: 'L' }),
    Z('L', 'L', 'L', { R: 'empi', L: 'palmT', he: 'מאה־אמפי', ja: 'mae-empi', note: 'המרפק מכה אל כף היד' }),
    S('heiko', 'L', 'F', 'R', { L: 'kLbot', R: 'kLtop', look: 'R', via: true, t: 'n' }),
    KK('yoko', 'R', 'R', 'F', { R: 'uraken@R', look: 'R' }),
    Z('R', 'R', 'R', { L: 'empi', R: 'palmT', he: 'מאה־אמפי', ja: 'mae-empi', note: 'המרפק מכה אל כף היד' }),
    Z('L', 'F', 'L', { R: 'shutouchi', L: 'hitai', hip: -30, wide: 0.85, he: 'שוטו־אוצ׳י ג׳ודאן', ja: 'jodan shuto-uchi', note: 'יד שמאל פתוחה מעל המצח' }),
    KK('mae', 'R', 'F', 'F', { lv: 'j' }),
    S('kosa', 'R', 'F', 0, { R: 'uraken', L: 'hiki', go: ['F', 0.7], kiai: true }),
    KO('L', 'BL', 'L', { L: 'kakiwake', t: 's' }),
    KK('mae', 'R', 'BL', 'BL', { lv: 'j' }),
    Z('R', 'BL', 'R', { R: 'tsuki_c', t: 'f' }),
    Z('R', 'BL', 0, { L: 'tsuki_c', t: 'f', hold: 0.4 }),
    KO('R', 'BR', 'R', { R: 'kakiwake', t: 's' }),
    KK('mae', 'L', 'BR', 'BR', { lv: 'j' }),
    Z('L', 'BR', 'L', { L: 'tsuki_c', t: 'f' }),
    Z('L', 'BR', 0, { R: 'tsuki_c', t: 'f', hold: 0.4 }),
    KO('L', 'B', 'L', { L: 'uchi', R: 'morote2', ...MOROTE }),
    KO('R', 'B', 'R', { R: 'uchi', L: 'morote2', ...MOROTE }),
    KO('L', 'B', 'L', { L: 'uchi', R: 'morote2', ...MOROTE }),
    Z('L', 'B', 0, { L: 'grab', he: 'אחיזת הראש בשתי הידיים', ja: '' }),
    KK('hiza', 'R', 'B', 'B', { L: 'pullDown', kiai: true, hold: 0.3 }),
    KO('L', 'F', 'R', { L: 'shuto', turn: '-' }),
    KO('R', 'F', 'R', { R: 'shuto' }),
  ],
});

kata('heian5', 'heian', 'הייאן גודאן', 'Heian Godan', 'שלווה, חמישית', {
  moves: [
    KO('L', 'L', 'L', { L: 'uchi' }),
    KO('L', 'L', 0, { R: 'tsuki_c', t: 'f' }),
    S('heisoku', 'L', 'F', 'R', { L: 'kagi', R: 'hiki', t: 's', look: 'F', note: 'מיזו־נגארה: האמה מונחת לרוחב החזה, לאט' }),
    KO('R', 'R', 'R', { R: 'uchi' }),
    KO('R', 'R', 0, { L: 'tsuki_c', t: 'f' }),
    S('heisoku', 'R', 'F', 'L', { R: 'kagi', L: 'hiki', t: 's', look: 'F', note: 'מיזו־נגארה: האמה מונחת לרוחב החזה, לאט' }),
    KO('R', 'F', 'R', { R: 'uchi', L: 'morote2', ...MOROTE }),
    Z('L', 'F', 'L', { L: 'jujiG' }),
    Z('L', 'F', 0, { L: 'jujiJ', t: 'f', hold: 0.3 }),
    Z('L', 'F', 0, { L: 'osaeLow', t: 'f', note: 'הידיים מסתובבות ולוחצות מטה אל המותן הימנית' }),
    Z('L', 'F', 0, { L: 'tsuki_c', t: 'f', he: 'צוקי צ׳ודאן', ja: 'chudan-zuki' }),
    Z('R', 'F', 'R', { R: 'tsuki_c', kiai: true }),
    KB('R', 'R', 'R', { R: 'gedan@B', L: 'hiki', look: 'B', turn: '-', stomp: true, he: 'פומיקומי + גדאן־בראי', ja: 'fumikomi, gedan-barai' }),
    KB('R', 'R', 0, { L: 'haishu@F', R: 'hiki', look: 'F', t: 's' }),
    KK('mika', 'R', 'F', 'F', { L: 'palmSide@F', look: 'F', note: 'הבעיטה פוגעת בכף יד שמאל' }),
    KB('L', 'L', 'R', { R: 'empi@F', L: 'palmT@F', look: 'F', he: 'מאה־אמפי', ja: 'mae-empi' }),
    S('kosa', 'R', 'F', 'L', { R: 'uchi', L: 'morote2', ...MOROTE }),
    Z('L', 'B', 'L', { R: 'ageZuki', L: 'morote2', wide: 0.7, look: 'B', he: 'צוקי־אגה לאחור', ja: 'tsuki-age', note: 'מבט לאחור, האגרוף עולה' }),
    S('kosa', 'R', 'R', 0, { L: 'jujiG', go: ['B', 1.1], air: true, kiai: true, note: 'קפיצה' }),
    Z('R', 'B', 'R', { R: 'uchi', L: 'morote2', ...MOROTE }),
    Z('L', 'L', 'L', { R: 'gedanShuto@L', L: 'shoulder', hip: 80, turn: '-', look: 'L' }),
    KO('L', 'L', 0, { L: 'gedan@L', R: 'manjiHi', hip: 80, look: 'L', t: 'm', he: 'מנג׳י־אוקה', ja: 'manji-uke' }),
    S('heisoku', 'L', 'F', 'L', { L: 'hold', R: 'hold', t: 's', look: 'R', he: 'הייסוקו־דאצ׳י, הידיים נשארות', ja: '' }),
    Z('R', 'R', 'R', { L: 'gedanShuto@R', R: 'shoulder', hip: 80, look: 'R' }),
    KO('R', 'R', 0, { R: 'gedan@R', L: 'manjiHi', hip: 80, look: 'R', t: 'm', he: 'מנג׳י־אוקה', ja: 'manji-uke' }),
  ],
});
