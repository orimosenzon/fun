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

// ---------- טקי ----------
// כל הקאטה בקיבה־דאצ'י על קו אחד, הגוף פונה לחזית. half(a) בונה חצי קאטה: a הצד שאליו הולכים קודם.
function tekki1Half(a, last) {
  const b = a === 'R' ? 'L' : 'R';           // a: היד והרגל של הצד הראשון
  const top = (side) => (side === 'R' ? { L: 'kRtop', R: 'kRbot' } : { L: 'kLbot', R: 'kLtop' });
  return [
    // אל הצד a: האישו־אוקה, אמפי
    ...(last ? [] : [S('kosaY', b, 'F', b, { via: true, t: 'n', look: a, L: 'hold', R: 'hold' })]),
    last
      ? KB('L', 'F', 0, { [a]: 'haishu@' + a, [b]: 'hiki', look: a, t: 's' })
      : KB('L', 'F', a, { [a]: 'haishu@' + a, [b]: 'hiki', look: a, knee: true, stomp: true, t: 'm' }),
    KB('L', 'F', 0, { [b]: 'empi@' + a, [a]: 'palmT@' + a, look: a, tw: a === 'R' ? 35 : -35, he: 'מאה־אמפי', ja: 'mae-empi', note: 'המרפק מכה אל כף היד' }),
    KB('L', 'F', 0, { ...top(a), look: b, t: 'm' }),
    KB('L', 'F', 0, { [b]: 'gedan@' + b, [a]: 'hiki', look: b }),
    KB('L', 'F', 0, { [a]: 'kagi', [b]: 'hiki', look: b }),
    // צעד שיכול אל הצד b, ואודה־אוקה לחזית
    S('kosaY', a, 'F', a, { via: true, t: 'n', look: 'F', L: 'hold', R: 'hold' }),
    KB('L', 'F', b, { [a]: 'uchi@F', [b]: 'hiki', look: 'F', knee: true, stomp: true }),
    KB('L', 'F', 0, { [b]: 'nagashiJ', [a]: 'gedan@F', look: 'F', t: 'f', he: 'נגשי־אוקה ג׳ודאן + גדאן־אוקה', ja: 'jodan nagashi-uke, gedan-uke' }),
    KB('L', 'F', 0, { [b]: 'uraUchi@F', [a]: 'underElbow', look: 'F', t: 'f', hold: 0.5 }),
    { ...KK('nami', b, 'F', 'F', { look: b, L: 'hold', R: 'hold' }), st: 'kiba', keep: true },
    KB('L', 'F', 0, { [b]: 'uchi@' + b, [a]: 'underElbow', look: b, t: 'f', hold: 0.4, he: 'סוקומן־אוקה', ja: 'sokumen-uke', note: 'חסימה באמה אל הצד' }),
    { ...KK('nami', a, 'F', 'F', { look: a, L: 'hold', R: 'hold' }), st: 'kiba', keep: true },
    KB('L', 'F', 0, { [b]: 'soto@' + a, [a]: 'underElbow', look: a, t: 'f', hold: 0.4, he: 'סוקומן־אוקה', ja: 'sokumen-uke', note: 'חסימה באמה אל הצד השני' }),
    KB('L', 'F', 0, { ...top(a), look: b, t: 'm' }),
    KB('L', 'F', 0, { [b]: 'tsuki_c@' + b, [a]: 'kagi', look: b, kiai: true, he: 'מורוטה־צוקי (צוקי + קאגי־צוקי)', ja: 'morote-zuki' }),
  ];
}
kata('tekki1', 'tekki', 'טקי שודאן', 'Tekki Shodan', 'רוכב הברזל, ראשונה', {
  start: 'heisoku', startArms: 'cover', end: { mv: 'R' },
  moves: [...tekki1Half('R', false), ...tekki1Half('L', true)],
});

// ---------- קאטות מתקדמות ----------
kata('bassai', 'adv', 'בסאי', 'Bassai', 'לפרוץ את המבצר', {
  start: 'heisoku', startArms: 'cover',
  moves: [
    S('kosa', 'R', 'F', 0, { go: ['F', 0.8], R: 'uchi', L: 'soete', he: 'אודה־אוקה, כף שמאל תומכת', ja: 'ude-uke' }),
    Z('L', 'B', 'L', { L: 'uchi', turn: '-' }),
    Z('L', 'B', 0, { R: 'uchi', hip: -25, t: 'f', hold: 0.4 }),
    Z('R', 'F', 'R', { L: 'soto', hip: -25 }),
    Z('R', 'F', 0, { R: 'uchi', t: 'f', hold: 0.4 }),
    Z('R', 'R', 'R', { R: 'sukuiG', L: 'hiki', wide: 0.6, lean: 28, via: true, t: 'n' }),
    Z('R', 'R', 0, { R: 'soto', note: 'קודם היד גורפת מלמטה (סוקוי), ואז חוסמת' }),
    Z('R', 'R', 0, { L: 'uchi', hip: -25, t: 'f', hold: 0.4 }),
    S('hachiji', 'L', 'F', 'L', { L: 'kRtop', R: 'kRbot', t: 'm' }),
    S('hachiji', 'L', 'F', 0, { L: 'tateShuto', R: 'hiki', t: 's' }),
    S('hachiji', 'L', 'F', 0, { R: 'tsuki_c', he: 'צוקי צ׳ודאן', ja: 'chudan-zuki' }),
    S('hachiji', 'L', 'F', 0, { R: 'uchi', hip: -35, t: 'f', hold: 0.3 }),
    S('hachiji', 'L', 'F', 0, { L: 'tsuki_c', he: 'צוקי צ׳ודאן', ja: 'chudan-zuki' }),
    S('hachiji', 'L', 'F', 0, { L: 'uchi', hip: 35, t: 'f', hold: 0.3 }),
    KO('R', 'F', 'R', { R: 'shuto' }),
    KO('L', 'F', 'L', { L: 'shuto' }),
    KO('R', 'F', 'R', { R: 'shuto' }),
    KO('L', 'F', 'R', { L: 'shuto', note: 'צעד לאחור' }),
    Z('L', 'F', 0, { R: 'grabHi', L: 'grabLo', t: 's', he: 'צוקאמי־אוקה', ja: 'tsukami-uke', note: 'שתי הידיים תופסות' }),
    KK('fumi', 'R', 'F', 'F', { L: 'pullDown', kiai: true, hold: 0.3, note: 'הידיים נמשכות אל המותן' }),
    KO('L', 'B', 'R', { L: 'shuto', turn: '-' }),
    KO('R', 'B', 'R', { R: 'shuto' }),
    S('heisoku', 'L', 'B', 'R', { L: 'ageBoth', t: 's' }),
    Z('R', 'B', 'R', { L: 'hasami', knee: true }),
    Z('R', 'B', 0, { R: 'tsuki_c', by: 0.3, t: 'f', hold: 0.4, note: 'יורי־אשי' }),
    Z('L', 'F', 'L', { R: 'gedanShuto', L: 'shoulder', turn: '-' }),
    S('heisoku', 'L', 'R', 'L', { L: 'gedan@F', R: 'manjiHi', look: 'F', t: 'm', he: 'מנג׳י־גמאה', ja: 'manji-gamae' }),
    KB('R', 'L', 'R', { R: 'gedan@F', L: 'hiki', look: 'F', knee: true, stomp: true, turn: '-', he: 'פומיקומי + גדאן־בראי', ja: 'fumikomi, gedan-barai' }),
    KB('R', 'L', 0, { L: 'haishu@B', R: 'hiki', look: 'B', t: 's' }),
    KK('mika', 'R', 'B', 'B', { L: 'palmSide@B', look: 'B', note: 'הבעיטה פוגעת בכף יד שמאל' }),
    KB('L', 'R', 'R', { R: 'empi@B', L: 'palmT@B', look: 'B', he: 'מאה־אמפי', ja: 'mae-empi' }),
    KB('L', 'R', 0, { R: 'gedan@B', L: 'chestFist', look: 'B', t: 'f', hold: 0.25 }),
    KB('L', 'R', 0, { L: 'gedan@B', R: 'chestFist', look: 'B', t: 'f', hold: 0.25 }),
    KB('L', 'R', 0, { R: 'gedan@B', L: 'chestFist', look: 'B', t: 'f', hold: 0.5 }),
    KB('L', 'R', 0, { L: 'kLbot', R: 'kLtop', look: 'B', via: true, t: 'n' }),
    Z('R', 'B', 'R', { L: 'yamaTop', R: 'yamaBot', wide: 0.85 }),
    S('heisoku', 'L', 'B', 'R', { L: 'kRtop', R: 'kRbot', t: 'm' }),
    Z('L', 'B', 'L', { R: 'yamaTop', L: 'yamaBot', wide: 0.85, knee: true }),
    S('heisoku', 'L', 'B', 'L', { L: 'kLbot', R: 'kLtop', t: 'm' }),
    Z('R', 'B', 'R', { L: 'yamaTop', R: 'yamaBot', wide: 0.85, knee: true }),
    Z('L', 'L', 'L', { R: 'sukuiG', L: 'hiki', turn: '-', lean: 18, he: 'סוקוי־אוקה גדאן', ja: 'gedan sukui-uke' }),
    Z('R', 'R', 0, { L: 'sukuiG', R: 'hiki', lean: 18, he: 'סוקוי־אוקה גדאן', ja: 'gedan sukui-uke' }),
    KO('R', 'FR', 'R', { R: 'shuto', t: 'm' }),
    KO('L', 'FL', 'L', { L: 'shuto', t: 's', kiai: true }),
  ],
});

// קטעים שחוזרים בקוואנקו
const kwShutoGeri = (d, back) => [   // שוטו־אוצ'י ג'ודאן, מאה־גרי, סיבוב למנג'י־אוקה, נוקיטה גדאן, גדאן־גמאה
  Z('L', d, 0, { R: 'shutouchi', L: 'hitai', hip: -30, he: 'שוטו־אוצ׳י ג׳ודאן', ja: 'jodan shuto-uchi', note: 'יד שמאל פתוחה מעל המצח' }),
  KK('mae', 'R', d, d, { lv: 'j' }),
  KO('L', back, 'R', { L: 'gedan', R: 'manjiHi', turn: '-', he: 'מנג׳י־אוקה', ja: 'manji-uke' }),
  Z('L', back, 0, { R: 'gedanShuto', L: 'shoulder', lean: 14, he: 'נוקיטה גדאן', ja: 'gedan nukite' }),
  S('teiji', 'L', back, 0, { L: 'gedan', R: 'hiki', t: 'm', he: 'גדאן־גמאה', ja: 'gedan-gamae' }),
];
const kwKeage = () => [     // קושי־גמאה, יוקו־גרי קאגה ואורקן לאחור, ואז שוטו־אוקה לחזית
  S('heiko', 'L', 'R', 'L', { L: 'kLbot', R: 'kLtop', look: 'B', via: true, t: 'n' }),
  KK('yoko', 'R', 'B', 'R', { R: 'uraken@B', look: 'B' }),
  S('heisoku', 'L', 'F', 'R', { via: true, t: 'x' }),
  KO('L', 'F', 'L', { L: 'shuto' }),
];
kata('kwanku', 'adv', 'קוואנקו', 'Kwanku', 'להביט אל השמיים', {
  startArms: 'side', drift: 1.8,
  moves: [
    S('hachiji', 'L', 'F', 0, { L: 'openLow', t: 'm', he: 'הידיים נפגשות למשולש', ja: '' }),
    S('hachiji', 'L', 'F', 0, { L: 'openUp', t: 's', headP: -18, he: 'הרמת הידיים, מבט אל השמיים', ja: '' }),
    S('hachiji', 'L', 'F', 0, { L: 'openSide', t: 'm', via: true }),
    S('hachiji', 'L', 'F', 0, { L: 'meetLow', t: 'm', he: 'הידיים נפרדות במעגל ונפגשות למטה', ja: '' }),
    KO('L', 'L', 'L', { L: 'haiwanO', R: 'shuto2' }),
    KO('R', 'R', 0, { R: 'haiwanO', L: 'shuto2' }),
    S('hachiji', 'L', 'F', 'L', { L: 'tateShuto', R: 'hiki', t: 's' }),
    S('hachiji', 'L', 'F', 0, { R: 'tsuki_c', he: 'צוקי צ׳ודאן', ja: 'chudan-zuki' }),
    S('hachiji', 'L', 'F', 0, { R: 'uchi', hip: -35, t: 'f', hold: 0.3 }),
    S('hachiji', 'L', 'F', 0, { L: 'tsuki_c', he: 'צוקי צ׳ודאן', ja: 'chudan-zuki' }),
    S('hachiji', 'L', 'F', 0, { L: 'uchi', hip: 35, t: 'f', hold: 0.3 }),
    ...kwKeage(),
    KO('R', 'F', 'R', { R: 'shuto' }),
    KO('L', 'F', 'L', { L: 'shuto' }),
    Z('R', 'F', 'R', { R: 'nukite', kiai: true }),
    ...kwShutoGeri('B', 'F'),
    ...kwShutoGeri('F', 'B'),
    S('heiko', 'R', 'B', 'L', { L: 'kRtop', R: 'kRbot', look: 'R', via: true, t: 'n' }),
    KK('yoko', 'L', 'R', 'B', { L: 'uraken@R', look: 'R' }),
    Z('L', 'R', 'L', { R: 'empi', L: 'palmT', he: 'מאה־אמפי', ja: 'mae-empi' }),
    S('heiko', 'L', 'B', 'R', { L: 'kLbot', R: 'kLtop', look: 'L', via: true, t: 'n' }),
    KK('yoko', 'R', 'L', 'B', { R: 'uraken@L', look: 'L' }),
    Z('R', 'L', 'R', { L: 'empi', R: 'palmT', he: 'מאה־אמפי', ja: 'mae-empi' }),
    KO('L', 'R', 'L', { L: 'shuto', turn: '-' }),
    KO('R', 'BR', 'R', { R: 'shuto' }),
    KO('R', 'L', 'R', { R: 'shuto' }),
    KO('L', 'BL', 'L', { L: 'shuto' }),
    Z('L', 'B', 'L', { R: 'shutouchi', L: 'hitai', hip: -30, he: 'שוטו־אוצ׳י ג׳ודאן', ja: 'jodan shuto-uchi' }),
    KK('mae', 'R', 'B', 'B', { lv: 'j' }),
    S('kosa', 'R', 'B', 0, { R: 'uraken', L: 'hiki', go: ['B', 1.3] }),
    Z('R', 'B', 'L', { R: 'uchi', note: 'רגל שמאל נסוגה' }),
    Z('R', 'B', 0, { L: 'tsuki_c', t: 'f' }),
    Z('R', 'B', 0, { R: 'tsuki_c', t: 'f', hold: 0.4 }),
    { ...KK('hiza', 'R', 'F', 'F', { R: 'uraUchi', L: 'soete', turn: '-', he: 'הייזה־גרי + אורה־צוקי', ja: 'hiza-geri, ura-zuki' }), hold: 0.25 },
    S('fuse', 'R', 'F', 'R', { L: 'floor', R: 'floor', look: 'F', he: 'ירידה אל הרצפה', ja: 'fuse', note: 'שתי כפות הידיים על הרצפה, המבט קדימה' }),
    KO('L', 'B', 0, { L: 'shutoG', dh: -0.07, turn: '-' }),
    KO('R', 'B', 'R', { R: 'shuto' }),
    Z('L', 'L', 'L', { L: 'uchi', turn: '-' }),
    Z('L', 'L', 0, { R: 'tsuki_c', t: 'f', hold: 0.4 }),
    Z('R', 'R', 'R', { R: 'uchi' }),
    Z('R', 'R', 0, { L: 'tsuki_c', t: 'f' }),
    Z('R', 'R', 0, { R: 'tsuki_c', t: 'f', hold: 0.4 }),
    ...kwKeage(),
    Z('R', 'F', 'R', { R: 'nukite' }),
    KB('L', 'R', 'L', { L: 'uraken@F', R: 'hiki', look: 'F', turn: '-', note: 'היד מסתובבת, סיבוב שמאלה על רגל ימין' }),
    KB('L', 'R', 0, { L: 'tettsui@F', R: 'hiki', look: 'F', go: ['F', 0.3], note: 'יורי־אשי' }),
    KB('L', 'R', 0, { R: 'empi@F', L: 'palmT@F', look: 'F', tw: -35, he: 'מאה־אמפי', ja: 'mae-empi' }),
    KB('L', 'R', 0, { L: 'kLbot', R: 'kLtop', look: 'B', t: 'm' }),
    KB('L', 'R', 0, { R: 'gedan@B', L: 'hiki', look: 'B' }),
    KB('R', 'L', 'L', { L: 'gedan@B', R: 'sideHi', look: 'B', turn: '-', note: 'סיבוב, יד ימין מורמת' }),
    KB('R', 'L', 0, { R: 'otoshi', L: 'hiki', dh: -0.10, lean: 18, headP: 20, stomp: true }),
    S('hachiji', 'L', 'L', 'R', { L: 'jujiJ', t: 'm' }),
    S('hachiji', 'L', 'B', 0, { L: 'crossChest', t: 's', turn: '+', he: 'הידיים יורדות אל החזה, סיבוב לאחור', ja: '' }),
    KK('mae', 'L', 'B', 'B', { via: true, t: 'x', hold: 0 }),
    KK('mae', 'R', 'B', 'B', { lv: 'j', jumpH: 0.30, kiai: true, he: 'נידאן־גרי', ja: 'nidan-geri', note: 'בעיטה כפולה בקפיצה' }),
    Z('R', 'B', 0, { R: 'uraken', L: 'hiki', go: ['B', 1.5] }),
    Z('L', 'F', 'L', { R: 'sukuiG', L: 'hiki', turn: '-', lean: 22, he: 'סוקוי־אוקה', ja: 'sukui-uke', note: 'היד גורפת מלמטה' }),
    S('hachiji', 'L', 'F', 'L', { L: 'openUp', t: 's', via: true }),
  ],
});

const jiKakiwake = (lead, d) => {      // קקיוואקה, מאה־גרי, שלושה אגרופים
  const o = lead === 'L' ? 'R' : 'L';
  return [
    Z(lead, d, lead, { [lead]: 'kakiwake', t: 's' }),
    KK('mae', o, d, d),
    Z(o, d, o, { [o]: 'tsuki_c', t: 'f' }),
    Z(o, d, 0, { [lead]: 'tsuki_c', t: 'f' }),
    Z(o, d, 0, { [o]: 'tsuki_c', t: 'f', hold: 0.4, he: 'צוקי צ׳ודאן', ja: 'chudan-zuki' }),
  ];
};
const jiManji = (lead, d) => {         // מנג'י־אוקה בקוקוטסו, ואז קאגי־צוקי בקיבה ביורי־אשי
  const o = lead === 'L' ? 'R' : 'L';
  return [
    KO(lead, d, lead, { [lead]: 'gedan', [o]: 'manjiHi', ...(lead === 'L' ? { turn: '-' } : {}), he: 'מנג׳י־אוקה', ja: 'manji-uke' }),
    KB('L', 'F', 0, { [o]: 'kagi', [lead]: 'hiki', look: d, go: [d, 0.25], note: 'יורי־אשי' }),
  ];
};
const jiKibaLine = (arm, extra = {}) => [   // שלושה צעדים בקיבה־דאצ'י לאחור, הגוף בצד
  KB('R', 'R', 'R', { R: arm + '@B', L: 'hiki', look: 'B', knee: true, ...extra }),
  KB('L', 'L', 'L', { L: arm + '@B', R: 'hiki', look: 'B', knee: true, turn: '+', ...extra }),
  KB('R', 'R', 'R', { R: arm + '@B', L: 'hiki', look: 'B', knee: true, turn: '-', ...extra }),
];
kata('jion', 'adv', 'ג׳יון', 'Jion', 'על שם מקדש ג׳יון', {
  start: 'heisoku', startArms: 'jiai', drift: 1.6,
  moves: [
    Z('R', 'F', 'L', { R: 'uchi', L: 'gedan', hip: 0, he: 'אודה־אוקה + גדאן־בראי', ja: 'ude-uke, gedan-barai', note: 'רגל שמאל נסוגה' }),
    ...jiKakiwake('L', 'FL'),
    ...jiKakiwake('R', 'FR'),
    Z('L', 'F', 'L', { L: 'age' }),
    Z('L', 'F', 0, { R: 'tsuki_c', t: 'f', hold: 0.35 }),
    Z('R', 'F', 'R', { R: 'age' }),
    Z('R', 'F', 0, { L: 'tsuki_c', t: 'f', hold: 0.35 }),
    Z('L', 'F', 'L', { L: 'age' }),
    Z('R', 'F', 'R', { R: 'tsuki_c', kiai: true }),
    ...jiManji('L', 'L'),
    ...jiManji('R', 'R'),
    Z('L', 'B', 'L', { L: 'gedan' }),
    ...jiKibaLine('teisho'),
    KO('L', 'L', 'L', { L: 'gedan', R: 'manjiHi', turn: '-', he: 'מנג׳י־אוקה', ja: 'manji-uke' }),
    S('heisoku', 'L', 'F', 'R', { L: 'haiwan@L', R: 'hitaiF@L', look: 'L', he: 'מורוטה־אוקה ג׳ודאן', ja: 'jodan morote-uke' }),
    KO('R', 'R', 'R', { R: 'gedan', L: 'manjiHi', he: 'מנג׳י־אוקה', ja: 'manji-uke' }),
    S('heisoku', 'L', 'F', 'L', { R: 'haiwan@R', L: 'hitaiF@R', look: 'R', he: 'מורוטה־אוקה ג׳ודאן', ja: 'jodan morote-uke' }),
    S('heisoku', 'L', 'F', 0, { L: 'gedanBoth', t: 's', look: 'F', he: 'ריוואן־גמאה', ja: 'ryowan-gamae', note: 'שתי הידיים נפתחות לאט לצדדים למטה' }),
    S('kosa', 'R', 'F', 0, { L: 'jujiG', go: ['F', 0.8], knee: true, stomp: true }),
    Z('R', 'F', 'L', { L: 'gedanBoth', note: 'רגל שמאל נסוגה' }),
    Z('L', 'F', 'L', { L: 'kakiwake', he: 'ריוואן אודה־אוקה', ja: 'ryowan ude-uke' }),
    Z('R', 'F', 'R', { L: 'jujiJF' }),
    Z('R', 'F', 0, { R: 'uraUchi', L: 'hiki', t: 'f' }),
    Z('R', 'F', 0, { L: 'tsuki_c', R: 'nagashiJ', t: 'f', he: 'צוקי צ׳ודאן + נגשי־אוקה ג׳ודאן', ja: 'chudan-zuki, jodan nagashi-uke' }),
    Z('R', 'F', 0, { R: 'uraUchi', L: 'underElbow', t: 'f', hold: 0.5 }),
    Z('L', 'R', 'L', { L: 'uchi', turn: '-' }),
    Z('R', 'R', 'R', { R: 'tsuki_c' }),
    Z('R', 'L', 'R', { R: 'uchi' }),
    Z('L', 'L', 'L', { L: 'tsuki_c' }),
    Z('L', 'B', 'L', { L: 'gedan' }),
    ...jiKibaLine('otoshiU', { stomp: true }),
    KB('L', 'F', 'L', { L: 'sideZuki@L', R: 'chestFist', look: 'L', turn: '-', go: ['L', 0.2], he: 'צוקי לצד, היד השנייה נמשכת אל החזה', ja: '', note: 'יורי־אשי שמאלה' }),
    KB('L', 'F', 0, { R: 'sideZuki@R', L: 'chestFist', look: 'R', go: ['R', 0.5], kiai: true, he: 'צוקי לצד, היד השנייה נמשכת אל החזה', ja: '', note: 'יורי־אשי ימינה' }),
  ],
  end: { mv: 'R' },
});

const emAge = (d, back) => [      // אגה־צוקי, ברך וזינוק לקוסה־דאצ'י עם צוקי, גדאן־בראי לאחור, וסיבוב לגדאן־בראי
  Z('L', d, 0, { R: 'ageZukiJ' }),
  KK('hiza', 'R', d, d, { via: true, t: 'f', hold: 0, R: 'grabHi', L: 'hiki' }),
  S('kosa', 'R', d, 0, { go: [d, 0.75], L: 'tsuki_c', R: 'shoulder', he: 'צוקי צ׳ודאן בזינוק', ja: 'chudan-zuki', note: 'אגרוף ימין נמשך אל הכתף השמאלית' }),
  Z('R', d, 'L', { L: 'gedan@' + back, R: 'hiki', look: back, he: 'גדאן־בראי לאחור', ja: 'ushiro gedan-barai', note: 'רגל שמאל נסוגה' }),
  Z('L', back, 0, { L: 'gedan' }),
];
kata('empi', 'adv', 'אמפי', 'Empi', 'מעוף הסנונית', {
  startArms: 'side', drift: 4.2,
  moves: [
    S('hachiji', 'L', 'F', 0, { L: 'kLbot', R: 'kLtop', t: 'm', via: true }),
    S('hiza', 'L', 'L', 'L', { R: 'gedan@F', L: 'chestFist', hip: 75, look: 'F', he: 'גדאן־בראי בכריעה', ja: 'gedan-barai', note: 'ברך ימין יורדת לרצפה, אגרוף שמאל אל החזה' }),
    S('hachiji', 'L', 'F', 'L', { L: 'kLbot', R: 'kLtop', t: 'm' }),
    Z('R', 'R', 'R', { R: 'gedan' }),
    KB('L', 'F', 'R', { L: 'kagi', R: 'hiki', look: 'F' }),
    Z('L', 'F', 'L', { L: 'gedan' }),
    ...emAge('F', 'B'),
    ...emAge('B', 'F'),
    KB('L', 'F', 'R', { L: 'haiwanO@F', R: 'hiki', look: 'F', t: 's' }),
    KB('L', 'F', 0, { L: 'tateShuto@F', R: 'hiki', look: 'F', t: 'm', stomp: true, note: 'רקיעה ברגל ימין' }),
    KB('L', 'F', 0, { R: 'tsuki_c@F', L: 'hiki', t: 'f', he: 'צוקי צ׳ודאן', ja: 'chudan-zuki' }),
    KB('L', 'F', 0, { L: 'tsuki_c@F', R: 'hiki', t: 'f', hold: 0.4, he: 'צוקי צ׳ודאן', ja: 'chudan-zuki' }),
    Z('L', 'L', 'L', { L: 'gedan' }),
    Z('L', 'L', 0, { R: 'ageZukiJ' }),
    KO('R', 'L', 'R', { R: 'shuto' }),
    KO('L', 'L', 0, { L: 'shuto', t: 'f', hold: 0.3, note: 'החלפת רגליים במקום' }),
    KO('L', 'L', 0, { R: 'tsuki_c', t: 'f', hold: 0.3 }),
    KO('R', 'L', 'R', { R: 'shuto' }),
    Z('L', 'R', 'L', { L: 'gedan' }),
    ...emAge('R', 'L'),
    Z('L', 'F', 'L', { R: 'teisho', L: 'osae', t: 's', he: 'טיישו', ja: 'teisho', note: 'כף היד עולה לאט, השנייה לוחצת מטה' }),
    Z('R', 'F', 'R', { L: 'teisho', R: 'osae', t: 's', he: 'טיישו', ja: 'teisho' }),
    Z('L', 'F', 'L', { R: 'teisho', L: 'osae', t: 's', he: 'טיישו', ja: 'teisho' }),
    KO('R', 'F', 'R', { R: 'shuto' }),
    Z('R', 'F', 0, { R: 'kokoHi', L: 'kokoLo', t: 'm', he: 'אחיזה והרמה', ja: 'koko-gamae', note: 'יד ימין למעלה, שמאל למטה' }),
    KO('R', 'F', 0, { R: 'shuto', air: true, airH: 0.3, spin: -360, go: ['F', 0.5], kiai: true, note: 'קפיצה עם סיבוב מלא שמאלה' }),
    KO('L', 'F', 'R', { L: 'shuto', note: 'צעד לאחור' }),
  ],
});
