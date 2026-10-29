/* fonts.js: הגופנים לכיתובים.
 *
 * שניים של המערכת (תמיד זמינים, גם בלי רשת) ושמונה מ-Google Fonts, שנטענים מה-<link>
 * ב-index.html. חלקם עם עברית, כי הממשק דו-לשוני. קנבס לא מחכה לגופן: אם מציירים
 * לפני שהוא נטען מקבלים גופן ברירת מחדל. לכן ensure() טוען את הגופן (ואת האותיות
 * שבטקסט) לפני ציור, וגם לפני כל ייצוא.
 */
window.C = window.C || {};

C.FONTS = [
  { id: 'arial', name: 'Arial', css: 'Arial, "Noto Sans Hebrew", sans-serif', system: true },
  { id: 'georgia', name: 'Georgia', css: 'Georgia, "Times New Roman", serif', system: true },
  { id: 'rubik', name: 'Rubik', css: '"Rubik", Arial, sans-serif' },
  { id: 'heebo', name: 'Heebo', css: '"Heebo", Arial, sans-serif' },
  { id: 'montserrat', name: 'Montserrat', css: '"Montserrat", Arial, sans-serif' },
  { id: 'bebas', name: 'Bebas Neue', css: '"Bebas Neue", Impact, sans-serif', noBold: true },
  { id: 'playfair', name: 'Playfair Display', css: '"Playfair Display", Georgia, serif' },
  { id: 'frank', name: 'Frank Ruhl Libre', css: '"Frank Ruhl Libre", Georgia, serif' },
  { id: 'caveat', name: 'Caveat', css: '"Caveat", cursive' },
  { id: 'amatic', name: 'Amatic SC', css: '"Amatic SC", cursive' },
];

C.font = (id) => C.FONTS.find((f) => f.id === id) || C.FONTS[0];
C.fontStack = (id) => C.font(id).css;

/** מחכה שהגופן של הכיתוב יהיה מוכן לציור. לא נכשל: בלי רשת פשוט מקבלים גופן חלופי. */
C.ensureFont = (cap) => {
  const f = C.font(cap.font);
  if (f.system || !document.fonts) return Promise.resolve();
  const spec = `${cap.bold && !f.noBold ? 700 : 400} 40px ${f.css}`;
  return Promise.race([
    document.fonts.load(spec, cap.text || 'Aa'),
    new Promise((r) => setTimeout(r, 4000)),
  ]).catch(() => {});
};
