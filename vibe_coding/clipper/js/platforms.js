/* platforms.js: הגדרות מוכנות לכל רשת חברתית.
 *
 * לכל פלטפורמה: יחס, אורך מקסימלי, ו"אזור בטוח": החלקים של המסך שהאפליקציה של
 * הרשת מכסה בכפתורים, בשם המשתמש ובכיתוב. הלוגו ממוקם בתוך האזור הבטוח, והתצוגה
 * המקדימה מסמנת את האזורים המכוסים כדי שלא ישימו שם משהו חשוב.
 *
 * המספרים נכונים למיטב הידיעה ל-10/2026. הרשתות משנות אותם מדי פעם (אינסטגרם
 * הגדילה את הרילס ל-3 דקות ב-2025, יוטיוב שורטס ל-3 דקות ב-2024), אז כדאי לבדוק
 * מחדש כשמשהו נדחה בהעלאה.
 *
 * safe = שברים מהגובה/רוחב: { top, bottom, left, right }
 * maxSec = null כשאין מגבלה מעשית
 */
window.C = window.C || {};

C.PLATFORMS = [
  {
    id: 'ig_reels', icon: '📸', format: '9:16', maxSec: 180,
    safe: { top: 0.12, bottom: 0.22, left: 0.06, right: 0.14 },
    name: { he: 'אינסטגרם רילס', en: 'Instagram Reels' },
    note: { he: 'עד 3 דקות. בגריד הפרופיל רואים רק את המרכז (3:4)', en: 'Up to 3 minutes. The profile grid shows only the center (3:4)' },
    needsAac: true,
  },
  {
    id: 'ig_story', icon: '⭕', format: '9:16', maxSec: 60,
    safe: { top: 0.14, bottom: 0.18, left: 0.05, right: 0.05 },
    name: { he: 'אינסטגרם סטורי', en: 'Instagram Story' },
    note: { he: 'סטורי ארוך מ-60 שניות נחתך לכמה סטוריז', en: 'Stories longer than 60s are split into several' },
    needsAac: true,
  },
  {
    id: 'ig_feed', icon: '🖼', format: '4:5', maxSec: null,
    safe: { top: 0.04, bottom: 0.04, left: 0.04, right: 0.04 },
    name: { he: 'אינסטגרם פיד', en: 'Instagram feed' },
    note: { he: '4:5 תופס הכי הרבה מקום בפיד', en: '4:5 takes the most room in the feed' },
    needsAac: true,
  },
  {
    id: 'tiktok', icon: '🎵', format: '9:16', maxSec: 600,
    safe: { top: 0.10, bottom: 0.20, left: 0.05, right: 0.15 },
    name: { he: 'טיקטוק', en: 'TikTok' },
    note: { he: 'עד 10 דקות בהעלאה', en: 'Up to 10 minutes when uploading' },
    needsAac: true,
  },
  {
    id: 'yt_shorts', icon: '▶️', format: '9:16', maxSec: 180,
    safe: { top: 0.08, bottom: 0.20, left: 0.05, right: 0.12 },
    name: { he: 'יוטיוב שורטס', en: 'YouTube Shorts' },
    note: { he: 'עד 3 דקות', en: 'Up to 3 minutes' },
  },
  {
    id: 'youtube', icon: '📺', format: '16:9', maxSec: 900,
    safe: { top: 0.04, bottom: 0.10, left: 0.04, right: 0.04 },
    name: { he: 'יוטיוב', en: 'YouTube' },
    note: { he: 'מעל 15 דקות צריך ערוץ מאומת', en: 'Over 15 minutes needs a verified channel' },
  },
  {
    id: 'facebook', icon: '👍', format: '4:5', maxSec: null,
    safe: { top: 0.04, bottom: 0.06, left: 0.04, right: 0.04 },
    name: { he: 'פייסבוק', en: 'Facebook' },
    note: { he: 'לרילס של פייסבוק עדיף 9:16', en: 'For Facebook Reels prefer 9:16' },
    needsAac: true,
  },
  {
    id: 'linkedin', icon: '💼', format: '1:1', maxSec: 600,
    safe: { top: 0.04, bottom: 0.06, left: 0.04, right: 0.04 },
    name: { he: 'לינקדאין', en: 'LinkedIn' },
    note: { he: 'עד 10 דקות', en: 'Up to 10 minutes' },
    needsAac: true,
  },
  {
    id: 'wa_status', icon: '💬', format: '9:16', maxSec: 60,
    safe: { top: 0.12, bottom: 0.14, left: 0.05, right: 0.05 },
    name: { he: 'סטטוס וואטסאפ', en: 'WhatsApp Status' },
    note: { he: 'עד דקה', en: 'Up to one minute' },
    needsAac: true,
  },
];

C.platform = (id) => C.PLATFORMS.find((p) => p.id === id) || null;

/** אזור בטוח לפורמט ידני: שוליים קטנים מכל צד */
C.DEFAULT_SAFE = { top: 0.04, bottom: 0.04, left: 0.04, right: 0.04 };
