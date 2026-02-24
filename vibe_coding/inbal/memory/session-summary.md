# סיכום שיחה - פרויקט inbal (קרמיקה)

## מצב נוכחי
- האפליקציה **עובדת** בכתובת: https://inbal-three.vercel.app
- לוגין: `admin@ceramics.co.il / admin123`
- תלמידים לבדיקה: `michal@example.com / student123` (ועוד 5)

## פרטי הפרויקט
- GitHub: orimosenzon/fun (branch: master)
- Vercel project: inbal (team: orimosenzons-projects)
- Supabase project: twboneztjbagmllufzwn (region: ap-south-1)

## env vars

### Vercel (production)
| Key | Status | ערך |
|-----|--------|-----|
| DATABASE_URL | ✅ תקין | Transaction pooler, port 6543, עם `?pgbouncer=true` |
| DIRECT_URL | ✅ תקין | חיבור ישיר port 5432 (למיגרציות) |
| NEXTAUTH_SECRET | ✅ תקין | |
| NEXTAUTH_URL | ✅ תקין | https://inbal-three.vercel.app |

### מקומי (.env.local)
- `DATABASE_URL` = חיבור ישיר (port 5432) — ה-pooler URL לא עובד מקומית
- `DIRECT_URL` = אותו חיבור ישיר
- **הערה**: ה-pooler URL של Supabase (eu-west-1) מחזיר `Tenant or user not found` מסביבה מקומית

## מה נעשה בסשן הזה

### Seed עם slotType
- ✅ `prisma/seed.ts` עודכן — כל enrollment ורישום כולל `preferredSlotType`
- ✅ 3 קבוצות, 6 תלמידים, 24 שיעורים (8 שבועות × 3) נוצרו בDB

### הגבלות ביטול/העברה (48 שעות)
- ✅ `PATCH /api/registrations/[id]` — ביטול חסום אם השיעור בפחות מ-48 שעות
- ✅ `POST /api/registrations/transfer` — העברה חסומה גם כן
- אדמין פטורה מההגבלה

### עמוד עריכת תלמיד
- ✅ `EditStudentForm.tsx` — כפתור "עריכה" ליד שם התלמיד
- עריכת שם, טלפון, וסוג עמדה לכל קבוצה
- ✅ `PATCH /api/students/[id]` — תומך עכשיו גם ב-`preferredSlotType` per enrollment

### שיפור ביצועים
- ✅ `vercel.json` — `{ "regions": ["bom1"] }` לקיצור מרחק לDB Mumbai

## ידוע ולא פתור
- ⚠️ איטיות: Vercel Hobby cold starts + מרחק DB (Mumbai). לא ניתן לפתרון מלא ללא שדרוג
- ⚠️ ה-pooler URL (eu-west-1) לא עובד מקומית — צריך להשתמש ב-DIRECT_URL

## משימות לסשן הבא

### 1. בדיקת end-to-end מלאה
- התחבר עם תלמידה (`michal@example.com / student123`)
- נווט לשבוע הבא ב-`/my` — בדוק שהלוח מציג נכון
- נסה לבטל שיעור / להעביר שיעור
- בדוק שהגבלת 48 שעות עובדת

### 2. עתידי
- SMS / התראות לתלמידים
- עמוד `/register` לתלמידים חדשים
- הוספת לוגו / עיצוב מותאם לענבל

## מיקום פרויקט
`/home/ori/fun/vibe_coding/inbal/`
