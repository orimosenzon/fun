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

## מה נעשה בסשן הזה (2026-02-26)

### תיקוני באגים ב-StudentSessionCard
- ✅ `handleCancel` — בודק `res.ok` ומציג הודעת שגיאה אם ביטול נחסם (48h)
- ✅ `canTransferHere` — בודק שהשיעור המקורי נמצא לפחות 48 שעות קדימה

### שיפורי ביצועים
- ✅ `lib/prisma.ts` — Singleton מתוקן: שומר instance גם ב-production
- ✅ `app/(admin)/dashboard/page.tsx` — 4 שאילתות DB מקבילות עם `Promise.all`
- ✅ `app/(admin)/payments/page.tsx` — 2 שאילתות DB מקבילות עם `Promise.all`
- ✅ skeleton loading screens: `dashboard/loading.tsx`, `students/loading.tsx`, `groups/loading.tsx`, `payments/loading.tsx`

### תיקון deploy + יומן מנהל
- ✅ Vercel GitHub webhook לא עובד — יש לעשות deploy ידני: `npx vercel --prod --scope orimosenzons-projects`
- ✅ `/schedule` חזר לעבוד אחרי deploy ידני

### כפתור ניווט ביומן
- ✅ כיוון חצים תוקן ב-`AdminWeekCalendar.tsx` ו-`WeekCalendar.tsx`
  - `‹` שמאל = שבוע/יום קודם | `›` ימין = שבוע/יום הבא
  - הבר מרנדר LTR (לא RTL) — פריסה לפי סדר DOM
- ✅ כפתור "היום" נוסף לשני הלוחות (admin + student)

## ⚠️ בעיה פעילה
- Vercel GitHub webhook לא עובד! בכל פעם שעושים push → צריך לעשות גם deploy ידני:
  ```
  npx vercel --prod --scope orimosenzons-projects
  ```

## ידוע ולא פתור
- ⚠️ איטיות: Vercel Hobby cold starts + מרחק DB (Mumbai). לא ניתן לפתרון מלא ללא שדרוג
- ⚠️ ה-pooler URL (eu-west-1) לא עובד מקומית — צריך להשתמש ב-DIRECT_URL

## משימות לסשן הבא

### 1. בדיקת end-to-end מלאה
- התחבר עם תלמידה (`michal@example.com / student123`)
- נווט לשבוע הבא ב-`/my` — בדוק שהלוח מציג נכון
- נסה לבטל שיעור / להעביר שיעור
- בדוק שהגבלת 48 שעות עובדת

### 2. תיקון Vercel webhook
- בדוק בממשק Vercel → Settings → Git → Deploy Hooks
- אם שבור — מחק ויצור מחדש או הגדר GitHub Action כתחליף

### 3. עתידי
- SMS / התראות לתלמידים
- עמוד `/register` לתלמידים חדשים
- הוספת לוגו / עיצוב מותאם לענבל

## מיקום פרויקט
`/home/ori/fun/vibe_coding/inbal/`
