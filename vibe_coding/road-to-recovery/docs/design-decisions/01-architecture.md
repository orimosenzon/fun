# החלטות ארכיטקטורה — מערכת בדרך להחלמה

## סקירה כללית

מערכת ווב לתיאום נסיעות חולים פלסטינים לבתי חולים ישראליים.
שלושה ממשקים נפרדים לשלושה תפקידים: רכז פלסטיני, רכזת ישראלית, נהג מתנדב.

---

## החלטה 1: Supabase Auth במקום NextAuth

**המצב:** הפרויקט הקיים (`inbal/`) משתמש ב-NextAuth.

**הבחירה:** Supabase Auth עם JWT מובנה.

**Rationale:**
- `auth.uid()` זמין ישירות בתוך PostgreSQL RLS policies
- עם NextAuth, היינו צריכים לשדר `user_id` לכל query ידנית — פותח פתח לבאגי security
- Supabase Auth מספק session management, token refresh, ו-cookie handling דרך `@supabase/ssr`
- פחות קוד, פחות נקודות כשל

**חלופה שנדחתה:** NextAuth + Prisma + PostgreSQL
- ידרוש כתיבת custom RLS logic בצד האפליקציה
- פחות secure (RLS מאכפת בDB, לא ב-application layer)

---

## החלטה 2: Route Groups לפי Role

**הבחירה:**
```
app/(dashboard)/coordinator/
app/(dashboard)/palestinian-coordinator/
app/(dashboard)/driver/
```

**Rationale:**
- כל ממשק עצמאי עם layout, sidebar, וניווט שונה
- `middleware.ts` מטפל ב-auth guard ו-redirect לפי role
- קל להוסיף ממשק חדש בעתיד (למשל: admin) בלי לשנות קוד קיים
- Next.js App Router Route Groups לא מופיעים ב-URL (טובים לארגון)

**חלופה שנדחתה:** נתיב אחד עם branch לפי role ב-component
- קוד ספגטי, קשה לתחזוקה, שגיאות permission קלות לשכוח

---

## החלטה 3: RLS כשכבת Security ראשית

**הבחירה:** כל טבלה מוגנת ע"י Row Level Security.

**Rationale:**
- הגנה ב-DB level — גם אם יש באג ב-API, DB מגן
- `get_my_role()` — helper function שרצה `SECURITY DEFINER STABLE`
  - `SECURITY DEFINER`: רצה עם הרשאות ה-owner (לא caller) — מניעת privilege escalation
  - `STABLE`: Postgres יכול לcache בתוך transaction — ביצועים
- RLS policies ברורות ומתועדות, לא logic מפוזר

**נקודה קריטית:**
- UPDATE ב-Postgres RLS לא יכול להגביל עמודות ספציפיות
- הגבלת שדות (למשל: נהג לא יכול לשנות `status` ישירות ל-`completed` בלי `in_progress`)
  מתבצעת ב-API route — לא רק ב-RLS

---

## החלטה 4: State Machine לסטטוסי נסיעה

**הבחירה:** `lib/utils/rideStatus.ts` מגדיר מעברים חוקיים בלבד.

**Rationale:**
- נסיעה לא יכולה לחזור מ-`completed` ל-`pending`
- Logic נאכף ב-API (`/api/rides/[id]/status`) לפני DB update
- UI רק מציג אפשרויות חוקיות — UX טוב יותר
- קל לדבג: מעברים חוקיים מוגדרים במקום אחד

**מעברים:**
```
pending → assigned (שיבוץ) / cancelled (ביטול)
assigned → in_progress (התחלה) / cancelled
in_progress → completed (סיום)
completed → return_needed (trigger אוטומטי)
cancelled → [סופי]
return_needed → [סופי - מטופל ע"י יצירת נסיעת חזרה]
```

---

## החלטה 5: distance_km נשמר ב-DB (לא מחושב מחדש)

**הבחירה:** חישוב מתרחש פעם אחת עם יצירת הנסיעה ב-server-side.

**Rationale:**
- Google Maps Distance Matrix API: $5 לכל 1000 קריאות
- 140 נסיעות/יום × 700 נהגים שצופים = 98,000 קריאות ליום → $490/יום = **יקר מאוד**
- שמירה פעם אחת: 140 קריאות ליום → **$0.70/יום**
- Fallback: אם API נכשל — הנסיעה נוצרת בלי מרחק (לא חוסמת פעולה)

**מימוש:** `/api/geocode` ו-`/api/rides` (POST) קוראים ל-`lib/maps.ts` בצד server

---

## החלטה 6: Race Condition בלקיחת נסיעה

**הבעיה:** 700 נהגים רואים אותן נסיעות פתוחות. שניים יכולים ללחוץ "קח" בו-זמנית.

**הפתרון:** UPDATE אטומי עם WHERE condition:
```sql
UPDATE rides
SET driver_id = $uid, status = 'assigned', assigned_at = NOW()
WHERE id = $id AND status = 'pending'
RETURNING *
```
- אם 0 שורות חזרות → 409 Conflict → UI מציג "הנסיעה כבר נלקחה"
- PostgreSQL מטפל ב-concurrency ברמת ה-DB — אין צורך ב-locks ב-application level

---

## החלטה 7: medical_notes vs driver_notes

**הבחירה:** שני שדות הערות נפרדים.

**Rationale:**
- `medical_notes`: אבחנות, מצב רפואי רגיש — רק רכז פלסטיני + רכזת ישראלית
- `driver_notes`: הוראות לוגיסטיות — "כניסה ראשית", "צריך כיסא גלגלים"
- API route של נהג תמיד select מפורש — ולעולם לא `select('*')`
- מגן על פרטיות מטופלים (HIPAA-like considerations)

---

## החלטה 8: i18n מקומי (לא ספריה)

**הבחירה:** `lib/i18n/` עם `he.json` + `ar.json` + hook פשוט.

**Rationale:**
- רק 2 שפות, מחרוזות ידועות מראש
- `next-intl`: ~50KB + complexity + SSR considerations
- `react-i18next`: ~100KB + configuration overhead
- Hook פשוט עם `localStorage` לשמירת העדפה: **~20 שורות קוד**
- עברית וערבית שתיהן RTL — `dir` של ה-HTML נשאר `rtl` תמיד

---

## החלטה 9: Supabase Realtime

**הבחירה:** `useNotifications.ts` ו-pages מסוימות משתמשים ב-Supabase Realtime Channels.

**Rationale:**
- רכזת רואה שנהג "לקח" נסיעה בלי refresh ידני
- נהג מקבל notification מיידי כששובץ
- Polling כל 5 שניות × 700+ משתמשים = עומס DB
- Supabase Realtime: WebSocket בנוי, כלול בחינם

**הגבלה:** Supabase Realtime לא מכבד RLS כברירת מחדל.
פתרון: סינון subscriptions לפי `user_id=eq.${userId}` — מבטיח שכל user רואה רק שלו.

---

## החלטה 10: bank_account מוצפן

**הבחירה:** AES-256 encryption ברמת האפליקציה לפני שמירה ב-DB.

**Rationale:**
- גם עם פריצה ל-DB: פרטי חשבון בנק לא נחשפים
- Encryption key נמצא ב-environment variable, לא ב-DB
- ל-MVP: Node.js `crypto` מובנה
- ל-Production: `SUPABASE_SERVICE_ROLE_KEY` + Supabase Vault

**הערה:** לא מומש ב-MVP הנוכחי — placeholder ב-schema. נדרש לפני production.

---

## מבנה קבצים וסיכום

```
lib/
  types/index.ts          ← TypeScript interfaces + state machine constants
  supabase/client.ts      ← Browser singleton client
  supabase/server.ts      ← Server client (per-request, cookie-aware)
  i18n/                   ← he.json + ar.json + hook
  utils/rideStatus.ts     ← State machine + color mapping
  utils/reimbursement.ts  ← חישוב סכום החזר
  utils/formatters.ts     ← תאריכים, מרחקים, סכומים
  hooks/useCurrentUser.ts ← Session + profile
  hooks/useNotifications.ts ← Realtime notifications
  maps.ts                 ← Google Maps wrapper (server-only)

app/
  middleware.ts           ← Auth guard + role-based routing
  (dashboard)/layout.tsx  ← Sidebar + Navbar (client component)
  api/...                 ← API routes עם role validation
```
