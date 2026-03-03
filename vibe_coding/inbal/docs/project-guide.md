# מדריך קבצי פרויקט inbal — סטודיו קרמיקה

## עץ קבצים

```
inbal/                          ← שורש הפרויקט
├── prisma/                     ← כל מה שקשור לבסיס הנתונים
│   ├── schema.prisma           ← מבנה בסיס הנתונים
│   └── seed.ts                 ← נתוני דוגמה לפיתוח
├── lib/                        ← לוגיקה משותפת — server בלבד
│   ├── auth.ts                 ← הגדרות אימות (NextAuth)
│   ├── prisma.ts               ← חיבור לDB (singleton)
│   ├── slots.ts                ← לוגיקת עמדות
│   ├── standby.ts              ← ניהול תור המתנה
│   └── twilio.ts               ← שליחת SMS
├── components/                 ← רכיבי UI משותפים לכל האפליקציה
│   ├── Navbar.tsx              ← סרגל ניווט עליון
│   └── SessionProvider.tsx     ← עטיפה ל-session
├── app/                        ← כל האפליקציה (Next.js App Router)
│   ├── layout.tsx              ← Layout שורש (RTL, עברית)
│   ├── page.tsx                ← Redirect לפי תפקיד
│   ├── globals.css             ← Tailwind + צבעים גלובליים
│   ├── login/                  ← עמוד כניסה (ציבורי)
│   │   └── page.tsx            ← עמוד כניסה
│   ├── register/               ← עמוד הרשמה עצמית (ציבורי)
│   │   └── page.tsx            ← עמוד הרשמה עצמית
│   ├── api/                    ← כל ה-API endpoints (Backend)
│   │   ├── auth/               ← אימות והרשמה
│   │   │   ├── [...nextauth]/route.ts  ← NextAuth handler
│   │   │   └── register/route.ts       ← הרשמה ציבורית
│   │   ├── students/           ← ניהול תלמידים (אדמין)
│   │   │   ├── route.ts        ← GET רשימה / POST יצירה (אדמין)
│   │   │   └── [id]/route.ts   ← GET פרטים / PATCH עריכה (אדמין)
│   │   ├── groups/             ← ניהול קבוצות
│   │   │   ├── route.ts        ← GET רשימה / POST יצירה
│   │   │   ├── [id]/route.ts   ← GET / PATCH / DELETE קבוצה
│   │   │   └── [id]/sessions/route.ts  ← POST יצירת שיעורים
│   │   ├── sessions/           ← ניהול שיעורים בודדים
│   │   │   └── [id]/           ← פעולות על שיעור ספציפי
│   │   │       ├── route.ts            ← GET / PATCH שיעור
│   │   │       ├── register/route.ts   ← POST רישום עצמי
│   │   │       ├── registrations/route.ts ← POST רישום (אדמין)
│   │   │       └── standby/route.ts    ← GET / POST / DELETE תור
│   │   ├── registrations/      ← פעולות על רישומים קיימים
│   │   │   ├── [id]/route.ts   ← PATCH שינוי סטטוס רישום
│   │   │   └── transfer/route.ts ← POST העברה בין שיעורים
│   │   ├── enrollments/route.ts ← POST/PATCH רישום לקבוצה
│   │   ├── my/sessions/route.ts ← GET שיעורי התלמיד הנוכחי
│   │   ├── payments/route.ts   ← GET / POST / DELETE תשלומים
│   │   └── cron/               ← משימות רקע אוטומטיות
│   │       └── process-standby/route.ts ← Cron כל 15 דקות
│   ├── (admin)/                ← ממשק אדמין (Route Group — לא מופיע ב-URL)
│   │   ├── layout.tsx          ← בדיקת הרשאות אדמין
│   │   ├── dashboard/          ← לוח בקרה ראשי
│   │   │   └── page.tsx        ← לוח בקרה ראשי
│   │   ├── schedule/           ← לוח שיעורים שבועי
│   │   │   ├── page.tsx        ← עמוד לוח השיעורים
│   │   │   └── AdminWeekCalendar.tsx ← תצוגת שבוע/יום
│   │   ├── groups/             ← ניהול קבוצות
│   │   │   ├── page.tsx        ← רשימת קבוצות
│   │   │   ├── new/page.tsx    ← יצירת קבוצה חדשה
│   │   │   └── [id]/           ← עמוד קבוצה ספציפית
│   │   │       ├── page.tsx              ← פרטי קבוצה
│   │   │       ├── AddStudentToGroup.tsx ← הוספת תלמיד
│   │   │       └── GenerateSessionsButton.tsx ← יצירת שיעורים
│   │   ├── students/           ← ניהול תלמידים
│   │   │   ├── page.tsx        ← רשימת תלמידים
│   │   │   ├── AddStudentForm.tsx ← הוספת תלמיד (modal)
│   │   │   └── [id]/           ← עמוד תלמיד ספציפי
│   │   │       ├── page.tsx                 ← פרטי תלמיד
│   │   │       ├── EditStudentForm.tsx      ← עריכת פרטים
│   │   │       ├── AddPaymentForm.tsx       ← רישום תשלום
│   │   │       └── ManageEnrollmentsSection.tsx ← ניהול קבוצות
│   │   ├── sessions/           ← פרטי שיעורים (אין רשימה — נכנסים מהלוח)
│   │   │   └── [id]/           ← עמוד שיעור ספציפי
│   │   │       ├── page.tsx             ← פרטי שיעור
│   │   │       ├── RegistrationActions.tsx ← כפתורי סטטוס
│   │   │       └── StandbyPanel.tsx     ← תצוגת תור המתנה
│   │   └── payments/           ← ניהול תשלומים
│   │       └── page.tsx        ← ניהול תשלומים
│   └── (student)/              ← ממשק תלמיד (Route Group — לא מופיע ב-URL)
│       ├── layout.tsx          ← בדיקת הרשאות תלמיד
│       └── my/                 ← עמוד "השיעורים שלי" (הדף היחיד של התלמיד)
│           ├── page.tsx              ← עמוד "השיעורים שלי"
│           ├── WeekCalendar.tsx      ← לוח שבועי
│           └── StudentSessionCard.tsx ← כרטיס שיעור + פעולות
├── docs/                       ← תיעוד הפרויקט
│   └── project-guide.md        ← המדריך הזה
├── vercel.json                 ← הגדרות Cron ל-Vercel
├── next.config.ts              ← הגדרות Next.js
└── package.json                ← תלויות הפרויקט
```

---

## הסבר מפורט לפי קטגוריות

---

### ⚙️ קבצי הגדרות

**`prisma/schema.prisma`**
מגדיר את מבנה כל הטבלאות ב-DB.
8 מודלים: `User`, `Account` (OAuth), `Group`, `GroupEnrollment`, `Session`, `SessionRegistration`, `Payment`, `StandbyEntry`. כולל Enum-ים לתפקידים, סטטוסים, וסוגי עמדות (WHEEL/NO_WHEEL).

**`vercel.json`**
מגדיר Cron שרץ כל 15 דקות ומפעיל את `/api/cron/process-standby` — טיפול בתור המתנה.

**`package.json`**
תלויות הפרויקט: Next.js 16, React 19, Prisma, NextAuth, bcryptjs, Twilio.

---

### 🔧 lib/ — לוגיקה משותפת

**`lib/auth.ts`**
הגדרות NextAuth — שלושה ספקים: אימייל/סיסמה (credentials), Google, Facebook. מצרף תפקיד ו-ID לכל session דרך JWT callback. מפנה משתמשים לא מחוברים ל-`/login`.

**`lib/prisma.ts`**
Singleton של Prisma Client. מונע יצירת חיבורים מרובים בזמן פיתוח (hot reload).

**`lib/slots.ts`**
לוגיקת עמדות: `WHEEL_SLOTS = 4`, `NO_WHEEL_SLOTS = 3`. הפונקציות `countSlots()` ו-`slotAvailable()` בודקות כמה עמדות פנויות לפי סוג.

**`lib/standby.ts`**
הלוגיקה המרכזית של תור המתנה. `notifyNextStandby()` בודק אם יש מקום פנוי ושולח SMS לראשון בתור, עם חלון זמן של שעה אחת להירשם.

**`lib/twilio.ts`**
עטיפה ל-Twilio API. `sendSms()` שולח SMS אם יש credentials בסביבה, אחרת מדפיס warning ולא נופל.

---

### 🔌 api/ — שרת (Backend)

#### אימות והרשמה

**`app/api/auth/[...nextauth]/route.ts`**
Handler סטנדרטי של NextAuth — מטפל בכל בקשות הכניסה/יציאה.

**`app/api/auth/register/route.ts`**
נקודת קצה ציבורית (ללא auth) להרשמת תלמידים חדשים. מאמת שדות, מצפין סיסמה, יוצר `User` עם תפקיד STUDENT.

#### תלמידים

**`app/api/students/route.ts`**
`GET` — רשימת כל התלמידים (אדמין בלבד).
`POST` — יצירת תלמיד חדש, רישום אוטומטי לשיעורים עתידיים אם נבחרה קבוצה.

**`app/api/students/[id]/route.ts`**
`GET` — פרטי תלמיד מלאים עם היסטוריה.
`PATCH` — עדכון שם/טלפון וסוג עמדה מועדפת לכל קבוצה.

#### קבוצות

**`app/api/groups/route.ts`**
`GET` — קבוצות פעילות עם הרישומים הבאים.
`POST` — יצירת קבוצה חדשה.

**`app/api/groups/[id]/route.ts`**
`GET/PATCH/DELETE` — קריאה, עדכון, מחיקה רכה של קבוצה.

**`app/api/groups/[id]/sessions/route.ts`**
`POST` — יוצר 8 שיעורים קדימה לפי יום/שעה של הקבוצה. רושם אוטומטית את כל התלמידים הפעילים.

#### שיעורים ורישומים

**`app/api/sessions/[id]/register/route.ts`**
רישום עצמי לשיעור. בודק זמינות עמדות, מוחק מהתור (אם היה), יוצר/מעדכן `SessionRegistration`.

**`app/api/registrations/[id]/route.ts`**
שינוי סטטוס רישום. ביטול מוגבל ל-48 שעות מראש עבור תלמידים. ביטול מפעיל `notifyNextStandby()`.

**`app/api/registrations/transfer/route.ts`**
העברת רישום משיעור לשיעור אחר. בודק זמינות, מבצע בtransaction אטומי, מודיע לתור על המקום שהתפנה.

#### תור המתנה

**`app/api/sessions/[id]/standby/route.ts`**
`POST` — הצטרפות לתור.
`DELETE` — עזיבה.
`GET` — רשימת התור (אדמין בלבד).

#### כספים וCron

**`app/api/payments/route.ts`**
ניהול תשלומים — יצירה, קריאה, מחיקה. לאדמין בלבד.

**`app/api/cron/process-standby/route.ts`**
רץ כל 15 דקות דרך Vercel Cron. מנקה פניות שפג תוקפן ומזמין את הבא בתור.

---

### 🎨 components/ — רכיבים משותפים

**`components/Navbar.tsx`**
סרגל ניווט עליון. מציג קישורים שונים לאדמין ולתלמיד. מדגיש את הדף הנוכחי. כפתור התנתקות.

**`components/SessionProvider.tsx`**
עטיפה ל-NextAuth SessionProvider — מאפשר לכל הקומפוננטות לגשת ל-session המשתמש.

---

### 📄 app/ — עמודי UI

**`app/layout.tsx`**
Layout שורשי. מגדיר `dir="rtl"`, עברית, ו-`SessionProvider` לכל האפליקציה.

**`app/page.tsx`**
עמוד בית שמפנה: אדמין → `/dashboard`, תלמיד → `/my`, לא מחובר → `/login`.

**`app/login/page.tsx`**
עמוד כניסה. טופס אימייל/סיסמה + כפתורי Google ו-Facebook. קישור לעמוד הרשמה.

**`app/register/page.tsx`**
עמוד הרשמה עצמית לתלמידים. שדות: שם, אימייל, טלפון, סיסמה. לאחר הרשמה — כניסה אוטומטית.

---

### 🔐 (admin)/ — ממשק אדמין

**`(admin)/layout.tsx`**
בודק שהמשתמש הוא ADMIN, אחרת מפנה ל-`/my`.

**`(admin)/dashboard/page.tsx`**
לוח בקרה: מספר תלמידים/קבוצות, הכנסה חודשית, 5 שיעורים קרובים.

**`(admin)/schedule/page.tsx` + `AdminWeekCalendar.tsx`**
לוח שיעורים שבועי. מציג ריבועי עמדות (מלאות/ריקות) ושמות תלמידים. תצוגת שבוע או יום.

**`(admin)/groups/`**
- `page.tsx` — רשימת קבוצות
- `new/page.tsx` — יצירה
- `[id]/page.tsx` — פרטים + רשימת תלמידים ושיעורים
- `AddStudentToGroup.tsx` — הוספת תלמיד לקבוצה
- `GenerateSessionsButton.tsx` — יצירת 8 שיעורים נוספים

**`(admin)/students/`**
- `page.tsx` — רשימת תלמידים עם מצב תשלום
- `AddStudentForm.tsx` — הוספה ידנית
- `[id]/page.tsx` — פרטי תלמיד מלאים
- `EditStudentForm.tsx` — עריכת פרטים
- `AddPaymentForm.tsx` — רישום תשלום
- `ManageEnrollmentsSection.tsx` — ניהול שיוך לקבוצות

**`(admin)/sessions/[id]/`**
- `page.tsx` — פרטי שיעור עם רשימת נרשמים וסטטוסים
- `RegistrationActions.tsx` — שינוי סטטוס (נוכח/נעדר/ביטל)
- `StandbyPanel.tsx` — תצוגת תור המתנה + ניהולו

**`(admin)/payments/page.tsx`**
הכנסה חודשית, מי לא שילם, היסטוריית תשלומים.

---

### 🎓 (student)/ — ממשק תלמיד

**`(student)/layout.tsx`**
בודק שהמשתמש מחובר, אחרת מפנה ללוגין.

**`(student)/my/page.tsx`**
עמוד "השיעורים שלי". טוען שיעורים לשבוע הנוכחי, רישומים קיימים, ומיקום בתור.

**`(student)/my/WeekCalendar.tsx`**
לוח שבועי עם ניווט. מציג ימים ושיעורים ומדגיש היום.

**`(student)/my/StudentSessionCard.tsx`**
הכרטיס המרכזי של השיעור. מציג: שעות, זמינות עמדות, כפתורי רישום/ביטול/העברה/הצטרפות לתור. מכיל מספר modals.

---

### 🗄️ זרימת נתונים מרכזית

```
DB (Supabase PostgreSQL)
        ↓
    Prisma ORM (lib/prisma.ts)
        ↓
    API Routes (app/api/)
        ↓
  Server Components (page.tsx)  ←→  Client Components (.tsx)
        ↓
    NextAuth Session (lib/auth.ts)
        ↓
    User Browser
```
