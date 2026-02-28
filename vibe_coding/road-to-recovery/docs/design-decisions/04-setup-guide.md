# מדריך הקמה — Development + Production

## שלב 1: Supabase Project

1. כנס ל-https://supabase.com/dashboard
2. צור project חדש ("road-to-recovery")
3. שמור: **Project URL** + **Anon Key** + **Service Role Key**
4. עבור ל-**SQL Editor** → הרץ את `supabase/schema.sql` בשלמותו

**בדיקה:** לאחר הרצה, עבור ל-Table Editor ווודא שקיימות הטבלאות:
`hospitals`, `users`, `rides`, `reimbursement_requests`, `notifications`

---

## שלב 2: Supabase Auth Settings

1. Authentication → Providers → Email: ודא שמופעל
2. Authentication → Settings:
   - **Site URL**: `http://localhost:3000` (development)
   - **Additional redirect URLs**: ריק לעכשיו

---

## שלב 3: Supabase Realtime

1. Database → Replication
2. הפעל Realtime על הטבלאות:
   - `rides` ✓
   - `notifications` ✓
   - (לא `users`, `hospitals` — לא נדרש Realtime)

---

## שלב 4: Google Maps API (אופציונלי ל-development)

1. https://console.cloud.google.com/apis/credentials
2. צור API Key חדש
3. הגבל ל-APIs: **Geocoding API** + **Distance Matrix API**
4. הגבל ל-IP של ה-server שלך (production)

**ל-development:** אם אין key — `/api/geocode` מחזיר mock data (45 ק"מ)

---

## שלב 5: .env.local

```bash
cp .env.local .env.local  # כבר קיים
```

מלא את הערכים:
```
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJ...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJ...
GOOGLE_MAPS_API_KEY=AIzaSy...
```

---

## שלב 6: הרצה

```bash
cd road-to-recovery
npm install
npm run dev
```

פתח: http://localhost:3000

---

## שלב 7: יצירת משתמשי בדיקה

### דרך Supabase Dashboard:
1. Authentication → Users → Invite User
2. צור 3 משתמשים:

| Email | Password | Role |
|-------|----------|------|
| coordinator@test.com | Test1234! | israeli_coordinator |
| pal@test.com | Test1234! | palestinian_coordinator |
| driver@test.com | Test1234! | driver |

### לאחר יצירה ב-Auth → הוסף ל-DB:
```sql
INSERT INTO users (id, role, name_he, phone, email) VALUES
  ('<auth-user-id-1>', 'israeli_coordinator', 'שרה כהן', '052-1111111', 'coordinator@test.com'),
  ('<auth-user-id-2>', 'palestinian_coordinator', 'محمد عمر', '050-2222222', 'pal@test.com'),
  ('<auth-user-id-3>', 'driver', 'דוד לוי', '054-3333333', 'driver@test.com');
```

**הערה:** ה-auth-user-id מופיע ב-Authentication → Users

---

## Production Deploy (Vercel)

```bash
# 1. Push לgithub
git init
git add .
git commit -m "feat: initial road-to-recovery MVP"
git remote add origin https://github.com/user/road-to-recovery.git
git push -u origin main

# 2. ב-Vercel Dashboard:
#    - Import repository
#    - הוסף environment variables (אותם כמו .env.local)
#    - עדכן Supabase Site URL ל-vercel domain
```

---

## E2E Test Flow

לבדיקה מלאה לאחר הקמה:

1. **רכז פלסטיני** (`pal@test.com`) → הגש בקשה חדשה
2. **רכזת ישראלית** (`coordinator@test.com`) → ראה בלוח היומי → שבץ נהג
3. **נהג** (`driver@test.com`) → ראה נסיעה → קח נסיעה → התחל → סיים
4. וודא: notification "נדרשת חזרה" הגיע לרכזת
5. **רכזת** → צור נסיעת חזרה
6. **נהג** → קח נסיעת חזרה → סיים → הגש בקשת החזר
7. **רכזת** → אשר/דחה בקשת החזר
8. וודא: notification הגיע לנהג
