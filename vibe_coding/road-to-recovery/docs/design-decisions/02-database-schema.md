# סכמת DB — הסבר מפורט

## עקרונות עיצוב

1. **UUIDs לכל Primary Key** — נמנעת מ-sequential IDs (security, distributed systems)
2. **TIMESTAMPTZ** ולא TIMESTAMP — תמיד עם timezone (UTC)
3. **trigger לעדכון updated_at** — אוטומטי, לא ידני
4. **Foreign keys עם ON DELETE CASCADE** — לניקוי נכון של נתונים
5. **Indexes על שדות שאילתא נפוצים** — `scheduled_at`, `status`, `driver_id`

---

## טבלת `hospitals`

```sql
id          UUID PRIMARY KEY
name_he     TEXT NOT NULL      -- שם בעברית
name_ar     TEXT NOT NULL      -- שם בערבית
address     TEXT NOT NULL
city        TEXT NOT NULL DEFAULT 'ישראל'
lat         DECIMAL(10, 8)     -- קואורדינטות לחישוב מרחק
lng         DECIMAL(11, 8)
is_active   BOOLEAN DEFAULT true
```

**למה נפרד?** ניתן לשנות שם בית חולים בלי לפגוע בנסיעות היסטוריות (FK לפי id).

**Seed data:** 8 בתי חולים ראשיים בישראל (שיבא, סורוקה, רמב"ם, הדסה, איכילוב, זיו, נצרת, מאיר)

---

## טבלת `users`

```sql
id                  UUID REFERENCES auth.users(id) ON DELETE CASCADE
role                user_role ENUM
name_he, name_ar    TEXT              -- שני שדות שם (ראה החלטה 2)
phone               TEXT NOT NULL
email               TEXT UNIQUE NOT NULL
bank_account        TEXT              -- מוצפן AES-256 (ב-production)
language_preference language_preference ENUM DEFAULT 'he'
is_active           BOOLEAN DEFAULT true
notes               TEXT              -- הערות פנימיות לרכזת בלבד
```

**למה `id` מ-`auth.users`?** Supabase Auth ו-public.users חולקים אותו UUID.
מאפשר RLS עם `auth.uid()` שמצביע ישירות ל-row.

**`name_ar` אופציונלי** — נהגים ישראלים לא צריכים שם בערבית.

---

## טבלת `rides` — לב המערכת

### שדות רגישים vs. ציבוריים
| שדה | מי רואה |
|-----|---------|
| `medical_notes` | רכז פלסטיני + רכזת ישראלית |
| `driver_notes` | כולם |
| `patient_phone` | רכז פלסטיני + רכזת + נהג (בנסיעה שלו) |
| `pickup_lat/lng` | מחושב, לא מוצג ישירות |

### ניהול נסיעות חזרה

```
rides
  id: "OUTBOUND-ID"
  is_return_ride: false
  return_ride_id: "RETURN-ID"    ← אחרי שנוצרה חזרה
  status: 'return_needed'        ← אחרי סיום הלוך

rides
  id: "RETURN-ID"
  is_return_ride: true
  outbound_ride_id: "OUTBOUND-ID"
```

**Flow:**
1. נסיעת הלוך מסתיימת (`completed`)
2. API בודק: האם קיימת נסיעת חזרה (`outbound_ride_id=OUTBOUND-ID`)?
3. אם לא → עדכון ל-`return_needed` + notification לרכזת
4. רכזת יוצרת נסיעת חזרה → `return_ride_id` מתעדכן ב-הלוך

### Indexes
```sql
idx_rides_scheduled_at  -- שאילתות לוח יומי (הנפוצות ביותר)
idx_rides_status        -- פילטר לפי סטטוס
idx_rides_driver_id     -- נסיעות של נהג ספציפי
idx_rides_created_by    -- נסיעות של רכז פלסטיני
idx_rides_hospital_id   -- פילטר לפי בית חולים
```

---

## טבלת `reimbursement_requests`

```sql
UNIQUE(ride_id, driver_id)  -- נסיעה אחת = בקשה אחת בלבד
```

**חישוב `amount_ils`:**
```
amount_ils = distance_km × rate_per_km
           = X ק"מ × ₪1.50 = ₪Y
```

`rate_per_km` נשמר ב-row — מאפשר לשנות תעריף עתידי בלי להשפיע על בקשות עבר.

---

## טבלת `notifications`

### למה שני שדות טקסט (he + ar)?

חלופה שנדחתה: מחרוזת template + interpolation בזמן קריאה.
בעיה: interpolation בצד לקוח, קשה לlocalize ביטויים מורכבים.

**בחירה:** שמירת הטקסט המוכן בשתי שפות בזמן יצירה (server-side).
נהג מקבל notification בשפה שהוגדרה ב-`language_preference`.

### Partial Index
```sql
CREATE INDEX idx_notifications_unread
ON notifications(user_id, read)
WHERE read = false;
```
Index קטן יותר — רק על התראות שלא נקראו. שאילתת ה-badge count מהירה.

---

## RLS — עיצוב ובדיקה

### `get_my_role()` — למה SECURITY DEFINER?

בלי SECURITY DEFINER:
```sql
-- user רגיל לא יכול לקרוא users table אם אין לו policy
SELECT role FROM users WHERE id = auth.uid()  -- ← שגיאה!
```

עם SECURITY DEFINER:
```sql
-- הפונקציה רצה עם הרשאות ה-owner (postgres superuser)
-- המשתמש לא יכול לנצל זאת כי הוא קורא לפונקציה, לא לטבלה ישירות
```

### בדיקת RLS ב-development

```sql
-- בדוק כמשתמש ספציפי
SET LOCAL ROLE 'authenticated';
SET LOCAL request.jwt.claim.sub = 'driver-user-uuid';
SELECT * FROM rides;  -- צריך להחזיר רק pending + הנסיעות של הנהג
```

---

## נתוני Seed

### בתי חולים (8 בתי חולים)
נטענים ב-`supabase/schema.sql` בסוף הקובץ.

### משתמשי בדיקה (ליצור ב-Supabase Dashboard)
```
israeli@test.com / password123 → role: israeli_coordinator
pal@test.com / password123 → role: palestinian_coordinator
driver@test.com / password123 → role: driver
```
