-- =============================================
-- "בדרך להחלמה" — Road to Recovery
-- Database Schema v1.0
-- =============================================

-- =============================================
-- EXTENSIONS
-- =============================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================
-- ENUMS
-- =============================================

CREATE TYPE user_role AS ENUM (
  'israeli_coordinator',       -- רכזת ישראלית (מנהל)
  'palestinian_coordinator',   -- רכז פלסטיני אזורי
  'driver'                     -- נהג מתנדב
);

CREATE TYPE ride_status AS ENUM (
  'pending',          -- הוגשה, ממתינה לשיבוץ
  'assigned',         -- שובץ נהג
  'in_progress',      -- בביצוע
  'completed',        -- הסתיימה
  'cancelled',        -- בוטלה
  'return_needed'     -- הלוך הסתיים, נדרשת נסיעת חזרה
);

CREATE TYPE reimbursement_status AS ENUM (
  'pending',    -- ממתין לאישור
  'approved',   -- אושר
  'rejected',   -- נדחה
  'paid'        -- שולם
);

CREATE TYPE notification_type AS ENUM (
  'ride_assigned',
  'ride_status_change',
  'return_ride_needed',
  'reimbursement_approved',
  'reimbursement_rejected',
  'new_ride_available'
);

CREATE TYPE language_preference AS ENUM ('he', 'ar');

-- =============================================
-- HELPER FUNCTION
-- =============================================

-- מחזירה את ה-role של המשתמש הנוכחי
-- SECURITY DEFINER = רצה עם הרשאות ה-owner, לא ה-caller
-- STABLE = Postgres יכול לcache בתוך transaction
CREATE OR REPLACE FUNCTION get_my_role()
RETURNS user_role AS $$
  SELECT role FROM public.users WHERE id = auth.uid()
$$ LANGUAGE sql SECURITY DEFINER STABLE;

-- trigger function לעדכון updated_at אוטומטי
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- HOSPITALS
-- ישות נפרדת — לא חלק מ-users
-- מאוחסנים עם שם בעברית וערבית
-- =============================================
CREATE TABLE hospitals (
  id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name_he     TEXT NOT NULL,
  name_ar     TEXT NOT NULL,
  address     TEXT NOT NULL,
  city        TEXT NOT NULL DEFAULT 'ישראל',
  lat         DECIMAL(10, 8) NOT NULL,
  lng         DECIMAL(11, 8) NOT NULL,
  is_active   BOOLEAN NOT NULL DEFAULT true,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================
-- USERS
-- מורחב מ-auth.users של Supabase
-- =============================================
CREATE TABLE users (
  id                    UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  role                  user_role NOT NULL,
  name_he               TEXT NOT NULL,
  name_ar               TEXT,                    -- אופציונלי (לנהגים שמדברים ערבית)
  phone                 TEXT NOT NULL,
  email                 TEXT UNIQUE NOT NULL,
  bank_account          TEXT,                    -- מוצפן AES-256 ברמת האפליקציה
  language_preference   language_preference NOT NULL DEFAULT 'he',
  is_active             BOOLEAN NOT NULL DEFAULT true,
  notes                 TEXT,                    -- הערות פנימיות — רכזת בלבד
  created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TRIGGER users_updated_at
  BEFORE UPDATE ON users
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================
-- RIDES
-- לב המערכת
-- =============================================
CREATE TABLE rides (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

  -- פרטי מטופל
  patient_name          TEXT NOT NULL,
  patient_phone         TEXT,

  -- לוגיסטיקה
  pickup_address        TEXT NOT NULL,
  pickup_lat            DECIMAL(10, 8),          -- מחושב ע"י Google Maps בעת יצירה
  pickup_lng            DECIMAL(11, 8),
  hospital_id           UUID NOT NULL REFERENCES hospitals(id),
  scheduled_at          TIMESTAMPTZ NOT NULL,

  -- סטטוס
  status                ride_status NOT NULL DEFAULT 'pending',

  -- קישורים
  created_by            UUID NOT NULL REFERENCES users(id),
  driver_id             UUID REFERENCES users(id),

  -- נסיעת חזרה
  is_return_ride        BOOLEAN NOT NULL DEFAULT false,
  outbound_ride_id      UUID REFERENCES rides(id),  -- הנסיעה ה"הלוך" (לנסיעות חזרה)
  return_ride_id        UUID REFERENCES rides(id),  -- נסיעת החזרה המקושרת

  -- מרחק (מחושב פעם אחת עם יצירה — לא בכל צפייה)
  distance_km           DECIMAL(8, 2),

  -- הערות — מופרדות: רפואי לרכזת, לוגיסטי לנהג
  medical_notes         TEXT,                    -- לא נחשף לנהג
  driver_notes          TEXT,                    -- נחשף לנהג

  -- timestamps מפורטים
  assigned_at           TIMESTAMPTZ,
  started_at            TIMESTAMPTZ,
  completed_at          TIMESTAMPTZ,
  cancelled_at          TIMESTAMPTZ,
  cancellation_reason   TEXT,

  created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TRIGGER rides_updated_at
  BEFORE UPDATE ON rides
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Indexes — שאילתות לפי תאריך הן הנפוצות ביותר
CREATE INDEX idx_rides_scheduled_at ON rides(scheduled_at);
CREATE INDEX idx_rides_status ON rides(status);
CREATE INDEX idx_rides_driver_id ON rides(driver_id);
CREATE INDEX idx_rides_created_by ON rides(created_by);
CREATE INDEX idx_rides_hospital_id ON rides(hospital_id);

-- =============================================
-- REIMBURSEMENT REQUESTS
-- =============================================
CREATE TABLE reimbursement_requests (
  id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  ride_id           UUID NOT NULL REFERENCES rides(id),
  driver_id         UUID NOT NULL REFERENCES users(id),

  distance_km       DECIMAL(8, 2) NOT NULL,
  rate_per_km       DECIMAL(6, 2) NOT NULL DEFAULT 1.50,  -- שקלים לק"מ
  amount_ils        DECIMAL(10, 2) NOT NULL,              -- = distance_km * rate_per_km

  status            reimbursement_status NOT NULL DEFAULT 'pending',

  approved_by       UUID REFERENCES users(id),
  approved_at       TIMESTAMPTZ,
  rejection_reason  TEXT,

  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- נסיעה אחת = בקשת החזר אחת בלבד
  UNIQUE(ride_id, driver_id)
);

CREATE TRIGGER reimbursement_updated_at
  BEFORE UPDATE ON reimbursement_requests
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE INDEX idx_reimbursement_driver_id ON reimbursement_requests(driver_id);
CREATE INDEX idx_reimbursement_status ON reimbursement_requests(status);

-- =============================================
-- NOTIFICATIONS
-- =============================================
CREATE TABLE notifications (
  id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  type        notification_type NOT NULL,

  -- כותרת ותוכן בשתי שפות
  title_he    TEXT NOT NULL,
  title_ar    TEXT NOT NULL,
  message_he  TEXT NOT NULL,
  message_ar  TEXT NOT NULL,

  -- קישור לנסיעה (אופציונלי)
  ride_id     UUID REFERENCES rides(id),

  read        BOOLEAN NOT NULL DEFAULT false,
  read_at     TIMESTAMPTZ,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_notifications_user_id ON notifications(user_id);
-- index חלקי — רק על ה-unread (קטן יותר, מהיר יותר לשאילתות badge)
CREATE INDEX idx_notifications_unread ON notifications(user_id, read) WHERE read = false;

-- =============================================
-- ROW LEVEL SECURITY
-- =============================================

ALTER TABLE hospitals ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE rides ENABLE ROW LEVEL SECURITY;
ALTER TABLE reimbursement_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;

-- HOSPITALS —————————————————————————————————

CREATE POLICY "hospitals_read_authenticated"
  ON hospitals FOR SELECT
  TO authenticated
  USING (is_active = true);

CREATE POLICY "hospitals_all_coordinator"
  ON hospitals FOR ALL
  TO authenticated
  USING (get_my_role() = 'israeli_coordinator');

-- USERS —————————————————————————————————————

-- כל משתמש רואה את עצמו
CREATE POLICY "users_read_self"
  ON users FOR SELECT
  TO authenticated
  USING (id = auth.uid());

-- רכזת ישראלית רואה ומנהלת הכל
CREATE POLICY "users_all_coordinator"
  ON users FOR ALL
  TO authenticated
  USING (get_my_role() = 'israeli_coordinator');

-- רכז פלסטיני רואה את עצמו בלבד (כבר מכוסה ע"י users_read_self)

-- נהגים — מותר לראות שמות נהגים אחרים (לא bank_account, לא notes)
-- מימוש: ב-API route — select מפורש ללא שדות רגישים
CREATE POLICY "users_read_drivers_basic"
  ON users FOR SELECT
  TO authenticated
  USING (
    role = 'driver'
    AND get_my_role() IN ('driver', 'israeli_coordinator', 'palestinian_coordinator')
  );

-- כל משתמש מעדכן את עצמו (לא role!)
CREATE POLICY "users_update_self"
  ON users FOR UPDATE
  TO authenticated
  USING (id = auth.uid())
  WITH CHECK (
    id = auth.uid()
    -- role לא ניתן לשינוי ע"י המשתמש עצמו
    AND role = (SELECT role FROM users WHERE id = auth.uid())
  );

-- RIDES —————————————————————————————————————

-- רכזת — הכל
CREATE POLICY "rides_all_coordinator"
  ON rides FOR ALL
  TO authenticated
  USING (get_my_role() = 'israeli_coordinator');

-- רכז פלסטיני — רק מה שיצר
CREATE POLICY "rides_select_pal_coordinator"
  ON rides FOR SELECT
  TO authenticated
  USING (
    get_my_role() = 'palestinian_coordinator'
    AND created_by = auth.uid()
  );

CREATE POLICY "rides_insert_pal_coordinator"
  ON rides FOR INSERT
  TO authenticated
  WITH CHECK (
    get_my_role() = 'palestinian_coordinator'
    AND created_by = auth.uid()
  );

-- נהג — נסיעות פתוחות (pending) + נסיעות שלו
CREATE POLICY "rides_select_driver"
  ON rides FOR SELECT
  TO authenticated
  USING (
    get_my_role() = 'driver'
    AND (
      status = 'pending'
      OR driver_id = auth.uid()
    )
  );

-- נהג — מעדכן רק נסיעות שלו (הגבלת שדות מתבצעת ב-API)
CREATE POLICY "rides_update_driver"
  ON rides FOR UPDATE
  TO authenticated
  USING (
    get_my_role() = 'driver'
    AND driver_id = auth.uid()
  );

-- REIMBURSEMENT REQUESTS ————————————————————

CREATE POLICY "reimbursements_all_coordinator"
  ON reimbursement_requests FOR ALL
  TO authenticated
  USING (get_my_role() = 'israeli_coordinator');

CREATE POLICY "reimbursements_select_own"
  ON reimbursement_requests FOR SELECT
  TO authenticated
  USING (
    get_my_role() = 'driver'
    AND driver_id = auth.uid()
  );

CREATE POLICY "reimbursements_insert_driver"
  ON reimbursement_requests FOR INSERT
  TO authenticated
  WITH CHECK (
    get_my_role() = 'driver'
    AND driver_id = auth.uid()
    AND EXISTS (
      SELECT 1 FROM rides
      WHERE rides.id = ride_id
      AND rides.driver_id = auth.uid()
      AND rides.status = 'completed'
    )
  );

-- NOTIFICATIONS ————————————————————————————

CREATE POLICY "notifications_own"
  ON notifications FOR ALL
  TO authenticated
  USING (user_id = auth.uid());

-- =============================================
-- SEED DATA — בתי חולים ראשיים בישראל
-- =============================================
INSERT INTO hospitals (name_he, name_ar, address, city, lat, lng) VALUES
  ('מרכז רפואי שיבא', 'المركز الطبي شيبا', 'דרך שיבא 2', 'תל השומר', 32.0550, 34.8400),
  ('בית חולים סורוקה', 'مستشفى سوروكا', 'רחוב פרומין 151', 'באר שבע', 31.2577, 34.8007),
  ('מרכז רפואי רמב"ם', 'المركز الطبي رمبام', 'דרך הרמב"ם', 'חיפה', 32.8133, 35.0333),
  ('בית חולים הדסה עין כרם', 'مستشفى هداسا عين كارم', 'קריית הדסה', 'ירושלים', 31.7741, 35.1535),
  ('מרכז רפואי תל אביב סוראסקי', 'المركز الطبي تل أبيب سوراسكي', 'וייצמן 6', 'תל אביב', 32.0880, 34.7900),
  ('בית חולים זיו', 'مستشفى زيف', 'דרך ירושלים 1', 'צפת', 32.9681, 35.4955),
  ('מרכז רפואי נצרת', 'المركز الطبي الناصرة', 'הגליל 1', 'נצרת', 32.7040, 35.2970),
  ('בית חולים מאיר', 'مستشفى مئير', 'כביש 4', 'כפר סבא', 32.1804, 34.9030);
