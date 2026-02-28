# API Contract — כל ה-Endpoints

## עקרונות כלליים

- **Auth:** כל endpoint דורש session תקף (Supabase JWT cookie)
- **RLS:** DB מסנן נתונים לפי role — API לא חייב לעשות הכל ידנית
- **Errors:** פורמט אחיד: `{ error: string }`
- **Success:** `{ data: T }` או `{ data: T[] }`

---

## Rides

### `GET /api/rides`

| Query Param | Type | Description |
|-------------|------|-------------|
| `date` | YYYY-MM-DD | לוח יומי |
| `status` | RideStatus | פילטר סטטוס |
| `hospital_id` | UUID | פילטר בית חולים |

**Response:** `RideWithDetails[]` (עם join ל-hospitals, users)

**RLS:** רכז פלסטיני רואה רק שלו; נהג רואה pending + שלו; רכזת הכל

---

### `POST /api/rides`

**Auth:** `palestinian_coordinator` בלבד (RLS מאכפת)

**Body:**
```typescript
{
  patient_name: string          // required
  patient_phone?: string
  pickup_address: string        // required — נ-geocoded אוטומטית
  hospital_id: string           // required
  scheduled_at: string          // required — ISO 8601
  medical_notes?: string
  driver_notes?: string
  is_return_ride?: boolean
  outbound_ride_id?: string     // required אם is_return_ride=true
}
```

**Logic:**
1. קריאה ל-Google Maps (server-side) לחישוב מרחק + geocoding
2. אם API נכשל — נסיעה נוצרת בלי מרחק (לא blocking)
3. אם `is_return_ride=true`: עדכון הנסיעה ה-outbound עם `return_ride_id`

**Response:** `201` עם הנסיעה שנוצרה

---

### `GET /api/rides/[id]`

**Response:** `RideWithDetails` (רכזת — כולל `medical_notes`; נהג — בלי)

---

### `PATCH /api/rides/[id]`

**Auth:** `israeli_coordinator` בלבד

**Body:** שדות חלקיים (רק: patient_name, patient_phone, pickup_address, scheduled_at, medical_notes, driver_notes)

---

### `POST /api/rides/[id]/take`

**Auth:** `driver` בלבד

**Logic:** UPDATE אטומי עם `WHERE status='pending'`
- `200` — הצליח
- `409 Conflict` — הנסיעה כבר נלקחה
- Side effect: notification לרכזת

---

### `PATCH /api/rides/[id]/status`

**Body:**
```typescript
{
  status: RideStatus
  cancellation_reason?: string  // required כש-status='cancelled'
}
```

**Validation:**
1. שאילתת נסיעה נוכחית
2. נהג רק לנסיעות שלו
3. בדיקת `getAllowedTransitions()` — state machine
4. Side effect: כש-`completed` + לא return ride → `return_needed` + notifications

---

### `POST /api/rides/[id]/assign`

**Auth:** `israeli_coordinator` בלבד

**Body:** `{ driver_id: string }`

**Logic:**
1. וידוא driver פעיל
2. UPDATE עם `WHERE status IN ('pending', 'assigned')` — ניתן להחליף נהג
3. notification לנהג

---

## Reimbursements

### `GET /api/reimbursements`

**RLS:** נהג רק שלו; רכזת הכל

---

### `POST /api/reimbursements`

**Auth:** `driver` בלבד

**Body:** `{ ride_id: string }`

**Validation:**
1. נסיעה קיימת + driver_id = המשתמש + status = `completed`
2. אין בקשה קיימת (UNIQUE constraint → 409)

**חישוב:** `amount_ils = distance_km × 1.50`

---

### `PATCH /api/reimbursements/[id]`

**Auth:** `israeli_coordinator` בלבד

**Body:**
```typescript
{
  action: 'approve' | 'reject'
  rejection_reason?: string  // required כש-action='reject'
}
```

**Logic:** UPDATE רק על `status='pending'`; notification לנהג

---

## Users

### `GET /api/users`

**Query Param:** `role=driver` (שאר roles: רכזת בלבד)

**שדות לפי role:**
- רכזת: הכל כולל email, notes
- שאר: id, name_he, name_ar, phone (ללא bank_account, email, notes)

---

### `POST /api/users`

**Auth:** `israeli_coordinator` בלבד

**Logic:**
1. יצירת auth user ב-Supabase Admin API (SERVICE_ROLE_KEY)
2. הכנסה ל-`public.users`
3. Rollback: אם הכנסה ל-DB נכשלה → מחיקת auth user

---

## Hospitals

### `GET /api/hospitals`

**Auth:** כל authenticated user

**Response:** `Hospital[]` (רק `is_active=true`)

---

## Geocode

### `POST /api/geocode`

**Auth:** כל authenticated user (קריאת API נעשית server-side)

**Body:** `{ pickup_address: string, hospital_id: string }`

**Response:**
```typescript
{
  distanceKm: number
  durationMinutes: number
  pickupLat: number
  pickupLng: number
  _mock?: true  // רק ב-development אם GOOGLE_MAPS_API_KEY לא מוגדר
}
```

---

## HTTP Status Codes

| Code | משמעות |
|------|--------|
| 200 | הצלחה |
| 201 | נוצר |
| 400 | שדה חסר / input שגוי |
| 401 | לא מחובר |
| 403 | מחובר אבל אין הרשאה |
| 404 | לא נמצא |
| 409 | Conflict (race condition, כפילות) |
| 500 | שגיאת server |
