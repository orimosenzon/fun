# Next Steps

## Current Status ✅
- App is built and runs locally (`npm run dev`)
- Supabase PostgreSQL connected (project: `twboneztjbagmllufzwn`, region: EU West)
- Database migrated (`prisma migrate dev`)
- Seed data loaded: 3 groups, 6 students, 24 sessions, payments
- Test accounts: `admin@ceramics.co.il / admin123` | `michal@example.com / student123`

---

## Phase A — Deploy to Production (Urgent)

### 1. Push code to GitHub
- [ ] Create a GitHub repository (public or private)
- [ ] `git remote add origin <github-url>`
- [ ] `git push -u origin master`

### 2. Deploy to Vercel
- [ ] Sign up / log in at [vercel.com](https://vercel.com)
- [ ] Import the GitHub repository
- [ ] Add all environment variables in Vercel project settings:
  - `DATABASE_URL` (transaction pooler URL from Supabase)
  - `DIRECT_URL` (direct connection URL from Supabase)
  - `NEXTAUTH_SECRET` (generate a strong random string)
  - `NEXTAUTH_URL` (your Vercel app URL, e.g. `https://inbal.vercel.app`)
  - `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` (see step 3)
  - `FACEBOOK_CLIENT_ID` / `FACEBOOK_CLIENT_SECRET` (optional)
- [ ] Click Deploy — first deploy

### 3. Google OAuth (so students can log in with Google)
- [ ] Go to [console.cloud.google.com](https://console.cloud.google.com)
- [ ] Create a new project
- [ ] Enable "Google+ API" or "Google Identity"
- [ ] Create OAuth 2.0 credentials (Web application)
- [ ] Add authorized redirect URI: `https://[your-domain]/api/auth/callback/google`
- [ ] Copy client ID and secret → add to Vercel env vars

---

## Phase B — Slots & Scheduling (Core Feature)

### Data model changes
- [ ] Replace the current `Group` model with a fixed weekly schedule (9 hardcoded sessions — days + times)
- [ ] Add `Slot` model: `sessionId`, `type` (WHEEL | NO_WHEEL | EXTRA), `position` (1–8)
- [ ] Add `SlotAssignment`: `slotId`, `userId`, `isDefault` (whether this is the student's default slot)

### Admin features
- [ ] Full weekly calendar view — all 9 sessions, all 8 slots per session, student names visible
- [ ] Drag or click to reassign any student to any slot
- [ ] Set/change a student's default slot (persists across weeks)
- [ ] Use the 8th (extra) slot to add a student beyond the normal 7

### Student features
- [ ] Weekly calendar view — same 9 sessions, showing free/taken per slot (no names)
- [ ] Self-assign to any free slot in any session
- [ ] See their own default slot highlighted
- [ ] Cancel their assignment for a specific session

---

## Phase C — Missing Features (Important)

### Session transfers
Currently only cancellation is supported. Need to add:
- [ ] UI for student to choose an alternative session
- [ ] Update `transferredFromId` and `transferredToId`
- [ ] Show "transferred from..." in session detail view

### Change password
- [ ] Settings page for student — change password
- [ ] "Forgot password" email (requires setting up an email provider)

### Export and reports
- [ ] Export session attendance as PDF or Excel
- [ ] Monthly report — revenue per group
- [ ] Export student list with payment status

---

## Phase D — UX Improvements

### Notifications (important)
- [ ] **WhatsApp / SMS** — automatic message to student who cancelled (cancellation confirmation)
- [ ] Reminder the day before a session
- [ ] Notify all students when teacher cancels a session

### Session management
- [ ] Admin-initiated session cancellation (not just attendance updates)
- [ ] Add a one-off session (outside the regular schedule)
- [ ] Reschedule a session to a different date

### Admin interface
- [ ] Visual calendar view of all sessions
- [ ] Monthly revenue chart
- [ ] Session topic / materials list

### Enrollments and payments
- [ ] Fixed monthly subscription (recurring charge, not manual)
- [ ] Self-registration for new students (not only via admin)
- [ ] Waiting list for full groups

---

## Phase E — Reliability and Security

- [ ] Rate limiting on API routes
- [ ] Verify a student can only cancel their own sessions
- [ ] Audit log — who changed what and when
- [ ] Automatic DB backup (Supabase handles this automatically on paid plans)
- [ ] Set a real `NEXTAUTH_SECRET` in production (not the dev default)

---

## Future Ideas (Not Urgent)

- Mobile app (React Native / PWA)
- Self-registration with online payment (Stripe / Tranzila)
- Google Calendar integration — sessions appear in student's calendar
- Work gallery — student uploads a photo of what they made in class
