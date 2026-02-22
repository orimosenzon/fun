# Ceramics Studio Management App — What It Does

## Purpose
A web app for managing a ceramics studio. Built for a ceramics teacher who runs multiple student groups and needs help tracking registrations, cancellations, transfers, and payments.

---

## Two Interfaces

### Admin Interface (the Teacher)
- **Main Dashboard** — upcoming sessions: how many are coming, who cancelled, who transferred. Stats: number of students, groups, monthly revenue.
- **Group Management** — create a group (name, day, time, duration, location, capacity). Auto-generate 8 sessions in advance. View enrolled students and sessions.
- **Session Management** — per session: full attendance list with status per student (attending / cancelled / transferred / absent). Update attendance in real time.
- **Student Management** — list of all students, add new student. Per student: their groups, attendance history, payment history.
- **Payment Management** — central view: who paid this month, who didn't. Add payment for a student (amount, description, date, type). Current month revenue.

### Student Interface
- **My Dashboard** — my groups, upcoming sessions, this month's payment status.
- **Cancel a Session** — one-time cancellation for a specific session (does not remove the student from the group).

---

## Main Flows

| Flow | Who | What happens |
|------|-----|--------------|
| Create group | Admin | Group is created + 8 sessions generated |
| Enroll student in group | Admin | Student is registered for all future sessions |
| One-time cancellation | Student / Admin | That session's status changes to "cancelled" |
| Transfer student | Admin | Status changes to "transferred", student appears in another session |
| Record payment | Admin | Payment is logged for student with details and date |
| Session rollover | — | Manual — add more sessions as needed |

---

## Weekly Schedule

The studio runs 9 fixed sessions per week:

| Day | Time |
|-----|------|
| Monday | 09:30–11:30 |
| Monday | 11:45–13:45 |
| Monday | 18:15–20:15 |
| Monday | 20:30–22:30 |
| Thursday | 12:15–14:15 |
| Thursday | 14:30–16:30 |
| Thursday | 18:15–20:15 |
| Thursday | 20:30–22:30 |
| Friday | 09:30–11:30 |

---

## Slots System

Each session has **8 slots**:
- 4 slots **with a pottery wheel** (אובן)
- 3 slots **without a pottery wheel**
- 1 **extra slot** — reserved for the teacher's discretion (special cases)

Every student has a **default slot** — their regular assigned spot across sessions.

### Student view (calendar-style, by week)
- Sees all 9 sessions in the week laid out like a calendar
- Per session: which slots are taken, which are free
- Can self-assign to an available slot
- Sees their own default slot highlighted

### Admin view (same calendar, richer)
- Sees the full name of the student in every occupied slot
- Can reassign any student to any slot in any session
- Can use the extra (8th) slot to add a student beyond the normal 7

---

## Authentication
Three options:
1. Email + password
2. Google
3. Facebook

A student added by the admin gets an initial password (`student123`) and can change it later.
