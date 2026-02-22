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

## Authentication
Three options:
1. Email + password
2. Google
3. Facebook

A student added by the admin gets an initial password (`student123`) and can change it later.
