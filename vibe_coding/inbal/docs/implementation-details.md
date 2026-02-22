# Implementation Details — Ceramics Studio App

## Tech Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Framework | Next.js (App Router) | 16.1 |
| Language | TypeScript | 5 |
| Styling | Tailwind CSS | 4 |
| ORM | Prisma | 5 |
| Database | PostgreSQL (Supabase) | — |
| Auth | NextAuth.js | 4 |
| Hosting | Vercel (planned) | — |

---

## Project Structure

```
inbal/
├── app/
│   ├── (admin)/          # Route group — admin pages
│   │   ├── layout.tsx    # checks user is logged in
│   │   ├── dashboard/    # main dashboard
│   │   ├── groups/       # groups (list, create, detail)
│   │   ├── sessions/     # individual session + attendance
│   │   ├── students/     # students (list, profile)
│   │   └── payments/     # payments
│   ├── (student)/        # Route group — student pages
│   │   └── my/           # personal dashboard
│   ├── api/              # API Routes (server-side)
│   │   ├── auth/         # NextAuth handler
│   │   ├── groups/       # CRUD for groups
│   │   ├── sessions/     # sessions + registrations
│   │   ├── registrations/# update attendance status
│   │   ├── students/     # students
│   │   ├── payments/     # payments
│   │   ├── enrollments/  # enroll in group
│   │   └── my/sessions   # personal sessions (student)
│   └── login/            # login page
├── components/
│   ├── Navbar.tsx        # main nav (role-aware)
│   └── SessionProvider.tsx # NextAuth client wrapper
├── lib/
│   ├── prisma.ts         # Prisma client (singleton)
│   └── auth.ts           # NextAuth config (providers + callbacks)
├── prisma/
│   ├── schema.prisma     # data model
│   ├── seed.ts           # sample data
│   └── migrations/       # DB migration history
└── docs/                 # documentation
```

---

## Data Model (Prisma Schema)

```
User
  ├── role: ADMIN | STUDENT
  ├── password (nullable — OAuth users don't have one)
  └── accounts[] → Account (Google/Facebook)

Group
  ├── dayOfWeek: 0–6 (Sunday–Saturday)
  ├── time: "10:00"
  ├── duration: minutes
  └── isActive: boolean

GroupEnrollment (User ↔ Group)
  └── status: ACTIVE | PAUSED | DROPPED

Session (a specific class occurrence)
  ├── groupId
  ├── date: DateTime
  └── status: SCHEDULED | CANCELLED | COMPLETED

SessionRegistration (User ↔ Session)
  ├── status: REGISTERED | CANCELLED | TRANSFERRED | ABSENT
  ├── transferredFromId (if student came from another group)
  └── transferredToId (if student was transferred to another group)

Payment
  ├── amount: Float
  ├── type: MONTHLY | SESSION | OTHER
  └── date: DateTime
```

---

## Core Business Logic

### Auto-generate sessions
`POST /api/groups/[id]/sessions`
- Calculates next date based on `dayOfWeek` and `time` of the group
- Creates N sessions (default: 8 weeks ahead)
- For each session — automatically registers all `ACTIVE` students in the group
- Skips weeks that already have a session

### Enroll student in group
`POST /api/enrollments`
- Creates a `GroupEnrollment`
- Adds a `SessionRegistration` for all future sessions in that group

### One-time cancellation
`PATCH /api/registrations/[id]`
- Changes status in `SessionRegistration` only
- Does not touch `GroupEnrollment` (student remains enrolled in the group)

---

## Authentication

### Providers
1. **Credentials** — email + bcrypt hash password check
2. **Google** — OAuth2 via NextAuth
3. **Facebook** — OAuth2 via NextAuth

### Role
- Role is stored in `User.role`
- Passed to JWT token via `jwt` callback
- Available in session via `session` callback
- Checked in every API route: `(session.user as { role: string }).role !== "ADMIN"`

### PrismaAdapter
- Automatically manages the `Account` table for OAuth accounts
- Uses JWT strategy (not DB-stored sessions)

---

## RTL / Hebrew
- `<html lang="he" dir="rtl">` in root layout
- Tailwind handles RTL correctly without extra config
- Dates displayed with `toLocaleDateString("he-IL")`

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL — transaction pooler (for the app) |
| `DIRECT_URL` | PostgreSQL — direct connection (for migrations) |
| `NEXTAUTH_SECRET` | JWT encryption key |
| `NEXTAUTH_URL` | App URL |
| `GOOGLE_CLIENT_ID/SECRET` | OAuth |
| `FACEBOOK_CLIENT_ID/SECRET` | OAuth |
