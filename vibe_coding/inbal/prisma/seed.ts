import { PrismaClient } from "@prisma/client";
import bcrypt from "bcryptjs";

const prisma = new PrismaClient({
  datasources: { db: { url: process.env.DIRECT_URL || process.env.DATABASE_URL } },
});

function nextWeekday(dayOfWeek: number, offsetWeeks = 0): Date {
  const now = new Date();
  const day = now.getDay();
  const daysUntil = (dayOfWeek - day + 7) % 7 || 7;
  const d = new Date(now);
  d.setDate(d.getDate() + daysUntil + offsetWeeks * 7);
  d.setHours(0, 0, 0, 0);
  return d;
}

function setTime(date: Date, time: string): Date {
  const [h, m] = time.split(":").map(Number);
  const d = new Date(date);
  d.setHours(h, m, 0, 0);
  return d;
}

async function main() {
  console.log("🗑️  מוחק נתונים ישנים...");
  await prisma.payment.deleteMany();
  await prisma.sessionRegistration.deleteMany();
  await prisma.session.deleteMany();
  await prisma.groupEnrollment.deleteMany();
  await prisma.group.deleteMany();
  await prisma.user.deleteMany();

  console.log("👤 יוצר משתמשים...");
  const adminPassword = await bcrypt.hash("admin123", 10);
  const studentPassword = await bcrypt.hash("student123", 10);

  await prisma.user.create({
    data: {
      name: "ענבל",
      email: "admin@ceramics.co.il",
      phone: "050-1234567",
      role: "ADMIN",
      password: adminPassword,
    },
  });

  const students = await Promise.all([
    prisma.user.create({ data: { name: "מיכל לוי",   email: "michal@example.com",  phone: "052-1111111", role: "STUDENT", password: studentPassword } }),
    prisma.user.create({ data: { name: "יעל כהן",    email: "yael@example.com",    phone: "052-2222222", role: "STUDENT", password: studentPassword } }),
    prisma.user.create({ data: { name: "שרה אברהם",  email: "sara@example.com",    phone: "052-3333333", role: "STUDENT", password: studentPassword } }),
    prisma.user.create({ data: { name: "רחל גולד",   email: "rachel@example.com",  phone: "052-4444444", role: "STUDENT", password: studentPassword } }),
    prisma.user.create({ data: { name: "נועה שמיר",  email: "noa@example.com",     phone: "052-5555555", role: "STUDENT", password: studentPassword } }),
    prisma.user.create({ data: { name: "דנה ברק",    email: "dana@example.com",    phone: "052-6666666", role: "STUDENT", password: studentPassword } }),
    prisma.user.create({ data: { name: "תמר כץ",     email: "tamar@example.com",   phone: "052-7777777", role: "STUDENT", password: studentPassword } }),
    prisma.user.create({ data: { name: "אורית לב",   email: "orit@example.com",    phone: "052-8888888", role: "STUDENT", password: studentPassword } }),
  ]);

  const [s1, s2, s3, s4, s5, s6, s7, s8] = students;

  console.log("👥 יוצר קבוצות...");
  // dayOfWeek: 0=ראשון, 1=שני, 2=שלישי, 3=רביעי, 4=חמישי, 5=שישי, 6=שבת
  const groupDefs = [
    // שני
    { name: "שני 09:30",  dayOfWeek: 1, time: "09:30", enrollments: [
      { user: s1, slotType: "WHEEL"    as const },
      { user: s2, slotType: "WHEEL"    as const },
      { user: s3, slotType: "NO_WHEEL" as const },
    ]},
    { name: "שני 11:45",  dayOfWeek: 1, time: "11:45", enrollments: [
      { user: s4, slotType: "WHEEL"    as const },
      { user: s5, slotType: "WHEEL"    as const },
      { user: s6, slotType: "NO_WHEEL" as const },
      { user: s7, slotType: "NO_WHEEL" as const },
    ]},
    { name: "שני 18:15",  dayOfWeek: 1, time: "18:15", enrollments: [
      { user: s2, slotType: "WHEEL"    as const },
      { user: s8, slotType: "WHEEL"    as const },
      { user: s3, slotType: "NO_WHEEL" as const },
    ]},
    { name: "שני 20:30",  dayOfWeek: 1, time: "20:30", enrollments: [
      { user: s1, slotType: "NO_WHEEL" as const },
      { user: s6, slotType: "WHEEL"    as const },
    ]},
    // חמישי
    { name: "חמישי 12:15", dayOfWeek: 4, time: "12:15", enrollments: [
      { user: s4, slotType: "WHEEL"    as const },
      { user: s7, slotType: "WHEEL"    as const },
      { user: s8, slotType: "NO_WHEEL" as const },
    ]},
    { name: "חמישי 14:30", dayOfWeek: 4, time: "14:30", enrollments: [
      { user: s5, slotType: "WHEEL"    as const },
      { user: s6, slotType: "NO_WHEEL" as const },
      { user: s1, slotType: "NO_WHEEL" as const },
    ]},
    { name: "חמישי 18:15", dayOfWeek: 4, time: "18:15", enrollments: [
      { user: s2, slotType: "WHEEL"    as const },
      { user: s3, slotType: "WHEEL"    as const },
      { user: s7, slotType: "NO_WHEEL" as const },
      { user: s8, slotType: "WHEEL"    as const },
    ]},
    { name: "חמישי 20:30", dayOfWeek: 4, time: "20:30", enrollments: [
      { user: s4, slotType: "NO_WHEEL" as const },
      { user: s5, slotType: "WHEEL"    as const },
    ]},
    // שישי
    { name: "שישי 09:30",  dayOfWeek: 5, time: "09:30", enrollments: [
      { user: s1, slotType: "WHEEL"    as const },
      { user: s2, slotType: "NO_WHEEL" as const },
      { user: s6, slotType: "WHEEL"    as const },
      { user: s8, slotType: "NO_WHEEL" as const },
    ]},
  ];

  console.log("📅 יוצר שיעורים (8 שבועות)...");
  let totalSessions = 0;

  for (const def of groupDefs) {
    const group = await prisma.group.create({
      data: {
        name: def.name,
        dayOfWeek: def.dayOfWeek,
        time: def.time,
        duration: 120,
        location: "סטודיו ראשי",
        maxStudents: 7,
      },
    });

    // enrollments
    await Promise.all(
      def.enrollments.map((e) =>
        prisma.groupEnrollment.create({
          data: {
            userId: e.user.id,
            groupId: group.id,
            status: "ACTIVE",
            preferredSlotType: e.slotType,
          },
        })
      )
    );

    // 8 weeks of sessions
    for (let week = 0; week < 8; week++) {
      const baseDate = nextWeekday(def.dayOfWeek, week);
      const sessionDate = setTime(baseDate, def.time);
      const isPast = week < 2;

      const session = await prisma.session.create({
        data: {
          groupId: group.id,
          date: sessionDate,
          status: isPast ? "COMPLETED" : "SCHEDULED",
        },
      });

      for (let i = 0; i < def.enrollments.length; i++) {
        const enrollment = def.enrollments[i];
        let status: "REGISTERED" | "ABSENT" | "CANCELLED" = "REGISTERED";
        if (isPast && i === 0 && week === 0) status = "ABSENT";
        if (isPast && i === 1 && week === 1) status = "CANCELLED";

        await prisma.sessionRegistration.create({
          data: {
            sessionId: session.id,
            userId: enrollment.user.id,
            status,
            slotType: enrollment.slotType,
          },
        });
      }

      totalSessions++;
    }
  }

  console.log("💰 יוצר תשלומים...");
  for (const student of students.slice(0, 6)) {
    await prisma.payment.createMany({
      data: [
        { userId: student.id, amount: 350, description: "תשלום ינואר 2026", type: "MONTHLY", date: new Date("2026-01-05") },
        { userId: student.id, amount: 350, description: "תשלום פברואר 2026", type: "MONTHLY", date: new Date("2026-02-05") },
      ],
    });
  }
  for (const student of students.slice(6)) {
    await prisma.payment.create({
      data: { userId: student.id, amount: 350, description: "תשלום ינואר 2026", type: "MONTHLY", date: new Date("2026-01-05") },
    });
  }

  console.log("\n✅ Seed הושלם!");
  console.log(`👑 מנהלת:  admin@ceramics.co.il  /  admin123`);
  console.log(`🎓 תלמידה: michal@example.com    /  student123`);
  console.log(`\n📊 ${groupDefs.length} קבוצות, ${students.length} תלמידים, ${totalSessions} שיעורים`);
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect());
