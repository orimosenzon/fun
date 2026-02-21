import { PrismaClient } from "@prisma/client";
import bcrypt from "bcryptjs";

const prisma = new PrismaClient();

const DAYS_HE = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

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
  // await prisma.account.deleteMany(); // uncomment after first migration
  await prisma.user.deleteMany();

  console.log("👤 יוצר משתמשים...");
  const adminPassword = await bcrypt.hash("admin123", 10);
  const studentPassword = await bcrypt.hash("student123", 10);

  const admin = await prisma.user.create({
    data: {
      name: "ענבל",
      email: "admin@ceramics.co.il",
      phone: "050-1234567",
      role: "ADMIN",
      password: adminPassword,
    },
  });

  const students = await Promise.all([
    prisma.user.create({ data: { name: "מיכל לוי", email: "michal@example.com", phone: "052-1111111", role: "STUDENT", password: studentPassword } }),
    prisma.user.create({ data: { name: "יעל כהן", email: "yael@example.com", phone: "052-2222222", role: "STUDENT", password: studentPassword } }),
    prisma.user.create({ data: { name: "שרה אברהם", email: "sara@example.com", phone: "052-3333333", role: "STUDENT", password: studentPassword } }),
    prisma.user.create({ data: { name: "רחל גולד", email: "rachel@example.com", phone: "052-4444444", role: "STUDENT", password: studentPassword } }),
    prisma.user.create({ data: { name: "נועה שמיר", email: "noa@example.com", phone: "052-5555555", role: "STUDENT", password: studentPassword } }),
    prisma.user.create({ data: { name: "דנה ברק", email: "dana@example.com", phone: "052-6666666", role: "STUDENT", password: studentPassword } }),
  ]);

  console.log("👥 יוצר קבוצות...");
  const groups = await Promise.all([
    prisma.group.create({ data: { name: "קבוצת בוקר ראשון", description: "מתחילים - בוקר", dayOfWeek: 0, time: "09:30", duration: 90, location: "סטודיו ראשי", maxStudents: 8 } }),
    prisma.group.create({ data: { name: "קבוצת צהריים שלישי", description: "מתקדמות", dayOfWeek: 2, time: "11:00", duration: 120, location: "סטודיו ראשי", maxStudents: 6 } }),
    prisma.group.create({ data: { name: "קבוצת ערב חמישי", description: "ערב למבוגרים", dayOfWeek: 4, time: "18:00", duration: 90, location: "סטודיו ראשי", maxStudents: 10 } }),
  ]);

  const [g1, g2, g3] = groups;
  const [s1, s2, s3, s4, s5, s6] = students;

  const enrollmentData = [
    { groupId: g1.id, userId: s1.id },
    { groupId: g1.id, userId: s2.id },
    { groupId: g1.id, userId: s3.id },
    { groupId: g2.id, userId: s2.id },
    { groupId: g2.id, userId: s4.id },
    { groupId: g2.id, userId: s5.id },
    { groupId: g3.id, userId: s3.id },
    { groupId: g3.id, userId: s4.id },
    { groupId: g3.id, userId: s5.id },
    { groupId: g3.id, userId: s6.id },
  ];

  await Promise.all(
    enrollmentData.map((e) =>
      prisma.groupEnrollment.create({ data: { ...e, status: "ACTIVE" } })
    )
  );

  console.log("📅 יוצר שיעורים (8 שבועות)...");
  for (const group of groups) {
    const groupStudents = enrollmentData.filter((e) => e.groupId === group.id).map((e) => e.userId);

    for (let week = 0; week < 8; week++) {
      const baseDate = nextWeekday(group.dayOfWeek, week);
      const sessionDate = setTime(baseDate, group.time);
      const isPast = week < 2;

      const session = await prisma.session.create({
        data: {
          groupId: group.id,
          date: sessionDate,
          status: isPast ? "COMPLETED" : "SCHEDULED",
        },
      });

      for (const userId of groupStudents) {
        let status: "REGISTERED" | "ABSENT" | "CANCELLED" = "REGISTERED";
        if (isPast && userId === groupStudents[0] && week === 0) status = "ABSENT";
        if (isPast && userId === groupStudents[1] && week === 1) status = "CANCELLED";

        await prisma.sessionRegistration.create({
          data: { sessionId: session.id, userId, status },
        });
      }
    }
  }

  console.log("💰 יוצר תשלומים...");
  for (const student of students.slice(0, 4)) {
    await prisma.payment.createMany({
      data: [
        { userId: student.id, amount: 350, description: "תשלום ינואר 2026", type: "MONTHLY", date: new Date("2026-01-05") },
        { userId: student.id, amount: 350, description: "תשלום פברואר 2026", type: "MONTHLY", date: new Date("2026-02-05") },
      ],
    });
  }
  for (const student of students.slice(4)) {
    await prisma.payment.create({
      data: { userId: student.id, amount: 350, description: "תשלום ינואר 2026", type: "MONTHLY", date: new Date("2026-01-05") },
    });
  }

  console.log("\n✅ Seed הושלם!");
  console.log(`👑 מנהלת:  admin@ceramics.co.il  /  admin123`);
  console.log(`🎓 תלמידה: michal@example.com    /  student123`);
  console.log(`\n📊 ${groups.length} קבוצות, ${students.length} תלמידים, ${8 * groups.length} שיעורים`);
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect());
