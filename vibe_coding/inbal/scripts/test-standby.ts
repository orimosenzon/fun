/**
 * בדיקות standby — רץ ישירות מול ה-DB
 * הרצה: npx tsx scripts/test-standby.ts
 *
 * הסקריפט:
 * 1. מוצא שיעור עתידי עם רישומים
 * 2. מוסיף תלמיד לתור המתנה
 * 3. מבטל רישום → מפעיל notifyNextStandby
 * 4. בודק שה-standby קיבל notifiedAt
 * 5. בודק cron cleanup
 * 6. מנקה אחרי עצמו
 */

import { PrismaClient } from "@prisma/client";
import { notifyNextStandby } from "../lib/standby";
import { slotAvailable } from "../lib/slots";

const prisma = new PrismaClient();

let passed = 0;
let failed = 0;

function ok(label: string) {
  console.log(`  ✅ ${label}`);
  passed++;
}

function fail(label: string, detail?: unknown) {
  console.log(`  ❌ ${label}`, detail ?? "");
  failed++;
}

async function main() {
  console.log("\n=== בדיקות Standby ===\n");

  // --- בחר שיעור מתאים ---
  console.log("🔍 מחפש שיעור עתידי עם רישומים...");
  const session = await prisma.session.findFirst({
    where: {
      date: { gt: new Date() },
      status: "SCHEDULED",
      registrations: { some: { status: "REGISTERED" } },
    },
    include: {
      registrations: {
        where: { status: "REGISTERED" },
        include: { user: true },
      },
      group: true,
    },
    orderBy: { date: "asc" },
  });

  if (!session || session.registrations.length === 0) {
    fail("לא נמצא שיעור מתאים לבדיקה (צריך שיעור עתידי עם לפחות רישום אחד)");
    return;
  }
  console.log(`  נמצא: "${session.group.name}" — ${session.date.toLocaleDateString("he-IL")}`);
  console.log(`  רישומים פעילים: ${session.registrations.length}`);

  // בחר מועמד לביטול
  const regToCancel = session.registrations[0];
  const cancelledUser = regToCancel.user;

  // --- מצא תלמיד אחר לסטנד ביי ---
  const standbyUser = await prisma.user.findFirst({
    where: {
      role: "STUDENT",
      id: { not: cancelledUser.id },
      registrations: { none: { sessionId: session.id, status: "REGISTERED" } },
      standbyEntries: { none: { sessionId: session.id } },
    },
  });

  if (!standbyUser) {
    fail("לא נמצא תלמיד פנוי לתור המתנה");
    return;
  }
  console.log(`  תלמיד לביטול: ${cancelledUser.name}`);
  console.log(`  תלמיד לסטנד ביי: ${standbyUser.name}\n`);

  // ==============================
  // בדיקה 1: הצטרפות לתור המתנה
  // ==============================
  console.log("📋 בדיקה 1: הצטרפות לתור המתנה");

  // ודא שהסלוט מלא (אחרת אי אפשר להצטרף לסטנד ביי)
  const sessionWithRegs = await prisma.session.findUnique({
    where: { id: session.id },
    include: { registrations: { select: { slotType: true, status: true } } },
  });
  const wheelFree = slotAvailable(sessionWithRegs!.registrations, "WHEEL");
  const noWheelFree = slotAvailable(sessionWithRegs!.registrations, "NO_WHEEL");

  let standbyEntry: { id: string; notifiedAt: Date | null; expiresAt: Date | null } | null = null;
  let createdStandby = false;

  if (!wheelFree || !noWheelFree) {
    // יש לפחות סוג אחד מלא — אפשר להצטרף
    const slotTypeForStandby = !wheelFree ? "WHEEL" : "NO_WHEEL";

    standbyEntry = await prisma.standbyEntry.create({
      data: {
        userId: standbyUser.id,
        sessionId: session.id,
        slotType: slotTypeForStandby,
      },
    });
    createdStandby = true;

    if (standbyEntry.notifiedAt === null) {
      ok(`נוצרה רשומת standby, notifiedAt=null (תקין)`);
    } else {
      fail("נוצרה רשומת standby אבל notifiedAt לא null");
    }
  } else {
    // שיעור לא מלא — אי אפשר לבדוק סטנד ביי "אמיתי"
    // ניצור ידנית
    console.log("  השיעור לא מלא — יוצר standby ידנית לצורך הבדיקה");
    standbyEntry = await prisma.standbyEntry.create({
      data: {
        userId: standbyUser.id,
        sessionId: session.id,
        slotType: null,
      },
    });
    createdStandby = true;
    ok("נוצרה רשומת standby ידנית");
  }

  // ==============================
  // בדיקה 2: ביטול → notifyNextStandby
  // ==============================
  console.log("\n📋 בדיקה 2: ביטול רישום → notifyNextStandby");

  // בטל את הרישום ישירות ב-DB (עוקף הגבלת 48 שעות)
  await prisma.sessionRegistration.update({
    where: { id: regToCancel.id },
    data: { status: "CANCELLED" },
  });
  ok(`רישום של ${cancelledUser.name} בוטל`);

  // הפעל notifyNextStandby
  await notifyNextStandby(session.id);

  // בדוק שה-standby קיבל notifiedAt
  const updatedEntry = await prisma.standbyEntry.findUnique({
    where: { id: standbyEntry!.id },
  });

  if (updatedEntry?.notifiedAt !== null) {
    ok(`notifiedAt הוגדר: ${updatedEntry?.notifiedAt?.toISOString()}`);
  } else {
    fail("notifiedAt נשאר null — notifyNextStandby לא עבד");
  }

  if (updatedEntry?.expiresAt !== null) {
    ok(`expiresAt הוגדר: ${updatedEntry?.expiresAt?.toISOString()}`);
  } else {
    fail("expiresAt נשאר null");
  }

  // בדוק שexpiresAt = notifiedAt + 1 שעה (בקירוב)
  if (updatedEntry?.notifiedAt && updatedEntry?.expiresAt) {
    const diffMs = updatedEntry.expiresAt.getTime() - updatedEntry.notifiedAt.getTime();
    const diffHours = diffMs / (1000 * 60 * 60);
    if (Math.abs(diffHours - 1) < 0.01) {
      ok(`expiresAt = notifiedAt + 1 שעה בדיוק`);
    } else {
      fail(`expiresAt שגוי — הפרש: ${diffHours.toFixed(2)} שעות`);
    }
  }

  // ==============================
  // בדיקה 3: לא שולח הודעה כפולה
  // ==============================
  console.log("\n📋 בדיקה 3: לא שולח הודעה כפולה לאותו standby");

  const notifiedAtBefore = updatedEntry?.notifiedAt;
  await notifyNextStandby(session.id); // קריאה שנייה

  const entryAfterSecondCall = await prisma.standbyEntry.findUnique({
    where: { id: standbyEntry!.id },
  });

  if (
    entryAfterSecondCall?.notifiedAt?.getTime() === notifiedAtBefore?.getTime()
  ) {
    ok("notifiedAt לא השתנה בקריאה שנייה (תקין — לא שולח כפול)");
  } else {
    fail("notifiedAt השתנה בקריאה שנייה — באג כפילות");
  }

  // ==============================
  // בדיקה 4: cron cleanup — מחיקת רשומות שפגו
  // ==============================
  console.log("\n📋 בדיקה 4: cron cleanup — מחיקת standby שפג");

  // צור רשומת standby "שפגה" (notifiedAt לפני שעתיים)
  const standbyUser2 = await prisma.user.findFirst({
    where: {
      role: "STUDENT",
      id: { notIn: [cancelledUser.id, standbyUser.id] },
      standbyEntries: { none: { sessionId: session.id } },
    },
  });

  let expiredEntryId: string | null = null;

  if (standbyUser2) {
    const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000);
    const oneHourAgo = new Date(Date.now() - 1 * 60 * 60 * 1000);

    const expiredEntry = await prisma.standbyEntry.create({
      data: {
        userId: standbyUser2.id,
        sessionId: session.id,
        slotType: null,
        notifiedAt: twoHoursAgo,
        expiresAt: oneHourAgo, // פג לפני שעה
      },
    });
    expiredEntryId = expiredEntry.id;

    // הפעל cron cleanup ידנית
    const now = new Date();
    const expired = await prisma.standbyEntry.findMany({
      where: {
        notifiedAt: { not: null },
        expiresAt: { lt: now },
      },
      select: { id: true, sessionId: true },
    });

    if (expired.some((e) => e.id === expiredEntryId)) {
      ok("הרשומה הפגויה נמצאה ע״י לוגיקת ה-cron");
    } else {
      fail("הרשומה הפגויה לא נמצאה ע״י לוגיקת ה-cron");
    }

    await prisma.standbyEntry.deleteMany({
      where: { id: { in: expired.map((e) => e.id) } },
    });

    const afterCleanup = await prisma.standbyEntry.findUnique({
      where: { id: expiredEntryId },
    });

    if (!afterCleanup) {
      ok("רשומה פגויה נמחקה בהצלחה");
    } else {
      fail("רשומה פגויה לא נמחקה");
    }
  } else {
    console.log("  ⚠️  אין תלמיד שלישי — דילוג על בדיקת cron cleanup");
  }

  // ==============================
  // בדיקה 5: לא מודיע כשאין מקום
  // ==============================
  console.log("\n📋 בדיקה 5: notifyNextStandby לא מגדיר notifiedAt כשאין מקום");

  // הפוך את הביטול — חזור לרישום פעיל (כך השיעור מלא שוב)
  await prisma.sessionRegistration.update({
    where: { id: regToCancel.id },
    data: { status: "REGISTERED" },
  });

  // אפס את notifiedAt של standby
  await prisma.standbyEntry.update({
    where: { id: standbyEntry!.id },
    data: { notifiedAt: null, expiresAt: null },
  });

  // הפעל notifyNextStandby כשהשיעור מלא
  await notifyNextStandby(session.id);

  const entryWhenFull = await prisma.standbyEntry.findUnique({
    where: { id: standbyEntry!.id },
  });

  // בדיקה זו תצליח רק אם השיעור אכן מלא
  const sessionRefresh = await prisma.session.findUnique({
    where: { id: session.id },
    include: { registrations: { select: { slotType: true, status: true } } },
  });
  const isNowFull =
    !slotAvailable(sessionRefresh!.registrations, "WHEEL") &&
    !slotAvailable(sessionRefresh!.registrations, "NO_WHEEL");

  if (isNowFull) {
    if (entryWhenFull?.notifiedAt === null) {
      ok("לא נשלחה הודעה כשהשיעור מלא (תקין)");
    } else {
      fail("נשלחה הודעה למרות שהשיעור מלא — באג!");
    }
  } else {
    console.log("  ⚠️  השיעור לא מלא לחלוטין — דילוג על בדיקה זו");
  }

  // ==============================
  // ניקוי
  // ==============================
  console.log("\n🧹 מנקה נתוני בדיקה...");

  if (createdStandby) {
    await prisma.standbyEntry.deleteMany({
      where: { id: standbyEntry!.id },
    });
  }

  // החזר את הרישום למצב המקורי
  await prisma.sessionRegistration.update({
    where: { id: regToCancel.id },
    data: { status: "REGISTERED" },
  });

  ok("ניקוי הושלם");

  // ==============================
  // סיכום
  // ==============================
  console.log(`\n${"=".repeat(40)}`);
  console.log(`סיכום: ${passed} עברו ✅  |  ${failed} נכשלו ❌`);
  console.log("=".repeat(40) + "\n");

  if (failed > 0) process.exit(1);
}

main()
  .catch((e) => {
    console.error("שגיאה לא צפויה:", e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
