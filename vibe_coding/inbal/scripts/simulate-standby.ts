/**
 * simulate-standby.ts
 * מדמה מצב שבו אורי בסטנדביי ומתפנה מקום → שולח WhatsApp
 * הרצה: npx tsx scripts/simulate-standby.ts
 * ניקוי: npx tsx scripts/cleanup-simulation.ts
 */
import { PrismaClient } from "@prisma/client";
import { readFileSync } from "fs";
import { join } from "path";
import https from "https";

// טוען .env.local ידנית (Next.js לא טוען אותו מחוץ ל-dev server)
function loadEnv() {
  const content = readFileSync(join(process.cwd(), ".env.local"), "utf-8");
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIdx = trimmed.indexOf("=");
    if (eqIdx < 0) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    let val = trimmed.slice(eqIdx + 1).trim();
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'")))
      val = val.slice(1, -1);
    if (!process.env[key]) process.env[key] = val;
  }
}
loadEnv();

const prisma = new PrismaClient({
  datasources: { db: { url: process.env.DIRECT_URL || process.env.DATABASE_URL } },
});

const ORI_EMAIL   = "ori.sim@simulation.local";
const ORI_NAME    = "אורי (סימולציה)";
const ORI_PHONE   = "972508820997";
const INBAL_EMAIL = "admin@ceramics.co.il";
const INBAL_PHONE = "972544402058";

function sendTemplate(phone: string, templateName: string, params: string[], langCode = "he"): Promise<void> {
  let normalized = phone.replace(/\D/g, "");
  if (normalized.startsWith("0")) normalized = "972" + normalized.slice(1);

  const body = JSON.stringify({
    messaging_product: "whatsapp",
    to: normalized,
    type: "template",
    template: {
      name: templateName,
      language: { code: langCode },
      components: [{ type: "body", parameters: params.map((text) => ({ type: "text", text })) }],
    },
  });

  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        hostname: "graph.facebook.com",
        path: `/v21.0/${process.env.WHATSAPP_PHONE_NUMBER_ID}/messages`,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${process.env.WHATSAPP_ACCESS_TOKEN}`,
        },
      },
      (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
            console.log("   תגובת WhatsApp API:", data);
            resolve();
          } else {
            reject(new Error(`WhatsApp API שגיאה ${res.statusCode}: ${data}`));
          }
        });
      }
    );
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

function sendWhatsApp(phone: string, name: string): Promise<void> {
  return sendTemplate(phone, "standby_notification", [name]);
}

function sendAdminSpotOpened(phone: string, sessionLabel: string, waitlistText: string, firstPersonName: string): Promise<void> {
  return sendTemplate(phone, "admin_spot_opened", [sessionLabel, waitlistText, firstPersonName]);
}

function sendAdminSpotTaken(phone: string, studentName: string, sessionLabel: string): Promise<void> {
  return sendTemplate(phone, "_admin_spot_taken", [studentName, sessionLabel]);
}

async function main() {
  console.log("🔧 מכין סימולציה...\n");

  // 1. צור/עדכן את אורי כתלמיד
  const ori = await prisma.user.upsert({
    where: { email: ORI_EMAIL },
    update: { phone: ORI_PHONE, name: ORI_NAME },
    create: { email: ORI_EMAIL, name: ORI_NAME, phone: ORI_PHONE, role: "STUDENT", password: "sim-only" },
  });
  console.log(`👤 תלמיד: ${ori.name} | טלפון: ${ori.phone}`);

  // 2. עדכן טלפון של ענבל
  const inbal = await prisma.user.update({
    where: { email: INBAL_EMAIL },
    data: { phone: INBAL_PHONE },
  });
  console.log(`👑 מנהלת: ${inbal.name} | טלפון: ${inbal.phone}`);

  // 3. מצא שיעור עתידי מתוזמן
  const session = await prisma.session.findFirst({
    where: { status: "SCHEDULED", date: { gt: new Date() } },
    include: {
      registrations: { select: { id: true, slotType: true, status: true } },
      group: { select: { name: true } },
    },
    orderBy: { date: "asc" },
  });
  if (!session) throw new Error("לא נמצא שיעור עתידי מתוזמן");

  const activeRegs = session.registrations.filter((r) => r.status === "REGISTERED");
  const wheelCount   = activeRegs.filter((r) => r.slotType === "WHEEL").length;
  const noWheelCount = activeRegs.filter((r) => r.slotType === "NO_WHEEL").length;
  console.log(`\n📅 שיעור: ${session.group.name} — ${session.date.toLocaleString("he-IL")}`);
  console.log(`   מקומות לפני: ${wheelCount}/4 אובניים, ${noWheelCount}/3 ידני`);

  // 4. מלא את כל המקומות עם משתמשים זמניים (כדי שהשיעור יהיה מלא)
  const tempUserIds: string[] = [];

  for (let i = wheelCount; i < 4; i++) {
    const u = await prisma.user.create({
      data: { email: `sim-wheel-${i}@simulation.local`, name: `סים ${i}`, phone: `97200${i}00000`, role: "STUDENT", password: "sim" },
    });
    await prisma.sessionRegistration.create({
      data: { sessionId: session.id, userId: u.id, status: "REGISTERED", slotType: "WHEEL" },
    });
    tempUserIds.push(u.id);
    console.log(`   ➕ WHEEL מלא (${i + 1}/4)`);
  }
  for (let i = noWheelCount; i < 3; i++) {
    const u = await prisma.user.create({
      data: { email: `sim-nowheel-${i}@simulation.local`, name: `סים NW${i}`, phone: `97200${i}11111`, role: "STUDENT", password: "sim" },
    });
    await prisma.sessionRegistration.create({
      data: { sessionId: session.id, userId: u.id, status: "REGISTERED", slotType: "NO_WHEEL" },
    });
    tempUserIds.push(u.id);
    console.log(`   ➕ NO_WHEEL מלא (${i + 1}/3)`);
  }

  // שמור מידע לניקוי
  const { writeFileSync } = await import("fs");
  writeFileSync(
    join(process.cwd(), "scripts/.sim-state.json"),
    JSON.stringify({ oriId: ori.id, sessionId: session.id, tempUserIds }, null, 2)
  );

  // 5. הוסף את אורי לרשימת המתנה
  await prisma.standbyEntry.deleteMany({ where: { userId: ori.id, sessionId: session.id } });
  const standby = await prisma.standbyEntry.create({
    data: { userId: ori.id, sessionId: session.id, slotType: null },
  });
  console.log(`\n⏳ אורי בסטנדביי (id: ${standby.id})`);

  // 6. בטל רישום אחד → מקום מתפנה
  const toCancel = await prisma.sessionRegistration.findFirst({
    where: { sessionId: session.id, status: "REGISTERED", slotType: "WHEEL" },
  });
  if (!toCancel) throw new Error("לא נמצא רישום לביטול");
  await prisma.sessionRegistration.update({ where: { id: toCancel.id }, data: { status: "CANCELLED" } });
  console.log(`🔓 מקום WHEEL התפנה`);

  // 7. notifyNextStandby — מצא מועמד ושלח WhatsApp
  const updatedSession = await prisma.session.findUnique({
    where: { id: session.id },
    include: { registrations: { select: { slotType: true, status: true } } },
  });
  if (!updatedSession) throw new Error("שיעור לא נמצא");

  const candidate = await prisma.standbyEntry.findFirst({
    where: { sessionId: session.id, notifiedAt: null },
    include: { user: { select: { name: true, phone: true } } },
    orderBy: { createdAt: "asc" },
  });
  if (!candidate) throw new Error("לא נמצא מועמד בסטנדביי");

  const now = new Date();
  await prisma.standbyEntry.update({
    where: { id: candidate.id },
    data: { notifiedAt: now, expiresAt: new Date(now.getTime() + 60 * 60 * 1000) },
  });

  // 7א. שלח לאורי הודעת standby
  console.log(`\n📱 [1/3] שולח WhatsApp לאורי (standby_notification)...`);
  await sendWhatsApp(candidate.user.phone!, candidate.user.name ?? "");

  // 7ב. שלח לענבל שמקום נפתח (admin_spot_opened)
  const sessionLabel = `${session.group.name} — ${session.date.toLocaleDateString("he-IL")}`;
  const waitlistText = `${candidate.user.name}`;
  console.log(`\n📱 [2/3] שולח WhatsApp לענבל (admin_spot_opened)...`);
  await sendAdminSpotOpened(INBAL_PHONE, sessionLabel, waitlistText, candidate.user.name ?? "");

  // 7ג. סמולציה: אורי אישר → שלח לענבל שמקום נתפס (admin_spot_taken)
  console.log(`\n📱 [3/3] שולח WhatsApp לענבל (admin_spot_taken — אורי אישר)...`);
  await sendAdminSpotTaken(INBAL_PHONE, candidate.user.name ?? "", sessionLabel);

  console.log(`\n✅ סימולציה הצליחה! בדוק WhatsApp אצלך ואצל ענבל.`);
  console.log(`\n🧹 לניקוי: npx tsx scripts/cleanup-simulation.ts`);
}

main().catch(console.error).finally(() => prisma.$disconnect());
