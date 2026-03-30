/**
 * cleanup-simulation.ts
 * מוחק את כל נתוני הסימולציה שנוצרו על ידי simulate-standby.ts
 * הרצה: npx tsx scripts/cleanup-simulation.ts
 */
import { PrismaClient } from "@prisma/client";
import { readFileSync } from "fs";
import { join } from "path";

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

async function main() {
  const statePath = join(process.cwd(), "scripts/.sim-state.json");
  let state: { oriId: string; sessionId: string; tempUserIds: string[] };
  try {
    state = JSON.parse(readFileSync(statePath, "utf-8"));
  } catch {
    console.error("לא נמצא קובץ מצב (.sim-state.json) — הרץ simulate-standby.ts קודם");
    process.exit(1);
  }

  const { oriId, sessionId, tempUserIds } = state;

  console.log("🧹 מנקה נתוני סימולציה...\n");

  // מחק standby entries של אורי
  const sb = await prisma.standbyEntry.deleteMany({ where: { userId: oriId, sessionId } });
  console.log(`✅ StandbyEntry של אורי: ${sb.count} נמחקו`);

  // מחק רישומי משתמשים זמניים
  const regs = await prisma.sessionRegistration.deleteMany({ where: { userId: { in: tempUserIds } } });
  console.log(`✅ SessionRegistration זמניות: ${regs.count} נמחקו`);

  // מחק משתמשים זמניים
  const users = await prisma.user.deleteMany({ where: { id: { in: tempUserIds } } });
  console.log(`✅ משתמשים זמניים: ${users.count} נמחקו`);

  // מחק משתמש אורי הסימולציה
  const ori = await prisma.user.deleteMany({ where: { id: oriId } });
  console.log(`✅ משתמש אורי (סימולציה): ${ori.count} נמחקו`);

  // שחזר רישומים שבוטלו — מחק את הביטולים הזמניים (הרישומים המקוריים שבוטלו ישארו ביטולים)
  // הערה: לא משחזרים CANCELLED חזרה ל-REGISTERED כי זו פעולה הרסנית יותר
  console.log(`ℹ️  רישום שבוטל לצורך הסימולציה נשאר CANCELLED (אפשר לשחזר ידנית מה-dashboard)`);

  // מחק קובץ מצב
  const { unlinkSync } = await import("fs");
  try { unlinkSync(statePath); } catch { /* already gone */ }

  console.log(`\n✅ ניקוי הסתיים.`);
}

main().catch(console.error).finally(() => prisma.$disconnect());
