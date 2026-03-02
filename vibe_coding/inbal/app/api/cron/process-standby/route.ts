import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { notifyNextStandby } from "@/lib/standby";

/**
 * Cron job — רץ כל 15 דקות.
 * מוחק סטנד ביי שתם תוקפם (notifiedAt + 1h עברה ולא נרשמו)
 * ומודיע לבא אחריהם בתור.
 */
export async function GET(req: Request) {
  const secret = process.env.CRON_SECRET;
  if (secret) {
    const auth = req.headers.get("authorization");
    if (auth !== `Bearer ${secret}`) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
  }

  const now = new Date();

  // מצא כל הרשומות שפג תוקפן
  const expired = await prisma.standbyEntry.findMany({
    where: {
      notifiedAt: { not: null },
      expiresAt: { lt: now },
    },
    select: { id: true, sessionId: true },
  });

  if (expired.length === 0) {
    return NextResponse.json({ processed: 0 });
  }

  // מחק את הרשומות שפגו
  await prisma.standbyEntry.deleteMany({
    where: { id: { in: expired.map((e) => e.id) } },
  });

  // קבץ לפי sessionId ושלח הודעה לבא בתור
  const sessionIds = [...new Set(expired.map((e) => e.sessionId))];
  await Promise.allSettled(sessionIds.map((sid) => notifyNextStandby(sid)));

  return NextResponse.json({ processed: expired.length });
}
