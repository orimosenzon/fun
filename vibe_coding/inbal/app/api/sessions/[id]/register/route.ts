import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { slotAvailable } from "@/lib/slots";
import { sendAdminSpotTaken } from "@/lib/whatsapp";

export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const session = await getServerSession(authOptions);
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { id: sessionId } = await params;
  const body = await req.json();
  const { slotType } = body as { slotType?: "WHEEL" | "NO_WHEEL" };

  const userSession = session.user as { email?: string; role?: string; id?: string };
  const isAdmin = userSession.role === "ADMIN";

  const user = await prisma.user.findUnique({ where: { email: userSession.email! } });
  if (!user) return NextResponse.json({ error: "User not found" }, { status: 404 });

  // מנהל יכול לרשום תלמיד אחר; תלמיד — רק את עצמו
  const targetUserId = isAdmin && body.userId ? body.userId : user.id;

  const classSession = await prisma.session.findUnique({
    where: { id: sessionId },
    include: { registrations: { select: { slotType: true, status: true, userId: true } } },
  });
  if (!classSession) return NextResponse.json({ error: "Session not found" }, { status: 404 });
  if (classSession.status !== "SCHEDULED") {
    return NextResponse.json({ error: "השיעור אינו פתוח לרישום" }, { status: 400 });
  }

  // בדוק שאין כבר רישום פעיל
  const existing = classSession.registrations.find(
    (r) => r.userId === targetUserId && r.status === "REGISTERED"
  );
  if (existing) {
    return NextResponse.json({ error: "כבר רשום/ה לשיעור זה" }, { status: 400 });
  }

  // קבע slotType — אם לא נשלח, נסה לפי preference של המשתמש
  let resolvedSlotType = slotType;
  if (!resolvedSlotType) {
    const enrollment = await prisma.groupEnrollment.findFirst({
      where: { userId: targetUserId, group: { sessions: { some: { id: sessionId } } } },
    });
    resolvedSlotType = (enrollment?.preferredSlotType as "WHEEL" | "NO_WHEEL") ?? "WHEEL";
  }

  // בדוק זמינות
  if (!slotAvailable(classSession.registrations, resolvedSlotType)) {
    return NextResponse.json({ error: "אין מקום פנוי בסלוט המבוקש" }, { status: 400 });
  }

  // בדוק אם יש standby entry לפני המחיקה (כדי לדעת אם לשלוח הודעה לאדמין)
  const hadStandby = await prisma.standbyEntry.findUnique({
    where: { userId_sessionId: { userId: targetUserId, sessionId } },
  });

  await prisma.standbyEntry.deleteMany({ where: { userId: targetUserId, sessionId } });

  // צור רישום (upsert — אולי יש רישום ישן מבוטל)
  const reg = await prisma.sessionRegistration.upsert({
    where: { userId_sessionId: { userId: targetUserId, sessionId } },
    update: { status: "REGISTERED", slotType: resolvedSlotType, note: null },
    create: { userId: targetUserId, sessionId, status: "REGISTERED", slotType: resolvedSlotType },
  });

  // אם תפסה מקום מרשימת המתנה — הודע לאדמין
  if (hadStandby) {
    const targetUser = await prisma.user.findUnique({
      where: { id: targetUserId },
      select: { name: true },
    });
    const admin = await prisma.user.findFirst({
      where: { role: "ADMIN" },
      select: { phone: true },
    });
    const fullSession = await prisma.session.findUnique({
      where: { id: sessionId },
      include: { group: { select: { name: true } } },
    });
    if (admin?.phone && fullSession) {
      const sessionLabel = `${fullSession.group.name} — ${fullSession.date.toLocaleString("he-IL", {
        weekday: "long",
        day: "numeric",
        month: "long",
        hour: "2-digit",
        minute: "2-digit",
      })}`;
      sendAdminSpotTaken(
        admin.phone,
        targetUser?.name ?? user.name ?? "",
        sessionLabel
      ).catch((err) => console.error("WhatsApp admin spot-taken failed:", err));
    }
  }

  return NextResponse.json(reg, { status: 201 });
}
