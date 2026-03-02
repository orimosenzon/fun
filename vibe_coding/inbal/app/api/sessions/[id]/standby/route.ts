import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { slotAvailable, WHEEL_SLOTS, NO_WHEEL_SLOTS } from "@/lib/slots";

// POST — הצטרפות לרשימת המתנה
export async function POST(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const session = await getServerSession(authOptions);
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { id: sessionId } = await params;
  const body = await req.json();
  const { slotType } = body as { slotType?: "WHEEL" | "NO_WHEEL" };

  const userSession = session.user as { email?: string };
  const user = await prisma.user.findUnique({ where: { email: userSession.email! } });
  if (!user) return NextResponse.json({ error: "User not found" }, { status: 404 });

  const classSession = await prisma.session.findUnique({
    where: { id: sessionId },
    include: {
      registrations: { select: { slotType: true, status: true, userId: true } },
    },
  });
  if (!classSession) return NextResponse.json({ error: "Session not found" }, { status: 404 });
  if (classSession.status !== "SCHEDULED") {
    return NextResponse.json({ error: "השיעור אינו פתוח" }, { status: 400 });
  }

  // אם כבר רשום ופעיל — אין צורך בסטנד ביי
  const activeReg = classSession.registrations.find(
    (r) => r.userId === user.id && r.status === "REGISTERED"
  );
  if (activeReg) {
    return NextResponse.json({ error: "כבר רשום/ה לשיעור זה" }, { status: 400 });
  }

  // בדוק שהסלוט המבוקש אכן מלא
  const checkType = slotType ?? "WHEEL";
  const wheelFull = !slotAvailable(classSession.registrations, "WHEEL");
  const noWheelFull = !slotAvailable(classSession.registrations, "NO_WHEEL");

  if (slotType && slotAvailable(classSession.registrations, slotType)) {
    return NextResponse.json({ error: "יש מקום פנוי — ניתן להירשם ישירות" }, { status: 400 });
  }
  if (!slotType && !wheelFull && !noWheelFull) {
    return NextResponse.json({ error: "יש מקום פנוי — ניתן להירשם ישירות" }, { status: 400 });
  }

  // בדוק שאין כבר סטנד ביי לשיעור הזה
  const existingStandby = await prisma.standbyEntry.findUnique({
    where: { userId_sessionId: { userId: user.id, sessionId } },
  });
  if (existingStandby) {
    return NextResponse.json({ error: "כבר ברשימת המתנה לשיעור זה" }, { status: 400 });
  }

  const entry = await prisma.standbyEntry.create({
    data: { userId: user.id, sessionId, slotType: slotType ?? null },
  });

  // חשב מיקום בתור
  const position = await prisma.standbyEntry.count({
    where: { sessionId, createdAt: { lte: entry.createdAt } },
  });

  return NextResponse.json({ ...entry, position }, { status: 201 });
}

// DELETE — עזיבת רשימת המתנה (תלמיד = עצמו, אדמין = userId param)
export async function DELETE(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const session = await getServerSession(authOptions);
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { id: sessionId } = await params;
  const userSession = session.user as { email?: string; role?: string };
  const isAdmin = userSession.role === "ADMIN";

  // אדמין יכול להעביר userId כ-query param
  const url = new URL(req.url);
  const targetUserIdParam = url.searchParams.get("userId");

  let targetUserId: string;
  if (isAdmin && targetUserIdParam) {
    targetUserId = targetUserIdParam;
  } else {
    const user = await prisma.user.findUnique({ where: { email: userSession.email! } });
    if (!user) return NextResponse.json({ error: "User not found" }, { status: 404 });
    targetUserId = user.id;
  }

  await prisma.standbyEntry.deleteMany({ where: { userId: targetUserId, sessionId } });
  return NextResponse.json({ ok: true });
}

// GET — רשימת המתנה (אדמין בלבד)
export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const session = await getServerSession(authOptions);
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const userSession = session.user as { role?: string };
  if (userSession.role !== "ADMIN") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { id: sessionId } = await params;
  const entries = await prisma.standbyEntry.findMany({
    where: { sessionId },
    include: { user: { select: { id: true, name: true, phone: true, email: true } } },
    orderBy: { createdAt: "asc" },
  });

  return NextResponse.json(entries);
}
