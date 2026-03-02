import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { notifyNextStandby } from "@/lib/standby";

const CANCEL_HOURS_AHEAD = 48;

export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const session = await getServerSession(authOptions);
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const { id } = await params;
  const body = await req.json();

  const reg = await prisma.sessionRegistration.findUnique({
    where: { id },
    include: { session: true },
  });
  if (!reg) return NextResponse.json({ error: "Not found" }, { status: 404 });

  const userSession = session.user as { email?: string; role?: string };
  const isAdmin = userSession.role === "ADMIN";

  if (!isAdmin) {
    const user = await prisma.user.findUnique({ where: { email: userSession.email! } });
    if (!user || reg.userId !== user.id) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    // 48-hour cancellation restriction for students
    if (body.status === "CANCELLED") {
      const hoursUntilSession = (reg.session.date.getTime() - Date.now()) / (1000 * 60 * 60);
      if (hoursUntilSession < CANCEL_HOURS_AHEAD) {
        return NextResponse.json(
          { error: `לא ניתן לבטל פחות מ-${CANCEL_HOURS_AHEAD} שעות לפני השיעור` },
          { status: 400 }
        );
      }
    }
  }

  const updated = await prisma.sessionRegistration.update({ where: { id }, data: body });

  // אם בוטל — הודע לסטנד ביי
  if (body.status === "CANCELLED") {
    notifyNextStandby(reg.sessionId).catch((err) =>
      console.error("notifyNextStandby failed:", err)
    );
  }

  return NextResponse.json(updated);
}
