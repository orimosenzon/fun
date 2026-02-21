import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function PATCH(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const session = await getServerSession(authOptions);
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const { id } = await params;
  const body = await req.json();

  const reg = await prisma.sessionRegistration.findUnique({ where: { id } });
  if (!reg) return NextResponse.json({ error: "Not found" }, { status: 404 });

  const userSession = session.user as { email?: string; role?: string };
  if (userSession.role !== "ADMIN") {
    const user = await prisma.user.findUnique({ where: { email: userSession.email! } });
    if (!user || reg.userId !== user.id) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }
  }

  const updated = await prisma.sessionRegistration.update({ where: { id }, data: body });
  return NextResponse.json(updated);
}
