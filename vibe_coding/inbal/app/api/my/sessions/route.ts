import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const user = await prisma.user.findUnique({
    where: { email: session.user!.email! },
  });
  if (!user) return NextResponse.json({ error: "User not found" }, { status: 404 });

  const registrations = await prisma.sessionRegistration.findMany({
    where: {
      userId: user.id,
      session: { date: { gte: new Date() }, status: "SCHEDULED" },
    },
    include: {
      session: { include: { group: { select: { id: true, name: true, location: true } } } },
    },
    orderBy: { session: { date: "asc" } },
  });

  const payments = await prisma.payment.findMany({
    where: { userId: user.id },
    orderBy: { date: "desc" },
    take: 5,
  });

  return NextResponse.json({ registrations, payments, userId: user.id });
}
