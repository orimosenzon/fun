import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const groups = await prisma.group.findMany({
    where: { isActive: true },
    include: {
      enrollments: {
        where: { status: "ACTIVE" },
        include: { user: { select: { id: true, name: true, email: true } } },
      },
      sessions: {
        where: { date: { gte: new Date() }, status: "SCHEDULED" },
        orderBy: { date: "asc" },
        take: 1,
      },
    },
    orderBy: { dayOfWeek: "asc" },
  });
  return NextResponse.json(groups);
}

export async function POST(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role: string }).role !== "ADMIN") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const body = await req.json();
  const group = await prisma.group.create({ data: body });
  return NextResponse.json(group, { status: 201 });
}
