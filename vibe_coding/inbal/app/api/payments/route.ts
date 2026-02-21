import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role: string }).role !== "ADMIN") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { searchParams } = new URL(req.url);
  const userId = searchParams.get("userId");

  const payments = await prisma.payment.findMany({
    where: userId ? { userId } : {},
    include: { user: { select: { id: true, name: true, email: true } } },
    orderBy: { date: "desc" },
  });
  return NextResponse.json(payments);
}

export async function POST(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role: string }).role !== "ADMIN") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const body = await req.json();
  const payment = await prisma.payment.create({
    data: {
      userId: body.userId,
      amount: Number(body.amount),
      description: body.description,
      type: body.type || "MONTHLY",
      date: body.date ? new Date(body.date) : new Date(),
    },
    include: { user: { select: { id: true, name: true } } },
  });
  return NextResponse.json(payment, { status: 201 });
}

export async function DELETE(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role: string }).role !== "ADMIN") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }
  const { id } = await req.json();
  await prisma.payment.delete({ where: { id } });
  return NextResponse.json({ ok: true });
}
