import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import bcrypt from "bcryptjs";

export async function GET() {
  try {
    const user = await prisma.user.findUnique({
      where: { email: "admin@ceramics.co.il" },
      select: { id: true, email: true, role: true, password: true },
    });

    if (!user) return NextResponse.json({ error: "user not found" });

    const valid = await bcrypt.compare("admin123", user.password!);
    return NextResponse.json({
      found: true,
      role: user.role,
      hasPassword: !!user.password,
      passwordValid: valid,
    });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
