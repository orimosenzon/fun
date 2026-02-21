import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { redirect } from "next/navigation";
import { prisma } from "@/lib/prisma";
import Link from "next/link";

export default async function PaymentsPage() {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role?: string })?.role !== "ADMIN") redirect("/my");

  const now = new Date();
  const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);

  // All students with payment status
  const students = await prisma.user.findMany({
    where: { role: "STUDENT" },
    include: {
      enrollments: { where: { status: "ACTIVE" } },
      payments: {
        where: { date: { gte: monthStart } },
        orderBy: { date: "desc" },
      },
    },
    orderBy: { name: "asc" },
  });

  // Recent payments
  const recentPayments = await prisma.payment.findMany({
    include: { user: { select: { id: true, name: true } } },
    orderBy: { date: "desc" },
    take: 20,
  });

  const thisMonthTotal = recentPayments
    .filter((p) => new Date(p.date) >= monthStart)
    .reduce((sum, p) => sum + p.amount, 0);

  const unpaidStudents = students.filter(
    (s) => s.enrollments.length > 0 && s.payments.length === 0
  );
  const paidStudents = students.filter((s) => s.payments.length > 0);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-stone-800">תשלומים</h1>
        <div className="text-sm text-stone-500">
          {new Date().toLocaleDateString("he-IL", { month: "long", year: "numeric" })}
        </div>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white rounded-xl p-5 border border-stone-200">
          <div className="text-3xl font-bold text-amber-600">₪{thisMonthTotal.toFixed(0)}</div>
          <div className="text-sm text-stone-500 mt-1">הכנסות החודש</div>
        </div>
        <div className="bg-white rounded-xl p-5 border border-green-200 bg-green-50">
          <div className="text-3xl font-bold text-green-700">{paidStudents.length}</div>
          <div className="text-sm text-stone-500 mt-1">שילמו החודש</div>
        </div>
        <div className="bg-white rounded-xl p-5 border border-red-200 bg-red-50">
          <div className="text-3xl font-bold text-red-600">{unpaidStudents.length}</div>
          <div className="text-sm text-stone-500 mt-1">לא שילמו</div>
        </div>
      </div>

      {/* Unpaid students */}
      {unpaidStudents.length > 0 && (
        <div className="bg-white rounded-xl border border-red-200 overflow-hidden">
          <div className="px-5 py-4 border-b border-red-100 bg-red-50">
            <h2 className="font-semibold text-red-800">לא שילמו החודש ({unpaidStudents.length})</h2>
          </div>
          <div className="divide-y divide-stone-100">
            {unpaidStudents.map((s) => (
              <Link
                key={s.id}
                href={`/students/${s.id}`}
                className="flex items-center justify-between px-5 py-3 hover:bg-stone-50 transition"
              >
                <span className="font-medium text-stone-800">{s.name}</span>
                <span className="text-xs text-red-600">לא שילם →</span>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Recent payments */}
      <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-stone-100">
          <h2 className="font-semibold text-stone-800">תשלומים אחרונים</h2>
        </div>
        <div className="divide-y divide-stone-100">
          {recentPayments.map((p) => (
            <div key={p.id} className="flex items-center justify-between px-5 py-3">
              <div>
                <Link href={`/students/${p.user.id}`} className="font-medium text-stone-800 hover:text-amber-700">
                  {p.user.name}
                </Link>
                <div className="text-xs text-stone-400">{p.description}</div>
              </div>
              <div className="text-left">
                <div className="font-semibold text-stone-800">₪{p.amount}</div>
                <div className="text-xs text-stone-400">
                  {new Date(p.date).toLocaleDateString("he-IL")}
                </div>
              </div>
            </div>
          ))}
          {recentPayments.length === 0 && (
            <div className="p-8 text-center text-stone-400">אין תשלומים עדיין</div>
          )}
        </div>
      </div>
    </div>
  );
}
