import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { redirect, notFound } from "next/navigation";
import { prisma } from "@/lib/prisma";
import Link from "next/link";
import AddPaymentForm from "./AddPaymentForm";

export default async function StudentPage({ params }: { params: Promise<{ id: string }> }) {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role?: string })?.role !== "ADMIN") redirect("/my");

  const { id } = await params;
  const student = await prisma.user.findUnique({
    where: { id },
    include: {
      enrollments: {
        include: { group: true },
        orderBy: { startDate: "asc" },
      },
      registrations: {
        include: { session: { include: { group: { select: { name: true } } } } },
        orderBy: { session: { date: "desc" } },
        take: 20,
      },
      payments: { orderBy: { date: "desc" } },
    },
  });

  if (!student) notFound();

  const now = new Date();
  const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);
  const totalPaid = student.payments.reduce((sum, p) => sum + p.amount, 0);
  const paidThisMonth = student.payments.some((p) => new Date(p.date) >= monthStart);

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <Link href="/students" className="text-sm text-stone-400 hover:text-stone-600 mb-1 block">
          ← תלמידים
        </Link>
        <h1 className="text-2xl font-bold text-stone-800">{student.name}</h1>
        <div className="text-stone-500 text-sm mt-1 flex gap-4">
          <span>{student.email}</span>
          {student.phone && <span>{student.phone}</span>}
        </div>
      </div>

      {/* Groups */}
      <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-stone-100">
          <h2 className="font-semibold text-stone-800">קבוצות</h2>
        </div>
        <div className="divide-y divide-stone-100">
          {student.enrollments.map((e) => (
            <div key={e.id} className="px-5 py-3 flex items-center justify-between">
              <Link href={`/groups/${e.group.id}`} className="font-medium hover:text-amber-700">
                {e.group.name}
              </Link>
              <span className={`text-xs px-2 py-0.5 rounded-full ${
                e.status === "ACTIVE" ? "bg-green-100 text-green-700" :
                e.status === "PAUSED" ? "bg-yellow-100 text-yellow-700" :
                "bg-stone-100 text-stone-500"
              }`}>
                {e.status === "ACTIVE" ? "פעיל" : e.status === "PAUSED" ? "מושהה" : "עזב"}
              </span>
            </div>
          ))}
          {student.enrollments.length === 0 && (
            <div className="px-5 py-4 text-stone-400 text-sm">לא רשום לקבוצות</div>
          )}
        </div>
      </div>

      {/* Payments */}
      <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-stone-100 flex items-center justify-between">
          <div>
            <h2 className="font-semibold text-stone-800">תשלומים</h2>
            <div className="text-sm text-stone-500">סה&quot;כ: ₪{totalPaid.toFixed(0)}</div>
          </div>
          <div className="flex items-center gap-3">
            <span className={`text-xs px-2 py-1 rounded-full font-medium ${
              paidThisMonth ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
            }`}>
              {paidThisMonth ? "שילם החודש ✓" : "לא שילם החודש ✗"}
            </span>
            <AddPaymentForm studentId={id} studentName={student.name} />
          </div>
        </div>

        <div className="divide-y divide-stone-100">
          {student.payments.map((p) => (
            <div key={p.id} className="px-5 py-3 flex items-center justify-between">
              <div>
                <div className="text-sm font-medium text-stone-700">{p.description}</div>
                <div className="text-xs text-stone-400">
                  {new Date(p.date).toLocaleDateString("he-IL")}
                </div>
              </div>
              <div className="font-semibold text-stone-800">₪{p.amount}</div>
            </div>
          ))}
          {student.payments.length === 0 && (
            <div className="px-5 py-4 text-stone-400 text-sm">אין תשלומים</div>
          )}
        </div>
      </div>

      {/* Attendance */}
      <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-stone-100">
          <h2 className="font-semibold text-stone-800">היסטוריית שיעורים</h2>
        </div>
        <div className="divide-y divide-stone-100 max-h-72 overflow-y-auto">
          {student.registrations.map((r) => {
            const statusMap: Record<string, { label: string; cls: string }> = {
              REGISTERED: { label: "רשום", cls: "text-green-700 bg-green-50" },
              CANCELLED: { label: "ביטל", cls: "text-red-700 bg-red-50" },
              TRANSFERRED: { label: "הועבר", cls: "text-blue-700 bg-blue-50" },
              ABSENT: { label: "נעדר", cls: "text-stone-500 bg-stone-50" },
            };
            const st = statusMap[r.status] || { label: r.status, cls: "bg-stone-50 text-stone-500" };
            return (
              <div key={r.id} className="px-5 py-2.5 flex items-center justify-between">
                <div>
                  <Link href={`/sessions/${r.sessionId}`} className="text-sm text-stone-700 hover:text-amber-700">
                    {new Date(r.session.date).toLocaleDateString("he-IL", {
                      weekday: "short", day: "numeric", month: "short",
                    })}
                  </Link>
                  <span className="text-xs text-stone-400 mr-2">{r.session.group.name}</span>
                </div>
                <span className={`text-xs px-2 py-0.5 rounded-full ${st.cls}`}>{st.label}</span>
              </div>
            );
          })}
          {student.registrations.length === 0 && (
            <div className="px-5 py-4 text-stone-400 text-sm">אין היסטוריה</div>
          )}
        </div>
      </div>
    </div>
  );
}
