import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { redirect } from "next/navigation";
import { prisma } from "@/lib/prisma";
import MySessionActions from "./MySessionActions";

function statusBadge(status: string) {
  const map: Record<string, { label: string; cls: string }> = {
    REGISTERED: { label: "רשומה", cls: "bg-green-100 text-green-700" },
    CANCELLED: { label: "ביטלת", cls: "bg-red-100 text-red-700" },
    TRANSFERRED: { label: "הועברת", cls: "bg-blue-100 text-blue-700" },
  };
  return map[status] || { label: status, cls: "bg-stone-100 text-stone-600" };
}

export default async function MyPage() {
  const session = await getServerSession(authOptions);
  if (!session) redirect("/login");

  const user = await prisma.user.findUnique({
    where: { email: session.user!.email! },
    include: {
      enrollments: {
        where: { status: "ACTIVE" },
        include: { group: { select: { name: true, dayOfWeek: true, time: true } } },
      },
      payments: { orderBy: { date: "desc" }, take: 3 },
    },
  });

  if (!user) redirect("/login");

  // Upcoming registrations
  const registrations = await prisma.sessionRegistration.findMany({
    where: {
      userId: user.id,
      session: { date: { gte: new Date() }, status: "SCHEDULED" },
    },
    include: {
      session: { include: { group: { select: { name: true, location: true } } } },
    },
    orderBy: { session: { date: "asc" } },
    take: 10,
  });

  const now = new Date();
  const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);
  const paidThisMonth = user.payments.some((p) => new Date(p.date) >= monthStart);

  const DAYS = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-stone-800">שלום {user.name} 👋</h1>
        <p className="text-stone-500 text-sm mt-1">הקבוצות שלך</p>
      </div>

      {/* Groups */}
      <div className="flex flex-wrap gap-2">
        {user.enrollments.map((e) => (
          <div
            key={e.id}
            className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm"
          >
            <div className="font-semibold text-amber-800">{e.group.name}</div>
            <div className="text-stone-500">
              יום {DAYS[e.group.dayOfWeek]} • {e.group.time}
            </div>
          </div>
        ))}
        {user.enrollments.length === 0 && (
          <div className="text-stone-400 text-sm">לא רשומה לקבוצות עדיין</div>
        )}
      </div>

      {/* Payment status */}
      <div className={`rounded-xl p-4 border ${
        paidThisMonth
          ? "bg-green-50 border-green-200"
          : "bg-red-50 border-red-200"
      }`}>
        <div className="flex items-center gap-2">
          <span className="text-xl">{paidThisMonth ? "✅" : "⚠️"}</span>
          <div>
            <div className={`font-semibold ${paidThisMonth ? "text-green-800" : "text-red-800"}`}>
              {paidThisMonth ? "שילמת החודש" : "טרם שילמת החודש"}
            </div>
            {!paidThisMonth && (
              <div className="text-sm text-red-600">פנה/י למורה לתשלום</div>
            )}
          </div>
        </div>
      </div>

      {/* Upcoming sessions */}
      <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-stone-100">
          <h2 className="font-semibold text-stone-800">השיעורים הקרובים שלי</h2>
        </div>

        <div className="divide-y divide-stone-100">
          {registrations.map((reg) => {
            const { label, cls } = statusBadge(reg.status);
            const sessionDate = new Date(reg.session.date);
            return (
              <div key={reg.id} className="px-5 py-4">
                <div className="flex items-start justify-between">
                  <div>
                    <div className="font-medium text-stone-800">{reg.session.group.name}</div>
                    <div className="text-sm text-stone-500 mt-0.5">
                      {sessionDate.toLocaleDateString("he-IL", {
                        weekday: "long",
                        day: "numeric",
                        month: "long",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </div>
                    {reg.session.group.location && (
                      <div className="text-xs text-stone-400 mt-0.5">
                        📍 {reg.session.group.location}
                      </div>
                    )}
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${cls}`}>
                      {label}
                    </span>
                    {reg.status === "REGISTERED" && (
                      <MySessionActions
                        registrationId={reg.id}
                        sessionId={reg.sessionId}
                        userId={user.id}
                      />
                    )}
                  </div>
                </div>
              </div>
            );
          })}

          {registrations.length === 0 && (
            <div className="p-8 text-center text-stone-400">
              <div className="text-4xl mb-2">📅</div>
              <div>אין שיעורים קרובים</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
