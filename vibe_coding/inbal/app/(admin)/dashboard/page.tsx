import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { redirect } from "next/navigation";
import { prisma } from "@/lib/prisma";
import Link from "next/link";

const DAYS = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

function formatDate(date: Date) {
  return new Date(date).toLocaleDateString("he-IL", {
    weekday: "short",
    day: "numeric",
    month: "long",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function statusBadge(status: string) {
  const map: Record<string, { label: string; cls: string }> = {
    REGISTERED: { label: "יבוא", cls: "bg-green-100 text-green-800" },
    CANCELLED: { label: "ביטל", cls: "bg-red-100 text-red-800" },
    TRANSFERRED: { label: "הועבר", cls: "bg-blue-100 text-blue-800" },
    ABSENT: { label: "נעדר", cls: "bg-stone-100 text-stone-600" },
  };
  const s = map[status] || { label: status, cls: "bg-stone-100 text-stone-600" };
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${s.cls}`}>
      {s.label}
    </span>
  );
}

export default async function DashboardPage() {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role?: string })?.role !== "ADMIN") redirect("/my");

  const now = new Date();

  // Next 5 upcoming sessions across all groups
  const upcomingSessions = await prisma.session.findMany({
    where: { date: { gte: now }, status: "SCHEDULED" },
    include: {
      group: { select: { name: true } },
      registrations: {
        include: { user: { select: { name: true } } },
      },
    },
    orderBy: { date: "asc" },
    take: 5,
  });

  // Stats
  const totalStudents = await prisma.user.count({ where: { role: "STUDENT" } });
  const activeGroups = await prisma.group.count({ where: { isActive: true } });

  // This month's revenue
  const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);
  const payments = await prisma.payment.aggregate({
    where: { date: { gte: monthStart } },
    _sum: { amount: true },
  });

  return (
    <div>
      <h1 className="text-2xl font-bold text-stone-800 mb-6">שלום {session.user?.name} 👋</h1>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <div className="bg-white rounded-xl p-5 border border-stone-200">
          <div className="text-3xl font-bold text-amber-600">{totalStudents}</div>
          <div className="text-sm text-stone-500 mt-1">תלמידים פעילים</div>
        </div>
        <div className="bg-white rounded-xl p-5 border border-stone-200">
          <div className="text-3xl font-bold text-amber-600">{activeGroups}</div>
          <div className="text-sm text-stone-500 mt-1">קבוצות פעילות</div>
        </div>
        <div className="bg-white rounded-xl p-5 border border-stone-200">
          <div className="text-3xl font-bold text-amber-600">
            ₪{payments._sum.amount?.toFixed(0) ?? 0}
          </div>
          <div className="text-sm text-stone-500 mt-1">הכנסות החודש</div>
        </div>
      </div>

      {/* Upcoming Sessions */}
      <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-stone-100 flex items-center justify-between">
          <h2 className="font-semibold text-stone-800">שיעורים קרובים</h2>
          <Link href="/groups" className="text-sm text-amber-600 hover:underline">
            כל הקבוצות ←
          </Link>
        </div>

        {upcomingSessions.length === 0 ? (
          <div className="p-8 text-center text-stone-400">
            <div className="text-4xl mb-2">📅</div>
            <div>אין שיעורים מתוכננים</div>
          </div>
        ) : (
          <div className="divide-y divide-stone-100">
            {upcomingSessions.map((session) => {
              const registered = session.registrations.filter((r) => r.status === "REGISTERED");
              const cancelled = session.registrations.filter((r) => r.status === "CANCELLED");
              const transferred = session.registrations.filter((r) => r.status === "TRANSFERRED");

              return (
                <Link
                  key={session.id}
                  href={`/sessions/${session.id}`}
                  className="block px-5 py-4 hover:bg-stone-50 transition"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="font-medium text-stone-800">{session.group.name}</div>
                      <div className="text-sm text-stone-500 mt-0.5">
                        {formatDate(session.date)}
                      </div>
                    </div>
                    <div className="flex gap-2 text-sm">
                      <span className="bg-green-100 text-green-800 px-2 py-0.5 rounded-full text-xs font-medium">
                        {registered.length} יבואו
                      </span>
                      {cancelled.length > 0 && (
                        <span className="bg-red-100 text-red-800 px-2 py-0.5 rounded-full text-xs font-medium">
                          {cancelled.length} ביטלו
                        </span>
                      )}
                      {transferred.length > 0 && (
                        <span className="bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full text-xs font-medium">
                          {transferred.length} הועברו
                        </span>
                      )}
                    </div>
                  </div>

                  {registered.length > 0 && (
                    <div className="mt-2 text-sm text-stone-600">
                      {registered.map((r) => r.user.name).join("، ")}
                    </div>
                  )}
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
