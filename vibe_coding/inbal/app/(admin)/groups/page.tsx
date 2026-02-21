import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { redirect } from "next/navigation";
import { prisma } from "@/lib/prisma";
import Link from "next/link";

const DAYS = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

export default async function GroupsPage() {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role?: string })?.role !== "ADMIN") redirect("/my");

  const groups = await prisma.group.findMany({
    where: { isActive: true },
    include: {
      enrollments: { where: { status: "ACTIVE" } },
      sessions: {
        where: { date: { gte: new Date() }, status: "SCHEDULED" },
        orderBy: { date: "asc" },
        take: 1,
      },
    },
    orderBy: { dayOfWeek: "asc" },
  });

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-stone-800">קבוצות</h1>
        <Link
          href="/groups/new"
          className="bg-amber-600 hover:bg-amber-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition"
        >
          + קבוצה חדשה
        </Link>
      </div>

      {groups.length === 0 ? (
        <div className="bg-white rounded-xl border border-stone-200 p-12 text-center">
          <div className="text-5xl mb-3">👥</div>
          <div className="text-stone-500 mb-4">אין קבוצות עדיין</div>
          <Link href="/groups/new" className="text-amber-600 hover:underline text-sm">
            צרי קבוצה ראשונה
          </Link>
        </div>
      ) : (
        <div className="grid gap-4">
          {groups.map((group) => {
            const nextSession = group.sessions[0];
            return (
              <Link
                key={group.id}
                href={`/groups/${group.id}`}
                className="bg-white rounded-xl border border-stone-200 p-5 hover:border-amber-300 hover:shadow-sm transition block"
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="font-semibold text-stone-800 text-lg">{group.name}</h2>
                    <div className="text-stone-500 text-sm mt-1 flex items-center gap-3">
                      <span>📅 יום {DAYS[group.dayOfWeek]} {group.time}</span>
                      <span>⏱ {group.duration} דקות</span>
                      {group.location && <span>📍 {group.location}</span>}
                    </div>
                    {group.description && (
                      <div className="text-stone-400 text-sm mt-1">{group.description}</div>
                    )}
                  </div>
                  <div className="text-left text-sm">
                    <div className="font-semibold text-stone-700">
                      {group.enrollments.length}/{group.maxStudents}
                    </div>
                    <div className="text-stone-400 text-xs">תלמידים</div>
                  </div>
                </div>

                {nextSession && (
                  <div className="mt-3 pt-3 border-t border-stone-100 text-sm text-stone-500">
                    שיעור הבא:{" "}
                    {new Date(nextSession.date).toLocaleDateString("he-IL", {
                      weekday: "short",
                      day: "numeric",
                      month: "long",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </div>
                )}
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
