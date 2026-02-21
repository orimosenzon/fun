import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { redirect, notFound } from "next/navigation";
import { prisma } from "@/lib/prisma";
import Link from "next/link";
import AddStudentToGroup from "./AddStudentToGroup";
import GenerateSessionsButton from "./GenerateSessionsButton";

const DAYS = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

function statusBadge(status: string) {
  const map: Record<string, { label: string; cls: string }> = {
    REGISTERED: { label: "יבוא", cls: "bg-green-100 text-green-800" },
    CANCELLED: { label: "ביטל", cls: "bg-red-100 text-red-800" },
    TRANSFERRED: { label: "הועבר", cls: "bg-blue-100 text-blue-800" },
    ABSENT: { label: "נעדר", cls: "bg-stone-100 text-stone-600" },
    SCHEDULED: { label: "מתוכנן", cls: "bg-amber-100 text-amber-800" },
    COMPLETED: { label: "הסתיים", cls: "bg-stone-100 text-stone-600" },
    CANCELLED_SESSION: { label: "בוטל", cls: "bg-red-100 text-red-800" },
  };
  const s = map[status] || { label: status, cls: "bg-stone-100 text-stone-600" };
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${s.cls}`}>
      {s.label}
    </span>
  );
}

export default async function GroupPage({ params }: { params: Promise<{ id: string }> }) {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role?: string })?.role !== "ADMIN") redirect("/my");

  const { id } = await params;
  const group = await prisma.group.findUnique({
    where: { id },
    include: {
      enrollments: {
        include: { user: { select: { id: true, name: true, email: true, phone: true } } },
        orderBy: { startDate: "asc" },
      },
      sessions: {
        include: {
          registrations: {
            include: { user: { select: { id: true, name: true } } },
          },
        },
        orderBy: { date: "asc" },
        take: 20,
      },
    },
  });

  if (!group) notFound();

  const activeEnrollments = group.enrollments.filter((e) => e.status === "ACTIVE");
  const futureSessions = group.sessions.filter((s) => new Date(s.date) >= new Date());
  const pastSessions = group.sessions.filter((s) => new Date(s.date) < new Date());

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link href="/groups" className="text-sm text-stone-400 hover:text-stone-600 mb-1 block">
            ← קבוצות
          </Link>
          <h1 className="text-2xl font-bold text-stone-800">{group.name}</h1>
          <div className="text-stone-500 text-sm mt-1 flex items-center gap-3">
            <span>📅 יום {DAYS[group.dayOfWeek]} בשעה {group.time}</span>
            <span>⏱ {group.duration} דקות</span>
            {group.location && <span>📍 {group.location}</span>}
          </div>
        </div>
        <GenerateSessionsButton groupId={id} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Students */}
        <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
          <div className="px-5 py-4 border-b border-stone-100 flex items-center justify-between">
            <h2 className="font-semibold text-stone-800">
              תלמידים ({activeEnrollments.length}/{group.maxStudents})
            </h2>
          </div>

          <div className="divide-y divide-stone-100">
            {activeEnrollments.map((e) => (
              <div key={e.id} className="px-5 py-3 flex items-center justify-between">
                <div>
                  <Link href={`/students/${e.user.id}`} className="font-medium text-stone-800 hover:text-amber-700">
                    {e.user.name}
                  </Link>
                  {e.user.phone && (
                    <div className="text-xs text-stone-400">{e.user.phone}</div>
                  )}
                </div>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  e.status === "ACTIVE" ? "bg-green-100 text-green-700" :
                  e.status === "PAUSED" ? "bg-yellow-100 text-yellow-700" :
                  "bg-stone-100 text-stone-500"
                }`}>
                  {e.status === "ACTIVE" ? "פעיל" : e.status === "PAUSED" ? "מושהה" : "עזב"}
                </span>
              </div>
            ))}

            {activeEnrollments.length === 0 && (
              <div className="p-6 text-center text-stone-400 text-sm">אין תלמידים רשומים</div>
            )}
          </div>

          <div className="px-5 py-3 border-t border-stone-100">
            <AddStudentToGroup groupId={id} />
          </div>
        </div>

        {/* Upcoming Sessions */}
        <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
          <div className="px-5 py-4 border-b border-stone-100">
            <h2 className="font-semibold text-stone-800">שיעורים קרובים ({futureSessions.length})</h2>
          </div>

          <div className="divide-y divide-stone-100 max-h-96 overflow-y-auto">
            {futureSessions.map((s) => {
              const registered = s.registrations.filter((r) => r.status === "REGISTERED");
              const cancelled = s.registrations.filter((r) => r.status === "CANCELLED" || r.status === "TRANSFERRED");
              return (
                <Link
                  key={s.id}
                  href={`/sessions/${s.id}`}
                  className="block px-5 py-3 hover:bg-stone-50 transition"
                >
                  <div className="flex items-center justify-between">
                    <div className="text-sm font-medium text-stone-700">
                      {new Date(s.date).toLocaleDateString("he-IL", {
                        weekday: "short",
                        day: "numeric",
                        month: "short",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </div>
                    <div className="flex gap-1">
                      <span className="bg-green-100 text-green-800 px-2 py-0.5 rounded-full text-xs">
                        {registered.length} ✓
                      </span>
                      {cancelled.length > 0 && (
                        <span className="bg-red-100 text-red-800 px-2 py-0.5 rounded-full text-xs">
                          {cancelled.length} ✗
                        </span>
                      )}
                    </div>
                  </div>
                </Link>
              );
            })}

            {futureSessions.length === 0 && (
              <div className="p-6 text-center text-stone-400 text-sm">
                אין שיעורים מתוכננים. לחצי על &quot;הוסף שיעורים&quot;
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
