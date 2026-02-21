import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { redirect, notFound } from "next/navigation";
import { prisma } from "@/lib/prisma";
import Link from "next/link";
import RegistrationActions from "./RegistrationActions";

function statusLabel(status: string) {
  const map: Record<string, { label: string; cls: string }> = {
    REGISTERED: { label: "יבוא ✓", cls: "text-green-700 bg-green-50" },
    CANCELLED: { label: "ביטל ✗", cls: "text-red-700 bg-red-50" },
    TRANSFERRED: { label: "הועבר →", cls: "text-blue-700 bg-blue-50" },
    ABSENT: { label: "נעדר", cls: "text-stone-500 bg-stone-50" },
  };
  return map[status] || { label: status, cls: "text-stone-500 bg-stone-50" };
}

export default async function SessionPage({ params }: { params: Promise<{ id: string }> }) {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role?: string })?.role !== "ADMIN") redirect("/my");

  const { id } = await params;
  const s = await prisma.session.findUnique({
    where: { id },
    include: {
      group: true,
      registrations: {
        include: { user: { select: { id: true, name: true, phone: true } } },
        orderBy: { createdAt: "asc" },
      },
    },
  });

  if (!s) notFound();

  const registered = s.registrations.filter((r) => r.status === "REGISTERED");
  const cancelled = s.registrations.filter((r) => r.status === "CANCELLED");
  const transferred = s.registrations.filter((r) => r.status === "TRANSFERRED");
  const absent = s.registrations.filter((r) => r.status === "ABSENT");

  const sessionDate = new Date(s.date);
  const isPast = sessionDate < new Date();

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <Link href={`/groups/${s.groupId}`} className="text-sm text-stone-400 hover:text-stone-600 mb-1 block">
          ← {s.group.name}
        </Link>
        <h1 className="text-2xl font-bold text-stone-800">
          {sessionDate.toLocaleDateString("he-IL", {
            weekday: "long",
            day: "numeric",
            month: "long",
            year: "numeric",
          })}
        </h1>
        <div className="text-stone-500 text-sm mt-1">
          {s.group.time} • {s.group.location}
          {isPast && (
            <span className="mr-2 bg-stone-100 text-stone-600 px-2 py-0.5 rounded-full text-xs">עבר</span>
          )}
        </div>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: "יבואו", count: registered.length, cls: "border-green-200 bg-green-50" },
          { label: "ביטלו", count: cancelled.length, cls: "border-red-200 bg-red-50" },
          { label: "הועברו", count: transferred.length, cls: "border-blue-200 bg-blue-50" },
          { label: "נעדרו", count: absent.length, cls: "border-stone-200 bg-stone-50" },
        ].map((item) => (
          <div key={item.label} className={`rounded-xl border p-3 text-center ${item.cls}`}>
            <div className="text-2xl font-bold text-stone-800">{item.count}</div>
            <div className="text-xs text-stone-500">{item.label}</div>
          </div>
        ))}
      </div>

      {/* Registration list */}
      <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-stone-100">
          <h2 className="font-semibold text-stone-800">רשימת נוכחות ({s.registrations.length})</h2>
        </div>

        <div className="divide-y divide-stone-100">
          {s.registrations.map((reg) => {
            const { label, cls } = statusLabel(reg.status);
            return (
              <div key={reg.id} className="px-5 py-3 flex items-center justify-between">
                <div>
                  <Link href={`/students/${reg.user.id}`} className="font-medium text-stone-800 hover:text-amber-700">
                    {reg.user.name}
                  </Link>
                  {reg.user.phone && (
                    <div className="text-xs text-stone-400">{reg.user.phone}</div>
                  )}
                  {reg.note && (
                    <div className="text-xs text-stone-400 mt-0.5">📝 {reg.note}</div>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-xs px-2 py-1 rounded-full font-medium ${cls}`}>
                    {label}
                  </span>
                  <RegistrationActions
                    registrationId={reg.id}
                    currentStatus={reg.status}
                    userId={reg.user.id}
                    sessionId={s.id}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
