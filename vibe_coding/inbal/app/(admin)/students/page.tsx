import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { redirect } from "next/navigation";
import { prisma } from "@/lib/prisma";
import Link from "next/link";
import AddStudentForm from "./AddStudentForm";

export default async function StudentsPage() {
  const session = await getServerSession(authOptions);
  if (!session || (session.user as { role?: string })?.role !== "ADMIN") redirect("/my");

  const students = await prisma.user.findMany({
    where: { role: "STUDENT" },
    include: {
      enrollments: {
        where: { status: "ACTIVE" },
        include: { group: { select: { name: true } } },
      },
      payments: { orderBy: { date: "desc" }, take: 1 },
    },
    orderBy: { name: "asc" },
  });

  const now = new Date();
  const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-stone-800">תלמידים ({students.length})</h1>
        <AddStudentForm />
      </div>

      {students.length === 0 ? (
        <div className="bg-white rounded-xl border border-stone-200 p-12 text-center">
          <div className="text-5xl mb-3">🎓</div>
          <div className="text-stone-500">אין תלמידים עדיין</div>
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
          <div className="divide-y divide-stone-100">
            {students.map((student) => {
              const lastPayment = student.payments[0];
              const paidThisMonth = lastPayment && new Date(lastPayment.date) >= monthStart;

              return (
                <Link
                  key={student.id}
                  href={`/students/${student.id}`}
                  className="flex items-center justify-between px-5 py-4 hover:bg-stone-50 transition"
                >
                  <div>
                    <div className="font-medium text-stone-800">{student.name}</div>
                    <div className="text-sm text-stone-400">{student.email}</div>
                    {student.enrollments.length > 0 && (
                      <div className="text-xs text-stone-400 mt-0.5">
                        {student.enrollments.map((e) => e.group.name).join(" • ")}
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-sm">
                    {student.phone && (
                      <span className="text-stone-400 text-xs">{student.phone}</span>
                    )}
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                      paidThisMonth
                        ? "bg-green-100 text-green-700"
                        : "bg-red-100 text-red-700"
                    }`}>
                      {paidThisMonth ? "שילם" : "לא שילם"}
                    </span>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
