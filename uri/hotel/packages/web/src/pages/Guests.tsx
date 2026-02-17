import { useGuests } from "@/hooks/useGuests";

export function Guests() {
  const { data: guests, isLoading } = useGuests();

  if (isLoading) return <div className="text-center py-10 text-slate-500">טוען...</div>;

  return (
    <div>
      <h2 className="text-2xl font-bold text-slate-800 mb-6">אורחים</h2>

      <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-50">
            <tr>
              <th className="text-right px-4 py-3 font-medium text-slate-600">שם</th>
              <th className="text-right px-4 py-3 font-medium text-slate-600">טלפון</th>
              <th className="text-right px-4 py-3 font-medium text-slate-600">אימייל</th>
              <th className="text-right px-4 py-3 font-medium text-slate-600">מדינה</th>
            </tr>
          </thead>
          <tbody>
            {guests?.map((g) => (
              <tr key={g.id} className="border-t hover:bg-slate-50">
                <td className="px-4 py-3 font-medium">{g.firstName} {g.lastName}</td>
                <td className="px-4 py-3" dir="ltr">{g.phone}</td>
                <td className="px-4 py-3" dir="ltr">{g.email ?? "-"}</td>
                <td className="px-4 py-3">{g.country}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {guests?.length === 0 && <p className="text-center text-slate-400 py-8">אין אורחים</p>}
      </div>
    </div>
  );
}
