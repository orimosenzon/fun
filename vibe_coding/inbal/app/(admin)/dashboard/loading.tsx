export default function DashboardLoading() {
  return (
    <div className="animate-pulse" dir="rtl">
      <div className="h-8 w-48 bg-stone-200 rounded-lg mb-6" />

      <div className="grid grid-cols-3 gap-4 mb-8">
        {[0, 1, 2].map((i) => (
          <div key={i} className="bg-white rounded-xl p-5 border border-stone-200">
            <div className="h-9 w-20 bg-stone-200 rounded mb-2" />
            <div className="h-4 w-24 bg-stone-100 rounded" />
          </div>
        ))}
      </div>

      <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-stone-100">
          <div className="h-5 w-32 bg-stone-200 rounded" />
        </div>
        <div className="divide-y divide-stone-100">
          {[0, 1, 2, 3, 4].map((i) => (
            <div key={i} className="px-5 py-4 flex items-start justify-between">
              <div className="space-y-2">
                <div className="h-4 w-28 bg-stone-200 rounded" />
                <div className="h-3 w-40 bg-stone-100 rounded" />
              </div>
              <div className="h-6 w-16 bg-stone-100 rounded-full" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
