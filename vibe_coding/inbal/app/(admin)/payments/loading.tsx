export default function PaymentsLoading() {
  return (
    <div className="space-y-6 animate-pulse" dir="rtl">
      <div className="flex items-center justify-between">
        <div className="h-8 w-28 bg-stone-200 rounded-lg" />
        <div className="h-4 w-24 bg-stone-100 rounded" />
      </div>

      <div className="grid grid-cols-3 gap-4">
        {[0, 1, 2].map((i) => (
          <div key={i} className="bg-white rounded-xl p-5 border border-stone-200">
            <div className="h-9 w-20 bg-stone-200 rounded mb-2" />
            <div className="h-4 w-24 bg-stone-100 rounded" />
          </div>
        ))}
      </div>

      <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-stone-100">
          <div className="h-5 w-36 bg-stone-200 rounded" />
        </div>
        <div className="divide-y divide-stone-100">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="flex items-center justify-between px-5 py-3">
              <div className="space-y-1.5">
                <div className="h-4 w-24 bg-stone-200 rounded" />
                <div className="h-3 w-32 bg-stone-100 rounded" />
              </div>
              <div className="space-y-1.5 text-left">
                <div className="h-4 w-14 bg-stone-200 rounded" />
                <div className="h-3 w-20 bg-stone-100 rounded" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
