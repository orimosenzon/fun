export default function StudentsLoading() {
  return (
    <div className="animate-pulse" dir="rtl">
      <div className="flex items-center justify-between mb-6">
        <div className="h-8 w-36 bg-stone-200 rounded-lg" />
        <div className="h-9 w-28 bg-stone-200 rounded-lg" />
      </div>

      <div className="bg-white rounded-xl border border-stone-200 overflow-hidden">
        <div className="divide-y divide-stone-100">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="flex items-center justify-between px-5 py-4">
              <div className="space-y-2">
                <div className="h-4 w-24 bg-stone-200 rounded" />
                <div className="h-3 w-36 bg-stone-100 rounded" />
                <div className="h-3 w-28 bg-stone-100 rounded" />
              </div>
              <div className="h-6 w-14 bg-stone-100 rounded-full" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
