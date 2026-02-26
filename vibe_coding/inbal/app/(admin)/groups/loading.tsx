export default function GroupsLoading() {
  return (
    <div className="animate-pulse" dir="rtl">
      <div className="flex items-center justify-between mb-6">
        <div className="h-8 w-28 bg-stone-200 rounded-lg" />
        <div className="h-9 w-32 bg-stone-200 rounded-lg" />
      </div>

      <div className="grid gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="bg-white rounded-xl border border-stone-200 p-5">
            <div className="flex items-start justify-between">
              <div className="space-y-2">
                <div className="h-5 w-32 bg-stone-200 rounded" />
                <div className="h-3 w-52 bg-stone-100 rounded" />
              </div>
              <div className="space-y-1 text-left">
                <div className="h-5 w-8 bg-stone-200 rounded" />
                <div className="h-3 w-14 bg-stone-100 rounded" />
              </div>
            </div>
            <div className="mt-3 pt-3 border-t border-stone-100">
              <div className="h-3 w-44 bg-stone-100 rounded" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
