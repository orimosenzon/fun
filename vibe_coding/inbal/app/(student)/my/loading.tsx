const DAYS_HE = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

export default function MyLoading() {
  return (
    <div className="space-y-3 animate-pulse" dir="rtl">
      <div className="flex items-center justify-between">
        <div className="h-7 w-28 bg-stone-200 rounded-lg" />
        <div className="h-9 w-52 bg-stone-200 rounded-xl" />
      </div>

      <div className="overflow-x-auto rounded-xl border border-stone-200 bg-white shadow-sm">
        <div className="min-w-[560px]">
          {/* Day headers */}
          <div className="grid grid-cols-7 border-b border-stone-200">
            {DAYS_HE.map((d, i) => (
              <div key={i} className="px-2 py-3 text-center border-r border-stone-100 last:border-r-0">
                <div className="h-3 w-8 bg-stone-200 rounded mx-auto mb-2" />
                <div className="h-7 w-7 bg-stone-200 rounded-full mx-auto" />
              </div>
            ))}
          </div>

          {/* Session cells */}
          <div className="grid grid-cols-7 min-h-[160px] p-1.5 gap-1.5">
            {[2, 4].map((col) =>
              Array.from({ length: 7 }, (_, i) =>
                i === col ? (
                  <div key={i} className="rounded-lg border border-stone-200 p-2 space-y-2">
                    <div className="h-3 w-full bg-stone-200 rounded" />
                    <div className="h-2 w-10 bg-stone-100 rounded" />
                    <div className="flex gap-0.5">
                      {Array.from({ length: 4 }).map((_, j) => (
                        <div key={j} className="w-4 h-4 rounded-full bg-stone-200" />
                      ))}
                    </div>
                    <div className="flex gap-0.5">
                      {Array.from({ length: 3 }).map((_, j) => (
                        <div key={j} className="w-4 h-4 rounded-full bg-stone-200" />
                      ))}
                    </div>
                  </div>
                ) : (
                  <div key={i} />
                )
              )
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
