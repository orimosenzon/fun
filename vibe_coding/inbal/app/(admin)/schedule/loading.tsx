const DAYS_HE = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"];

export default function ScheduleLoading() {
  return (
    <div className="space-y-3 animate-pulse" dir="rtl">
      <div className="flex items-center justify-between">
        <div className="h-8 w-36 bg-stone-200 rounded-lg" />
        <div className="h-9 w-52 bg-stone-200 rounded-xl" />
      </div>

      <div className="overflow-x-auto rounded-xl border border-stone-200 bg-white shadow-sm">
        <div className="min-w-[600px]">
          <div className="grid grid-cols-7 border-b border-stone-200">
            {DAYS_HE.map((_, i) => (
              <div key={i} className="px-2 py-3 text-center border-r border-stone-100 last:border-r-0">
                <div className="h-3 w-8 bg-stone-200 rounded mx-auto mb-2" />
                <div className="h-7 w-7 bg-stone-200 rounded-full mx-auto" />
              </div>
            ))}
          </div>

          <div className="grid grid-cols-7 min-h-[200px] p-1.5 gap-1.5">
            {[1, 3, 5].map((col) =>
              Array.from({ length: 7 }, (_, i) =>
                i === col ? (
                  <div key={i} className="rounded-lg border border-stone-200 p-2 space-y-1.5">
                    <div className="bg-stone-100 rounded p-1.5 space-y-1">
                      <div className="h-3 w-full bg-stone-200 rounded" />
                      <div className="h-2 w-12 bg-stone-200 rounded" />
                    </div>
                    <div className="grid grid-cols-2 gap-1.5">
                      {[0, 1].map((col) => (
                        <div key={col} className="space-y-0.5">
                          <div className="h-2 w-8 bg-stone-200 rounded" />
                          {Array.from({ length: 3 }).map((_, j) => (
                            <div key={j} className="h-4 w-full bg-stone-100 rounded" />
                          ))}
                        </div>
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
