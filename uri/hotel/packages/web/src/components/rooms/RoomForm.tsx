import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { createRoomSchema, ROOM_TYPES, ROOM_TYPE_LABELS_HE } from "@hotel/shared";
import type { CreateRoomInput } from "@hotel/shared";

interface RoomFormProps {
  onSubmit: (data: CreateRoomInput) => void;
  onCancel: () => void;
  isLoading?: boolean;
}

export function RoomForm({ onSubmit, onCancel, isLoading }: RoomFormProps) {
  const { register, handleSubmit, formState: { errors } } = useForm<CreateRoomInput>({
    resolver: zodResolver(createRoomSchema),
    defaultValues: { maxGuests: 2, amenities: [] },
  });

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">מספר חדר</label>
          <input {...register("number")} className="w-full border rounded-lg px-3 py-2 text-sm" placeholder="101" />
          {errors.number && <p className="text-red-500 text-xs mt-1">{errors.number.message}</p>}
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">קומה</label>
          <input {...register("floor", { valueAsNumber: true })} type="number" className="w-full border rounded-lg px-3 py-2 text-sm" />
          {errors.floor && <p className="text-red-500 text-xs mt-1">{errors.floor.message}</p>}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">סוג חדר</label>
          <select {...register("type")} className="w-full border rounded-lg px-3 py-2 text-sm">
            {ROOM_TYPES.map((t) => (
              <option key={t} value={t}>{ROOM_TYPE_LABELS_HE[t]}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">מחיר ללילה (₪)</label>
          <input {...register("basePrice", { valueAsNumber: true })} type="number" className="w-full border rounded-lg px-3 py-2 text-sm" />
          {errors.basePrice && <p className="text-red-500 text-xs mt-1">{errors.basePrice.message}</p>}
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-slate-700 mb-1">מקסימום אורחים</label>
        <input {...register("maxGuests", { valueAsNumber: true })} type="number" className="w-full border rounded-lg px-3 py-2 text-sm" />
      </div>

      <div>
        <label className="block text-sm font-medium text-slate-700 mb-1">הערות</label>
        <textarea {...register("notes")} className="w-full border rounded-lg px-3 py-2 text-sm" rows={2} />
      </div>

      <div className="flex gap-3 justify-end pt-2">
        <button type="button" onClick={onCancel} className="px-4 py-2 text-sm border rounded-lg text-slate-600 hover:bg-slate-50">
          ביטול
        </button>
        <button type="submit" disabled={isLoading} className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
          {isLoading ? "שומר..." : "שמור"}
        </button>
      </div>
    </form>
  );
}
