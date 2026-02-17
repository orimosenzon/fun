import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { createGuestSchema } from "@hotel/shared";
import type { CreateGuestInput } from "@hotel/shared";

interface GuestFormProps {
  onSubmit: (data: CreateGuestInput) => void;
  isLoading?: boolean;
}

export function GuestForm({ onSubmit, isLoading }: GuestFormProps) {
  const { register, handleSubmit, formState: { errors } } = useForm<CreateGuestInput>({
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    resolver: zodResolver(createGuestSchema) as any,
    defaultValues: { country: "IL" },
  });

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">שם פרטי</label>
          <input {...register("firstName")} className="w-full border rounded-lg px-3 py-2 text-sm" />
          {errors.firstName && <p className="text-red-500 text-xs mt-1">{errors.firstName.message}</p>}
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">שם משפחה</label>
          <input {...register("lastName")} className="w-full border rounded-lg px-3 py-2 text-sm" />
          {errors.lastName && <p className="text-red-500 text-xs mt-1">{errors.lastName.message}</p>}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">טלפון</label>
          <input {...register("phone")} className="w-full border rounded-lg px-3 py-2 text-sm" dir="ltr" />
          {errors.phone && <p className="text-red-500 text-xs mt-1">{errors.phone.message}</p>}
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">אימייל</label>
          <input {...register("email")} type="email" className="w-full border rounded-lg px-3 py-2 text-sm" dir="ltr" />
        </div>
      </div>
      <div>
        <label className="block text-sm font-medium text-slate-700 mb-1">ת.ז. / דרכון</label>
        <input {...register("idNumber")} className="w-full border rounded-lg px-3 py-2 text-sm" dir="ltr" />
      </div>
      <button type="submit" disabled={isLoading} className="w-full bg-blue-600 text-white py-2 rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50">
        {isLoading ? "שומר..." : "שמור אורח"}
      </button>
    </form>
  );
}
