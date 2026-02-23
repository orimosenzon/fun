export const WHEEL_SLOTS = 4;
export const NO_WHEEL_SLOTS = 3;
export const TOTAL_SLOTS = 7;

export function countSlots(registrations: { slotType: string | null; status: string }[]) {
  const active = registrations.filter((r) => r.status === "REGISTERED");
  return {
    wheel: active.filter((r) => r.slotType === "WHEEL").length,
    noWheel: active.filter((r) => r.slotType === "NO_WHEEL").length,
  };
}

export function slotAvailable(
  registrations: { slotType: string | null; status: string }[],
  slotType: "WHEEL" | "NO_WHEEL"
): boolean {
  const counts = countSlots(registrations);
  if (slotType === "WHEEL") return counts.wheel < WHEEL_SLOTS;
  return counts.noWheel < NO_WHEEL_SLOTS;
}
