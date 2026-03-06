import { describe, it, expect } from "vitest";
import { slotAvailable, countSlots, WHEEL_SLOTS, NO_WHEEL_SLOTS } from "@/lib/slots";

type Reg = { slotType: string | null; status: string };

const registered = (slotType: string): Reg => ({ slotType, status: "REGISTERED" });
const cancelled = (slotType: string): Reg => ({ slotType, status: "CANCELLED" });

describe("countSlots", () => {
  it("מחזיר אפס כשאין רישומים", () => {
    expect(countSlots([])).toEqual({ wheel: 0, noWheel: 0 });
  });

  it("סופר רק רישומים פעילים (לא מבוטלים)", () => {
    const regs = [registered("WHEEL"), cancelled("WHEEL"), registered("NO_WHEEL")];
    expect(countSlots(regs)).toEqual({ wheel: 1, noWheel: 1 });
  });

  it("סופר נכון כמה מכל סוג", () => {
    const regs = [
      registered("WHEEL"),
      registered("WHEEL"),
      registered("NO_WHEEL"),
    ];
    expect(countSlots(regs)).toEqual({ wheel: 2, noWheel: 1 });
  });
});

describe("slotAvailable", () => {
  it("מקום פנוי כשהשיעור ריק", () => {
    expect(slotAvailable([], "WHEEL")).toBe(true);
    expect(slotAvailable([], "NO_WHEEL")).toBe(true);
  });

  it("מקום תפוס כשסוג WHEEL מלא", () => {
    const regs = Array.from({ length: WHEEL_SLOTS }, () => registered("WHEEL"));
    expect(slotAvailable(regs, "WHEEL")).toBe(false);
  });

  it("מקום תפוס כשסוג NO_WHEEL מלא", () => {
    const regs = Array.from({ length: NO_WHEEL_SLOTS }, () => registered("NO_WHEEL"));
    expect(slotAvailable(regs, "NO_WHEEL")).toBe(false);
  });

  it("WHEEL מלא אבל NO_WHEEL פנוי — NO_WHEEL זמין", () => {
    const regs = Array.from({ length: WHEEL_SLOTS }, () => registered("WHEEL"));
    expect(slotAvailable(regs, "WHEEL")).toBe(false);
    expect(slotAvailable(regs, "NO_WHEEL")).toBe(true);
  });

  it("ביטול לא נספר — מקום חוזר להיות פנוי", () => {
    const regs = [
      ...Array.from({ length: WHEEL_SLOTS - 1 }, () => registered("WHEEL")),
      cancelled("WHEEL"), // ביטול לא אמור לסגור מקום
    ];
    expect(slotAvailable(regs, "WHEEL")).toBe(true);
  });
});
