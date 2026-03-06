import { describe, it, expect, vi, beforeEach } from "vitest";
import { prismaMock } from "./prisma.mock";

// mock שליחת SMS — לא נשלח כלום בבדיקות
vi.mock("@/lib/twilio", () => ({
  sendSms: vi.fn().mockResolvedValue(undefined),
}));

import { notifyNextStandby } from "@/lib/standby";
import { sendSms } from "@/lib/twilio";

// נתוני עזר
const SESSION_ID = "session-1";

const makeSession = (wheelRegs: number, noWheelRegs: number) => ({
  id: SESSION_ID,
  date: new Date("2026-04-01T12:00:00"),
  status: "SCHEDULED",
  group: { name: "חמישי 12:15" },
  registrations: [
    ...Array.from({ length: wheelRegs }, (_, i) => ({
      slotType: "WHEEL",
      status: "REGISTERED",
      userId: `wheel-user-${i}`,
    })),
    ...Array.from({ length: noWheelRegs }, (_, i) => ({
      slotType: "NO_WHEEL",
      status: "REGISTERED",
      userId: `nowheel-user-${i}`,
    })),
  ],
});

const makeEntry = (overrides = {}) => ({
  id: "entry-1",
  userId: "student-1",
  sessionId: SESSION_ID,
  slotType: null,
  notifiedAt: null,
  expiresAt: null,
  createdAt: new Date(),
  user: { name: "מיכל לוי", phone: "0521111111", email: "michal@example.com" },
  ...overrides,
});

describe("notifyNextStandby", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("לא עושה כלום אם אין רשומות standby", async () => {
    prismaMock.standbyEntry.findMany.mockResolvedValue([]);

    await notifyNextStandby(SESSION_ID);

    expect(prismaMock.standbyEntry.update).not.toHaveBeenCalled();
    expect(sendSms).not.toHaveBeenCalled();
  });

  it("לא עושה כלום אם השיעור לא קיים", async () => {
    prismaMock.standbyEntry.findMany.mockResolvedValue([makeEntry()]);
    prismaMock.session.findUnique.mockResolvedValue(null);

    await notifyNextStandby(SESSION_ID);

    expect(prismaMock.standbyEntry.update).not.toHaveBeenCalled();
  });

  it("לא שולח הודעה כשהשיעור מלא (4 WHEEL + 3 NO_WHEEL)", async () => {
    prismaMock.standbyEntry.findMany.mockResolvedValue([makeEntry()]);
    prismaMock.session.findUnique.mockResolvedValue(makeSession(4, 3) as any);

    await notifyNextStandby(SESSION_ID);

    expect(prismaMock.standbyEntry.update).not.toHaveBeenCalled();
    expect(sendSms).not.toHaveBeenCalled();
  });

  it("שולח הודעה לראשון בתור כשיש מקום", async () => {
    prismaMock.standbyEntry.findMany.mockResolvedValue([makeEntry()]);
    prismaMock.session.findUnique.mockResolvedValue(makeSession(0, 0) as any);
    prismaMock.standbyEntry.update.mockResolvedValue(makeEntry() as any);

    await notifyNextStandby(SESSION_ID);

    expect(prismaMock.standbyEntry.update).toHaveBeenCalledOnce();
    expect(sendSms).toHaveBeenCalledOnce();
    expect(sendSms).toHaveBeenCalledWith("0521111111", expect.stringContaining("מיכל לוי"));
  });

  it("מגדיר expiresAt = notifiedAt + 1 שעה", async () => {
    prismaMock.standbyEntry.findMany.mockResolvedValue([makeEntry()]);
    prismaMock.session.findUnique.mockResolvedValue(makeSession(0, 0) as any);
    prismaMock.standbyEntry.update.mockResolvedValue(makeEntry() as any);

    const before = Date.now();
    await notifyNextStandby(SESSION_ID);
    const after = Date.now();

    const call = prismaMock.standbyEntry.update.mock.calls[0][0];
    const { notifiedAt, expiresAt } = call.data as { notifiedAt: Date; expiresAt: Date };

    expect(notifiedAt.getTime()).toBeGreaterThanOrEqual(before);
    expect(notifiedAt.getTime()).toBeLessThanOrEqual(after);

    const diffHours = (expiresAt.getTime() - notifiedAt.getTime()) / (1000 * 60 * 60);
    expect(diffHours).toBeCloseTo(1, 5);
  });

  it("מדלג על תלמיד ש-slotType שלו לא פנוי — מודיע לבא אחריו", async () => {
    const wheelOnlyEntry = makeEntry({ id: "entry-1", slotType: "WHEEL" });
    const noWheelEntry = makeEntry({
      id: "entry-2",
      userId: "student-2",
      slotType: "NO_WHEEL",
      user: { name: "רחל גולד", phone: "0522222222", email: "rachel@example.com" },
    });

    prismaMock.standbyEntry.findMany.mockResolvedValue([wheelOnlyEntry, noWheelEntry]);
    // WHEEL מלא, NO_WHEEL פנוי
    prismaMock.session.findUnique.mockResolvedValue(makeSession(4, 0) as any);
    prismaMock.standbyEntry.update.mockResolvedValue(noWheelEntry as any);

    await notifyNextStandby(SESSION_ID);

    // אמור לקפוץ על wheelOnlyEntry ולהודיע ל-noWheelEntry
    const call = prismaMock.standbyEntry.update.mock.calls[0][0];
    expect(call.where.id).toBe("entry-2");
    expect(sendSms).toHaveBeenCalledWith("0522222222", expect.any(String));
  });

  it("לא שולח SMS אם אין מספר טלפון", async () => {
    const entryNoPhone = makeEntry({ user: { name: "ללא טלפון", phone: null, email: "x@x.com" } });
    prismaMock.standbyEntry.findMany.mockResolvedValue([entryNoPhone]);
    prismaMock.session.findUnique.mockResolvedValue(makeSession(0, 0) as any);
    prismaMock.standbyEntry.update.mockResolvedValue(entryNoPhone as any);

    await notifyNextStandby(SESSION_ID);

    expect(prismaMock.standbyEntry.update).toHaveBeenCalledOnce();
    expect(sendSms).not.toHaveBeenCalled();
  });
});
