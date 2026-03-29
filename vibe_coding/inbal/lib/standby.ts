import { prisma } from "@/lib/prisma";
import { sendStandbyNotification } from "@/lib/whatsapp";
import { slotAvailable } from "@/lib/slots";

const STANDBY_EXPIRY_HOURS = 1;

/**
 * אחרי שמקום מתפנה בשיעור — מודיע לתלמיד הראשון בתור שלא קיבל הודעה עדיין.
 * אם כולם כבר קיבלו הודעה (ועדיין לא נרשמו), לא שולח שוב.
 */
export async function notifyNextStandby(sessionId: string): Promise<void> {
  // מצא את כל הסטנד ביי לשיעור, מסודר לפי זמן הצטרפות
  const entries = await prisma.standbyEntry.findMany({
    where: { sessionId, notifiedAt: null },
    include: { user: { select: { name: true, phone: true, email: true } } },
    orderBy: { createdAt: "asc" },
  });

  if (entries.length === 0) return;

  // בדוק שאכן יש מקום פנוי (לפעמים מתפנה ומתמלא מהר)
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    include: {
      registrations: { select: { slotType: true, status: true } },
      group: { select: { name: true } },
    },
  });
  if (!session) return;

  // מצא רשומה שהסלוט המבוקש שלה פנוי
  const candidate = entries.find((entry) => {
    if (!entry.slotType) {
      // כל סוג — בדוק שיש מקום כלשהו
      return (
        slotAvailable(session.registrations, "WHEEL") ||
        slotAvailable(session.registrations, "NO_WHEEL")
      );
    }
    return slotAvailable(session.registrations, entry.slotType as "WHEEL" | "NO_WHEEL");
  });

  if (!candidate) return;

  const now = new Date();
  const expiresAt = new Date(now.getTime() + STANDBY_EXPIRY_HOURS * 60 * 60 * 1000);

  await prisma.standbyEntry.update({
    where: { id: candidate.id },
    data: { notifiedAt: now, expiresAt },
  });

  // שלח WhatsApp אם יש מספר טלפון
  if (candidate.user.phone) {
    await sendStandbyNotification(candidate.user.phone).catch((err) =>
      console.error("WhatsApp send failed:", err)
    );
  }
}
