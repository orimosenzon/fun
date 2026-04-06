const PHONE_NUMBER_ID = process.env.WHATSAPP_PHONE_NUMBER_ID;
const ACCESS_TOKEN = process.env.WHATSAPP_ACCESS_TOKEN;

async function sendTemplate(
  phone: string,
  templateName: string,
  parameters: { type: "text"; text: string }[]
): Promise<void> {
  if (!PHONE_NUMBER_ID || !ACCESS_TOKEN) {
    console.warn("WhatsApp env vars not set — message not sent");
    return;
  }

  const normalized = phone.replace(/\D/g, "").replace(/^0/, "972");
  const url = `https://graph.facebook.com/v21.0/${PHONE_NUMBER_ID}/messages`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${ACCESS_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messaging_product: "whatsapp",
      to: normalized,
      type: "template",
      template: {
        name: templateName,
        language: { code: "he" },
        components: [{ type: "body", parameters }],
      },
    }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(`WhatsApp send failed (${templateName}): ${JSON.stringify(err)}`);
  }
}

export function sendStandbyNotification(phone: string, name: string): Promise<void> {
  return sendTemplate(phone, "standby_notification", [{ type: "text", text: name }]);
}

// נשלח לענבל כשמקום מתפנה
// {{1}} = תיאור השיעור, {{2}} = רשימת המתנה, {{3}} = שם הראשון/ה בתור
export function sendAdminSpotOpened(
  phone: string,
  sessionLabel: string,
  waitlistText: string,
  firstPersonName: string
): Promise<void> {
  return sendTemplate(phone, "admin_spot_opened", [
    { type: "text", text: sessionLabel },
    { type: "text", text: waitlistText },
    { type: "text", text: firstPersonName },
  ]);
}

// נשלח לענבל כשמקום נתפס
// {{1}} = שם התלמיד/ה, {{2}} = תיאור השיעור
export function sendAdminSpotTaken(
  phone: string,
  studentName: string,
  sessionLabel: string
): Promise<void> {
  return sendTemplate(phone, "admin_spot_taken", [
    { type: "text", text: studentName },
    { type: "text", text: sessionLabel },
  ]);
}
