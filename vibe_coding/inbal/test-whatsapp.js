/**
 * WhatsApp test script — sends hello_world template to a phone number.
 * Usage: WHATSAPP_ACCESS_TOKEN=<token> node test-whatsapp.js
 */

const PHONE_NUMBER_ID = "968039886402071";
const ACCESS_TOKEN = process.env.WHATSAPP_ACCESS_TOKEN;
const TO = "972508820997";

if (!ACCESS_TOKEN) {
  console.error("Missing WHATSAPP_ACCESS_TOKEN environment variable");
  process.exit(1);
}

async function sendTestMessage() {
  const url = `https://graph.facebook.com/v21.0/${PHONE_NUMBER_ID}/messages`;

  const body = {
    messaging_product: "whatsapp",
    to: TO,
    type: "template",
    template: {
      name: "hello_world",
      language: { code: "en_US" },
    },
  };

  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${ACCESS_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const data = await res.json();
  if (res.ok) {
    console.log("✓ Message sent!", JSON.stringify(data, null, 2));
  } else {
    console.error("✗ Error:", JSON.stringify(data, null, 2));
  }
}

sendTestMessage();
