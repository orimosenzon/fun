/**
 * WhatsApp test script — sends hello_world template to a phone number.
 * Usage: node test-whatsapp.js
 */

const PHONE_NUMBER_ID = "968039886402071";
const ACCESS_TOKEN = "EAALGBGGBAksBRPAQaA3rDP1KfWw4HvQVzxb5OXp8rkrRvFqZBbz59SkVkucGeCR4HuVyZCFNxj0Mc1VgK3hbGnM2z78OBcJZALC1Lbc1wqBS8sr6kQkUveDEsIrJoEmXMJB3VcVwZCov3fZA6q5V2PVmW1xR7mawQ7U1D3CVy1S2fAcFJZBlxmcZBFQs9pXAkgft6EeapKKeamdxZA1FsEFIHNkqyc2goRPl5telr7ylOxTBpOD4yXN38xJSr1zLd1z6xFWYlWlSzNButmVfMWY0Hj3DwqcBUNoZD";
const TO = "972508820997";

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
