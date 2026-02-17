import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

async function main() {
  // Create rooms - 30 rooms across 3 floors
  const roomsData = [];

  const types = ["SINGLE", "DOUBLE", "TWIN", "SUITE", "FAMILY"] as const;
  const prices = { SINGLE: 350, DOUBLE: 500, TWIN: 500, SUITE: 900, FAMILY: 750 };
  const maxGuestsMap = { SINGLE: 1, DOUBLE: 2, TWIN: 2, SUITE: 3, FAMILY: 5 };

  for (let floor = 1; floor <= 3; floor++) {
    for (let room = 1; room <= 10; room++) {
      const number = `${floor}${String(room).padStart(2, "0")}`;
      let type: (typeof types)[number];
      if (room <= 3) type = "SINGLE";
      else if (room <= 6) type = "DOUBLE";
      else if (room <= 8) type = "TWIN";
      else if (room === 9) type = "SUITE";
      else type = "FAMILY";

      roomsData.push({
        number,
        floor,
        type,
        basePrice: prices[type],
        maxGuests: maxGuestsMap[type],
        amenities: JSON.stringify(["wifi", "ac", "tv"]),
      });
    }
  }

  for (const room of roomsData) {
    await prisma.room.upsert({
      where: { number: room.number },
      update: {},
      create: room,
    });
  }

  // Create sample guests
  const guests = [
    { firstName: "ישראל", lastName: "ישראלי", phone: "050-1234567", email: "israel@example.com", country: "IL" },
    { firstName: "שרה", lastName: "כהן", phone: "052-9876543", email: "sarah@example.com", country: "IL" },
    { firstName: "John", lastName: "Smith", phone: "+1-555-0123", email: "john@example.com", country: "US" },
    { firstName: "דוד", lastName: "לוי", phone: "054-5551234", country: "IL" },
  ];

  for (const guest of guests) {
    await prisma.guest.upsert({
      where: { id: guests.indexOf(guest) + 1 },
      update: {},
      create: guest,
    });
  }

  // Create sample reservations
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const reservations = [
    {
      roomId: 1,
      guestId: 1,
      checkIn: new Date(today.getTime() - 2 * 86400000),
      checkOut: new Date(today.getTime() + 1 * 86400000),
      status: "CHECKED_IN" as const,
      totalPrice: 1050,
      numGuests: 1,
      source: "phone",
    },
    {
      roomId: 4,
      guestId: 2,
      checkIn: today,
      checkOut: new Date(today.getTime() + 3 * 86400000),
      status: "CONFIRMED" as const,
      totalPrice: 1500,
      numGuests: 2,
      source: "website",
    },
    {
      roomId: 9,
      guestId: 3,
      checkIn: new Date(today.getTime() + 1 * 86400000),
      checkOut: new Date(today.getTime() + 5 * 86400000),
      status: "CONFIRMED" as const,
      totalPrice: 3600,
      numGuests: 2,
      source: "booking.com",
    },
  ];

  for (const res of reservations) {
    await prisma.reservation.create({ data: res });
  }

  // Mark room 101 as occupied (has checked-in guest)
  await prisma.room.update({ where: { number: "101" }, data: { status: "OCCUPIED" } });

  console.log("Seed completed: 30 rooms, 4 guests, 3 reservations");
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
