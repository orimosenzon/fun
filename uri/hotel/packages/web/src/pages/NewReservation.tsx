import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAvailableRooms } from "@/hooks/useRooms";
import { useCreateReservation } from "@/hooks/useReservations";
import { useCreateGuest } from "@/hooks/useGuests";
import { GuestSearch } from "@/components/guests/GuestSearch";
import { GuestForm } from "@/components/guests/GuestForm";
import { RoomCard } from "@/components/rooms/RoomCard";
import { ROOM_TYPES, ROOM_TYPE_LABELS_HE } from "@hotel/shared";
import type { Guest, CreateGuestInput, RoomType } from "@hotel/shared";
import { differenceInDays } from "date-fns";
import { UserPlus } from "lucide-react";

export function NewReservation() {
  const navigate = useNavigate();
  const [checkIn, setCheckIn] = useState("");
  const [checkOut, setCheckOut] = useState("");
  const [roomType, setRoomType] = useState<string>("");
  const [selectedRoomId, setSelectedRoomId] = useState<number | null>(null);
  const [selectedRoomPrice, setSelectedRoomPrice] = useState(0);
  const [selectedGuest, setSelectedGuest] = useState<Guest | null>(null);
  const [showNewGuest, setShowNewGuest] = useState(false);
  const [numGuests, setNumGuests] = useState(1);
  const [notes, setNotes] = useState("");
  const [source, setSource] = useState<string>("walk-in");

  const { data: availableRooms } = useAvailableRooms(checkIn, checkOut, roomType || undefined);
  const createReservation = useCreateReservation();
  const createGuest = useCreateGuest();

  const nights = checkIn && checkOut ? differenceInDays(new Date(checkOut), new Date(checkIn)) : 0;
  const totalPrice = nights * selectedRoomPrice;

  const handleNewGuest = async (data: CreateGuestInput) => {
    const guest = await createGuest.mutateAsync(data);
    setSelectedGuest(guest);
    setShowNewGuest(false);
  };

  const handleSubmit = async () => {
    if (!selectedRoomId || !selectedGuest || !checkIn || !checkOut) return;
    await createReservation.mutateAsync({
      roomId: selectedRoomId,
      guestId: selectedGuest.id,
      checkIn,
      checkOut,
      numGuests,
      totalPrice,
      notes: notes || undefined,
      source: source as any,
    });
    navigate("/reservations");
  };

  return (
    <div className="max-w-4xl">
      <h2 className="text-2xl font-bold text-slate-800 mb-6">הזמנה חדשה</h2>

      {/* Step 1: Dates */}
      <div className="bg-white rounded-lg shadow-sm border p-5 mb-4">
        <h3 className="font-semibold text-slate-800 mb-3">1. תאריכים</h3>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-slate-600 mb-1">כניסה</label>
            <input type="date" value={checkIn} onChange={(e) => setCheckIn(e.target.value)} className="w-full border rounded-lg px-3 py-2 text-sm" dir="ltr" />
          </div>
          <div>
            <label className="block text-sm text-slate-600 mb-1">יציאה</label>
            <input type="date" value={checkOut} onChange={(e) => setCheckOut(e.target.value)} className="w-full border rounded-lg px-3 py-2 text-sm" dir="ltr" />
          </div>
          <div>
            <label className="block text-sm text-slate-600 mb-1">סוג חדר</label>
            <select value={roomType} onChange={(e) => setRoomType(e.target.value)} className="w-full border rounded-lg px-3 py-2 text-sm">
              <option value="">כל הסוגים</option>
              {ROOM_TYPES.map((t) => (
                <option key={t} value={t}>{ROOM_TYPE_LABELS_HE[t]}</option>
              ))}
            </select>
          </div>
        </div>
        {nights > 0 && <p className="text-sm text-slate-500 mt-2">{nights} לילות</p>}
      </div>

      {/* Step 2: Room */}
      {availableRooms && (
        <div className="bg-white rounded-lg shadow-sm border p-5 mb-4">
          <h3 className="font-semibold text-slate-800 mb-3">2. בחר חדר ({availableRooms.length} זמינים)</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {availableRooms.map((room) => (
              <div key={room.id} className={selectedRoomId === room.id ? "ring-2 ring-blue-500 rounded-lg" : ""}>
                <RoomCard
                  number={room.number}
                  floor={room.floor}
                  type={room.type as RoomType}
                  status={room.status as any}
                  basePrice={Number(room.basePrice)}
                  onClick={() => { setSelectedRoomId(room.id); setSelectedRoomPrice(Number(room.basePrice)); }}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Step 3: Guest */}
      {selectedRoomId && (
        <div className="bg-white rounded-lg shadow-sm border p-5 mb-4">
          <h3 className="font-semibold text-slate-800 mb-3">3. אורח</h3>
          {selectedGuest ? (
            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
              <span className="font-medium">{selectedGuest.firstName} {selectedGuest.lastName} - {selectedGuest.phone}</span>
              <button onClick={() => setSelectedGuest(null)} className="text-sm text-blue-600 hover:underline">שנה</button>
            </div>
          ) : (
            <div className="space-y-3">
              <GuestSearch onSelect={setSelectedGuest} />
              {!showNewGuest ? (
                <button onClick={() => setShowNewGuest(true)} className="flex items-center gap-2 text-sm text-blue-600 hover:underline">
                  <UserPlus size={16} />
                  אורח חדש
                </button>
              ) : (
                <div className="border rounded-lg p-4">
                  <GuestForm onSubmit={handleNewGuest} isLoading={createGuest.isPending} />
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Step 4: Details & Confirm */}
      {selectedGuest && (
        <div className="bg-white rounded-lg shadow-sm border p-5 mb-4">
          <h3 className="font-semibold text-slate-800 mb-3">4. פרטים ואישור</h3>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div>
              <label className="block text-sm text-slate-600 mb-1">מספר אורחים</label>
              <input type="number" value={numGuests} onChange={(e) => setNumGuests(Number(e.target.value))} min={1} className="w-full border rounded-lg px-3 py-2 text-sm" />
            </div>
            <div>
              <label className="block text-sm text-slate-600 mb-1">מקור הזמנה</label>
              <select value={source} onChange={(e) => setSource(e.target.value)} className="w-full border rounded-lg px-3 py-2 text-sm">
                <option value="walk-in">Walk-in</option>
                <option value="phone">טלפון</option>
                <option value="website">אתר</option>
                <option value="booking.com">Booking.com</option>
                <option value="expedia">Expedia</option>
                <option value="other">אחר</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-slate-600 mb-1">סה"כ</label>
              <p className="text-2xl font-bold text-slate-800">₪{totalPrice.toLocaleString()}</p>
              <p className="text-xs text-slate-500">{nights} לילות × ₪{selectedRoomPrice}</p>
            </div>
          </div>
          <div className="mb-4">
            <label className="block text-sm text-slate-600 mb-1">הערות</label>
            <textarea value={notes} onChange={(e) => setNotes(e.target.value)} className="w-full border rounded-lg px-3 py-2 text-sm" rows={2} />
          </div>
          <button
            onClick={handleSubmit}
            disabled={createReservation.isPending}
            className="w-full bg-green-600 text-white py-3 rounded-lg font-medium hover:bg-green-700 disabled:opacity-50"
          >
            {createReservation.isPending ? "יוצר הזמנה..." : "אשר הזמנה"}
          </button>
        </div>
      )}
    </div>
  );
}
