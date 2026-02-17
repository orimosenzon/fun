import { useState } from "react";
import { useSearchGuests } from "@/hooks/useGuests";
import type { Guest } from "@hotel/shared";
import { Search } from "lucide-react";

interface GuestSearchProps {
  onSelect: (guest: Guest) => void;
}

export function GuestSearch({ onSelect }: GuestSearchProps) {
  const [query, setQuery] = useState("");
  const { data: results } = useSearchGuests(query);

  return (
    <div className="relative">
      <div className="relative">
        <Search size={16} className="absolute top-2.5 right-3 text-slate-400" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="חפש אורח לפי שם או טלפון..."
          className="w-full border rounded-lg pr-9 pl-3 py-2 text-sm"
        />
      </div>
      {results && results.length > 0 && (
        <div className="absolute z-10 mt-1 w-full bg-white border rounded-lg shadow-lg max-h-48 overflow-auto">
          {results.map((guest) => (
            <button
              key={guest.id}
              onClick={() => { onSelect(guest); setQuery(""); }}
              className="w-full text-right px-3 py-2 hover:bg-slate-50 text-sm"
            >
              {guest.firstName} {guest.lastName} - {guest.phone}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
