import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "@/lib/queryClient";
import { AppShell } from "@/components/layout/AppShell";
import { Dashboard } from "@/pages/Dashboard";
import { Rooms } from "@/pages/Rooms";
import { Reservations } from "@/pages/Reservations";
import { NewReservation } from "@/pages/NewReservation";
import { CheckInOut } from "@/pages/CheckInOut";
import { Guests } from "@/pages/Guests";

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/rooms" element={<Rooms />} />
            <Route path="/reservations" element={<Reservations />} />
            <Route path="/reservations/new" element={<NewReservation />} />
            <Route path="/check-in-out" element={<CheckInOut />} />
            <Route path="/guests" element={<Guests />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
