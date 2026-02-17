import { NavLink } from "react-router-dom";
import { LayoutDashboard, BedDouble, CalendarDays, UserCheck, Users } from "lucide-react";
import { cn } from "@/lib/cn";

const navItems = [
  { to: "/", label: "לוח בקרה", icon: LayoutDashboard },
  { to: "/rooms", label: "חדרים", icon: BedDouble },
  { to: "/reservations", label: "הזמנות", icon: CalendarDays },
  { to: "/check-in-out", label: "כניסה / יציאה", icon: UserCheck },
  { to: "/guests", label: "אורחים", icon: Users },
];

export function Sidebar() {
  return (
    <aside className="w-60 bg-slate-800 text-white min-h-screen flex flex-col">
      <div className="p-5 border-b border-slate-700">
        <h1 className="text-xl font-bold">🏨 ניהול מלון</h1>
      </div>
      <nav className="flex-1 py-4">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 px-5 py-3 text-sm transition-colors",
                isActive ? "bg-slate-700 text-white font-medium" : "text-slate-300 hover:bg-slate-700/50 hover:text-white"
              )
            }
          >
            <item.icon size={20} />
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
