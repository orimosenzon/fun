"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { signOut, useSession } from "next-auth/react";

const adminLinks = [
  { href: "/", label: "ראשי", icon: "🏠" },
  { href: "/schedule", label: "לוח שיעורים", icon: "📆" },
  { href: "/groups", label: "קבוצות", icon: "👥" },
  { href: "/students", label: "תלמידים", icon: "🎓" },
  { href: "/payments", label: "תשלומים", icon: "💰" },
];

const studentLinks = [
  { href: "/my", label: "השיעורים שלי", icon: "📅" },
];

export default function Navbar() {
  const pathname = usePathname();
  const { data: session } = useSession();
  const isAdmin = (session?.user as { role?: string })?.role === "ADMIN";
  const links = isAdmin ? adminLinks : studentLinks;

  return (
    <nav className="bg-white border-b border-stone-200 sticky top-0 z-10">
      <div className="max-w-5xl mx-auto px-4 flex items-center justify-between h-14">
        <div className="flex items-center gap-1">
          <span className="text-2xl ml-3">🏺</span>
          <span className="font-bold text-stone-800 ml-4 hidden sm:block">סטודיו קרמיקה</span>
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition
                ${pathname === link.href
                  ? "bg-amber-100 text-amber-800"
                  : "text-stone-600 hover:bg-stone-100"
                }`}
            >
              <span>{link.icon}</span>
              <span>{link.label}</span>
            </Link>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <span className="text-sm text-stone-500 hidden sm:block">
            {session?.user?.name}
            {isAdmin && <span className="mr-1 text-amber-600 text-xs">(מנהלת)</span>}
          </span>
          <button
            onClick={() => signOut({ callbackUrl: "/login" })}
            className="text-sm text-stone-500 hover:text-red-600 transition"
          >
            יציאה
          </button>
        </div>
      </div>
    </nav>
  );
}
