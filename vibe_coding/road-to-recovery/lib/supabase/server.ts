import { createServerClient } from '@supabase/ssr'
import { cookies } from 'next/headers'

// Server-side client — קורא/כותב cookies לניהול session
// נוצר מחדש לכל request (לא singleton)
export async function createClient() {
  const cookieStore = await cookies()

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options)
            )
          } catch {
            // setAll יכול להיכשל ב-Server Components (read-only)
            // זה בסדר — middleware מטפל ברענון ה-session
          }
        },
      },
    }
  )
}
