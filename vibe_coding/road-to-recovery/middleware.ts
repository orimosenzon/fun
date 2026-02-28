import { createServerClient } from '@supabase/ssr'
import { NextResponse, type NextRequest } from 'next/server'
import type { UserRole } from '@/lib/types'

// נתיבים שדורשים כניסה לפי role
const ROLE_ROUTES: Record<string, UserRole[]> = {
  '/coordinator': ['israeli_coordinator'],
  '/palestinian-coordinator': ['palestinian_coordinator'],
  '/driver': ['driver'],
}

export async function middleware(request: NextRequest) {
  let supabaseResponse = NextResponse.next({ request })

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll()
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value)
          )
          supabaseResponse = NextResponse.next({ request })
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options)
          )
        },
      },
    }
  )

  // רענון session — חובה לקרוא לפני כל שימוש ב-session
  const { data: { user } } = await supabase.auth.getUser()

  const { pathname } = request.nextUrl

  // דף login — אם כבר מחובר, redirect ל-dashboard
  if (pathname === '/login') {
    if (user) {
      return redirectToDashboard(request, user.id, supabase)
    }
    return supabaseResponse
  }

  // דף ראשי — redirect
  if (pathname === '/') {
    if (user) {
      return redirectToDashboard(request, user.id, supabase)
    }
    return NextResponse.redirect(new URL('/login', request.url))
  }

  // API routes — לא דורשים middleware (Supabase Auth מטפל)
  if (pathname.startsWith('/api/')) {
    return supabaseResponse
  }

  // בדיקת auth לדשבורד
  if (!user) {
    return NextResponse.redirect(new URL('/login', request.url))
  }

  // בדיקת role
  for (const [routePrefix, allowedRoles] of Object.entries(ROLE_ROUTES)) {
    if (pathname.startsWith(routePrefix)) {
      const { data: userProfile } = await supabase
        .from('users')
        .select('role')
        .eq('id', user.id)
        .single()

      if (!userProfile || !allowedRoles.includes(userProfile.role as UserRole)) {
        // משתמש ניסה לגשת לנתיב לא מורשה — redirect ל-dashboard שלו
        return redirectToDashboard(request, user.id, supabase)
      }
      break
    }
  }

  return supabaseResponse
}

async function redirectToDashboard(
  request: NextRequest,
  userId: string,
  supabase: ReturnType<typeof createServerClient>
): Promise<NextResponse> {
  const { data: userProfile } = await supabase
    .from('users')
    .select('role')
    .eq('id', userId)
    .single()

  const roleRoutes: Record<UserRole, string> = {
    israeli_coordinator: '/coordinator/rides',
    palestinian_coordinator: '/palestinian-coordinator/rides',
    driver: '/driver/rides',
  }

  const route = userProfile
    ? roleRoutes[userProfile.role as UserRole]
    : '/login'

  return NextResponse.redirect(new URL(route, request.url))
}

export const config = {
  matcher: [
    // כל הנתיבים מלבד static assets
    '/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
}
