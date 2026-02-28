import { redirect } from 'next/navigation'
import { createClient } from '@/lib/supabase/server'
import type { UserRole } from '@/lib/types'

// דף ראשי — redirect לפי role
export default async function Home() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) {
    redirect('/login')
  }

  const { data: profile } = await supabase
    .from('users')
    .select('role')
    .eq('id', user.id)
    .single()

  const roleRoutes: Record<UserRole, string> = {
    israeli_coordinator: '/coordinator/rides',
    palestinian_coordinator: '/palestinian-coordinator/rides',
    driver: '/driver/rides',
  }

  const route = profile ? roleRoutes[profile.role as UserRole] : '/login'
  redirect(route)
}
