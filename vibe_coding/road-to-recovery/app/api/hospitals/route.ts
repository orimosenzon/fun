import { NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'

// GET /api/hospitals
// מחזיר את כל בתי החולים הפעילים (RLS מגבילה לmauthenticated בלבד)
export async function GET() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { data, error } = await supabase
    .from('hospitals')
    .select('id, name_he, name_ar, address, city, lat, lng')
    .eq('is_active', true)
    .order('name_he')

  if (error) return NextResponse.json({ error: error.message }, { status: 500 })

  return NextResponse.json({ data })
}
