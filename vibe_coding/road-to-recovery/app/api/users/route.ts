import { NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import type { UserRole } from '@/lib/types'

// GET /api/users
// רכזת ישראלית — כולם; שאר ה-roles — לא מורשים
// תמיכה ב-?role=driver לסינון
export async function GET(request: Request) {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { data: profile } = await supabase
    .from('users')
    .select('role')
    .eq('id', user.id)
    .single()

  const { searchParams } = new URL(request.url)
  const roleFilter = searchParams.get('role') as UserRole | null

  // נהגים ורכזים פלסטינים — יכולים לראות רק נהגים (לצורך AssignDriver)
  // רכזת — הכל
  if (profile?.role !== 'israeli_coordinator' && roleFilter !== 'driver') {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 })
  }

  // שדות מוצגים לפי role: רכזת רואה הכל, שאר — רק שדות ציבוריים
  const isCoordinator = profile?.role === 'israeli_coordinator'
  const selectFields = isCoordinator
    ? 'id, role, name_he, name_ar, phone, email, is_active, notes, created_at'
    : 'id, role, name_he, name_ar, phone'  // ללא bank_account, email, notes

  let query = supabase
    .from('users')
    .select(selectFields)
    .eq('is_active', true)
    .order('name_he')

  if (roleFilter) {
    query = query.eq('role', roleFilter)
  }

  const { data, error } = await query

  if (error) return NextResponse.json({ error: error.message }, { status: 500 })

  return NextResponse.json({ data })
}

// POST /api/users
// רכזת יוצרת משתמש חדש (Supabase Admin API)
export async function POST(request: Request) {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

  const { data: profile } = await supabase
    .from('users')
    .select('role')
    .eq('id', user.id)
    .single()

  if (profile?.role !== 'israeli_coordinator') {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 })
  }

  const body = await request.json()
  const { email, password, role, name_he, name_ar, phone, language_preference, notes } = body

  if (!email || !password || !role || !name_he || !phone) {
    return NextResponse.json({ error: 'Missing required fields' }, { status: 400 })
  }

  // יצירת auth user עם Admin API (דורש SERVICE_ROLE_KEY)
  // ה-trigger של Supabase יוסיף לטבלת users
  const { createClient: createAdminClient } = await import('@supabase/supabase-js')
  const adminSupabase = createAdminClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  )

  const { data: authData, error: authError } = await adminSupabase.auth.admin.createUser({
    email,
    password,
    email_confirm: true,
  })

  if (authError || !authData.user) {
    return NextResponse.json({ error: authError?.message ?? 'Failed to create auth user' }, { status: 500 })
  }

  // הוספה לטבלת users
  const { data, error } = await adminSupabase
    .from('users')
    .insert({
      id: authData.user.id,
      role,
      name_he,
      name_ar: name_ar ?? null,
      phone,
      email,
      language_preference: language_preference ?? 'he',
      notes: notes ?? null,
    })
    .select()
    .single()

  if (error) {
    // rollback — מחיקת auth user אם הוספה ל-DB נכשלה
    await adminSupabase.auth.admin.deleteUser(authData.user.id)
    return NextResponse.json({ error: error.message }, { status: 500 })
  }

  return NextResponse.json({ data }, { status: 201 })
}
