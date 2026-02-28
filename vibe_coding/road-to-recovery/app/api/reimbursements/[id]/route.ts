import { NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'

// PATCH /api/reimbursements/[id]
// רכזת ישראלית מאשרת או דוחה בקשת החזר
export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params
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

  const { action, rejection_reason } = await request.json()

  if (!['approve', 'reject'].includes(action)) {
    return NextResponse.json({ error: 'action must be approve or reject' }, { status: 400 })
  }

  if (action === 'reject' && !rejection_reason) {
    return NextResponse.json({ error: 'rejection_reason required when rejecting' }, { status: 400 })
  }

  const updates: Record<string, unknown> = {
    status: action === 'approve' ? 'approved' : 'rejected',
    approved_by: user.id,
    approved_at: new Date().toISOString(),
  }
  if (action === 'reject') updates.rejection_reason = rejection_reason

  const { data, error } = await supabase
    .from('reimbursement_requests')
    .update(updates)
    .eq('id', id)
    .eq('status', 'pending')   // אפשר לפעול רק על pending
    .select('*, driver:users!reimbursement_requests_driver_id_fkey(id, name_he)')
    .single()

  if (error || !data) {
    return NextResponse.json({ error: 'Reimbursement not found or already processed' }, { status: 404 })
  }

  // notification לנהג
  const reimbData = data as typeof data & { driver: { id: string; name_he: string } }
  await supabase.from('notifications').insert({
    user_id: reimbData.driver?.id,
    type: action === 'approve' ? 'reimbursement_approved' : 'reimbursement_rejected',
    title_he: action === 'approve' ? 'החזר נסיעה אושר' : 'החזר נסיעה נדחה',
    title_ar: action === 'approve' ? 'تمت الموافقة على استرداد السفر' : 'تم رفض استرداد السفر',
    message_he: action === 'approve'
      ? 'בקשת החזר הנסיעה שלך אושרה. הכסף יועבר בקרוב.'
      : `בקשת החזר נדחתה: ${rejection_reason}`,
    message_ar: action === 'approve'
      ? 'تمت الموافقة على طلب استرداد السفر. سيتم تحويل المبلغ قريباً.'
      : `تم رفض طلب الاسترداد: ${rejection_reason}`,
  })

  return NextResponse.json({ data })
}
