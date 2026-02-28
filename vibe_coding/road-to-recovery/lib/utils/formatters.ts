// =============================================
// Formatters — תאריכים, זמנים, טקסט
// =============================================

/**
 * פורמט תאריך לתצוגה — DD/MM/YYYY
 */
export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('he-IL', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  })
}

/**
 * פורמט שעה — HH:MM
 */
export function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString('he-IL', {
    hour: '2-digit',
    minute: '2-digit',
  })
}

/**
 * פורמט תאריך + שעה ביחד
 */
export function formatDateTime(iso: string): string {
  return `${formatDate(iso)} ${formatTime(iso)}`
}

/**
 * תאריך בפורמט YYYY-MM-DD (לשאילתות DB ולinput[type=date])
 */
export function toDateString(date: Date = new Date()): string {
  return date.toISOString().split('T')[0]
}

/**
 * האם תאריך הוא היום
 */
export function isToday(iso: string): boolean {
  return iso.startsWith(toDateString())
}

/**
 * זמן יחסי ("לפני 5 דקות", "לפני שעה")
 */
export function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)

  if (minutes < 1) return 'הרגע'
  if (minutes < 60) return `לפני ${minutes} דקות`
  if (hours < 24) return `לפני ${hours} שעות`
  return `לפני ${days} ימים`
}
