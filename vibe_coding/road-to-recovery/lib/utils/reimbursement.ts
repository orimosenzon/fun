// =============================================
// חישובי החזר נסיעה
// =============================================

// תעריף ברירת מחדל — שקל וחצי לק"מ
export const DEFAULT_RATE_PER_KM = 1.50

/**
 * מחשב סכום החזר
 * @param distanceKm - מרחק בקילומטרים
 * @param ratePerKm - תעריף לק"מ (ברירת מחדל: 1.5 ₪)
 * @returns סכום ב-₪, מעוגל ל-2 ספרות עשרוניות
 */
export function calculateReimbursement(
  distanceKm: number,
  ratePerKm: number = DEFAULT_RATE_PER_KM
): number {
  return Math.round(distanceKm * ratePerKm * 100) / 100
}

/**
 * פורמט סכום כספי לתצוגה
 */
export function formatAmount(amount: number): string {
  return `₪${amount.toFixed(2)}`
}

/**
 * פורמט מרחק לתצוגה
 */
export function formatDistance(km: number): string {
  return `${km.toFixed(1)} ק"מ`
}
