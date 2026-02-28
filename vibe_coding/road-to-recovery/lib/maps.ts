// =============================================
// Google Maps Distance Matrix API wrapper
// =============================================

interface DistanceResult {
  distanceKm: number
  durationMinutes: number
  originLat: number
  originLng: number
}

interface GeocodeResult {
  lat: number
  lng: number
  formattedAddress: string
}

/**
 * Geocode כתובת → קואורדינטות
 * קריאת server-side בלבד (API key לא נחשף ל-client)
 */
export async function geocodeAddress(address: string): Promise<GeocodeResult> {
  const apiKey = process.env.GOOGLE_MAPS_API_KEY
  if (!apiKey) throw new Error('GOOGLE_MAPS_API_KEY not configured')

  const url = new URL('https://maps.googleapis.com/maps/api/geocode/json')
  url.searchParams.set('address', address)
  url.searchParams.set('key', apiKey)
  url.searchParams.set('language', 'he')
  url.searchParams.set('region', 'il')

  const response = await fetch(url.toString())
  const data = await response.json()

  if (data.status !== 'OK' || !data.results[0]) {
    throw new Error(`Geocoding failed: ${data.status}`)
  }

  const { lat, lng } = data.results[0].geometry.location
  return {
    lat,
    lng,
    formattedAddress: data.results[0].formatted_address,
  }
}

/**
 * חישוב מרחק ממוצא ליעד
 * מחזיר מרחק ב-km וזמן נסיעה מוערך בדקות
 *
 * הקריאה מתבצעת פעם אחת בלבד עם יצירת הנסיעה — הערך נשמר ב-DB
 * (לא מחושב מחדש בכל צפייה — חיסכון בעלות API)
 */
export async function calculateDistance(
  originAddress: string,
  destinationLat: number,
  destinationLng: number
): Promise<DistanceResult> {
  const apiKey = process.env.GOOGLE_MAPS_API_KEY
  if (!apiKey) throw new Error('GOOGLE_MAPS_API_KEY not configured')

  // geocode המוצא תחילה
  const origin = await geocodeAddress(originAddress)

  const url = new URL('https://maps.googleapis.com/maps/api/distancematrix/json')
  url.searchParams.set('origins', `${origin.lat},${origin.lng}`)
  url.searchParams.set('destinations', `${destinationLat},${destinationLng}`)
  url.searchParams.set('key', apiKey)
  url.searchParams.set('language', 'he')
  url.searchParams.set('mode', 'driving')

  const response = await fetch(url.toString())
  const data = await response.json()

  if (data.status !== 'OK') {
    throw new Error(`Distance Matrix failed: ${data.status}`)
  }

  const element = data.rows[0]?.elements[0]
  if (!element || element.status !== 'OK') {
    throw new Error('Could not calculate distance for this route')
  }

  return {
    distanceKm: element.distance.value / 1000,       // מטרים → ק"מ
    durationMinutes: element.duration.value / 60,     // שניות → דקות
    originLat: origin.lat,
    originLng: origin.lng,
  }
}
