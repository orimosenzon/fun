// =============================================
// TypeScript Types — Road to Recovery
// מתאים לסכמת ה-DB ב-supabase/schema.sql
// =============================================

export type UserRole = 'israeli_coordinator' | 'palestinian_coordinator' | 'driver'
export type RideStatus = 'pending' | 'assigned' | 'in_progress' | 'completed' | 'cancelled' | 'return_needed'
export type ReimbursementStatus = 'pending' | 'approved' | 'rejected' | 'paid'
export type NotificationType =
  | 'ride_assigned'
  | 'ride_status_change'
  | 'return_ride_needed'
  | 'reimbursement_approved'
  | 'reimbursement_rejected'
  | 'new_ride_available'
export type LanguagePreference = 'he' | 'ar'

// =============================================
// DB Row Types (as returned from Supabase)
// =============================================

export interface Hospital {
  id: string
  name_he: string
  name_ar: string
  address: string
  city: string
  lat: number
  lng: number
  is_active: boolean
  created_at: string
}

export interface User {
  id: string
  role: UserRole
  name_he: string
  name_ar: string | null
  phone: string
  email: string
  bank_account: string | null     // מוצפן
  language_preference: LanguagePreference
  is_active: boolean
  notes: string | null            // רק לרכזת
  created_at: string
  updated_at: string
}

// User ציבורי — ללא שדות רגישים (לשימוש ב-UI של נהגים)
export interface PublicUser {
  id: string
  name_he: string
  name_ar: string | null
  phone: string
  role: UserRole
}

export interface Ride {
  id: string
  patient_name: string
  patient_phone: string | null
  pickup_address: string
  pickup_lat: number | null
  pickup_lng: number | null
  hospital_id: string
  scheduled_at: string
  status: RideStatus
  created_by: string
  driver_id: string | null
  is_return_ride: boolean
  outbound_ride_id: string | null
  return_ride_id: string | null
  distance_km: number | null
  medical_notes: string | null    // לא נחשף לנהג!
  driver_notes: string | null
  assigned_at: string | null
  started_at: string | null
  completed_at: string | null
  cancelled_at: string | null
  cancellation_reason: string | null
  created_at: string
  updated_at: string
}

// Ride עם join לטבלאות קשורות
export interface RideWithDetails extends Ride {
  hospital: Hospital
  creator: PublicUser
  driver: PublicUser | null
}

// Ride שנהג רואה — ללא medical_notes
export interface RideForDriver extends Omit<Ride, 'medical_notes'> {
  hospital: Hospital
}

export interface ReimbursementRequest {
  id: string
  ride_id: string
  driver_id: string
  distance_km: number
  rate_per_km: number
  amount_ils: number
  status: ReimbursementStatus
  approved_by: string | null
  approved_at: string | null
  rejection_reason: string | null
  created_at: string
  updated_at: string
}

export interface ReimbursementWithDetails extends ReimbursementRequest {
  ride: RideWithDetails
  driver: PublicUser
  approver: PublicUser | null
}

export interface Notification {
  id: string
  user_id: string
  type: NotificationType
  title_he: string
  title_ar: string
  message_he: string
  message_ar: string
  ride_id: string | null
  read: boolean
  read_at: string | null
  created_at: string
}

// =============================================
// API Request/Response Types
// =============================================

export interface CreateRideRequest {
  patient_name: string
  patient_phone?: string
  pickup_address: string
  hospital_id: string
  scheduled_at: string          // ISO 8601
  medical_notes?: string
  driver_notes?: string
  is_return_ride?: boolean
  outbound_ride_id?: string
}

export interface UpdateRideStatusRequest {
  status: RideStatus
  cancellation_reason?: string   // נדרש כשstatus='cancelled'
}

export interface AssignDriverRequest {
  driver_id: string
}

export interface CreateReimbursementRequest {
  ride_id: string
}

export interface ApproveReimbursementRequest {
  action: 'approve' | 'reject'
  rejection_reason?: string      // נדרש כש-action='reject'
}

export interface CreateUserRequest {
  email: string
  password: string
  role: UserRole
  name_he: string
  name_ar?: string
  phone: string
  language_preference?: LanguagePreference
  notes?: string
}

// =============================================
// State Machine — מעברי סטטוס חוקיים
// =============================================

export const VALID_STATUS_TRANSITIONS: Record<RideStatus, RideStatus[]> = {
  pending: ['assigned', 'cancelled'],
  assigned: ['in_progress', 'cancelled'],
  in_progress: ['completed'],
  completed: ['return_needed'],
  cancelled: [],                   // סופי
  return_needed: [],               // מטופל ע"י יצירת נסיעת חזרה
}

// =============================================
// UI Helper Types
// =============================================

export interface RideFilters {
  date?: string                   // YYYY-MM-DD
  status?: RideStatus | 'all'
  hospital_id?: string
  driver_id?: string
}

export interface PaginationParams {
  page: number
  limit: number
}

export interface ApiResponse<T> {
  data: T | null
  error: string | null
}
