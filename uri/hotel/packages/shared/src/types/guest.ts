export interface Guest {
  id: number;
  firstName: string;
  lastName: string;
  email: string | null;
  phone: string;
  idNumber: string | null;
  idType: string | null;
  country: string;
  address: string | null;
  notes: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface CreateGuestInput {
  firstName: string;
  lastName: string;
  phone: string;
  email?: string;
  idNumber?: string;
  idType?: string;
  country?: string;
  address?: string;
  notes?: string;
}

export interface UpdateGuestInput {
  firstName?: string;
  lastName?: string;
  phone?: string;
  email?: string | null;
  idNumber?: string | null;
  idType?: string | null;
  country?: string;
  address?: string | null;
  notes?: string | null;
}
