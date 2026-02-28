'use client'

import { useEffect, useState } from 'react'
import { createClient } from '@/lib/supabase/client'
import type { User } from '@/lib/types'

interface CurrentUserState {
  user: User | null
  loading: boolean
  error: string | null
}

export function useCurrentUser(): CurrentUserState {
  const [state, setState] = useState<CurrentUserState>({
    user: null,
    loading: true,
    error: null,
  })

  useEffect(() => {
    const supabase = createClient()

    async function fetchUser() {
      const { data: { user: authUser }, error: authError } = await supabase.auth.getUser()

      if (authError || !authUser) {
        setState({ user: null, loading: false, error: authError?.message ?? null })
        return
      }

      const { data, error } = await supabase
        .from('users')
        .select('*')
        .eq('id', authUser.id)
        .single()

      setState({
        user: data as User | null,
        loading: false,
        error: error?.message ?? null,
      })
    }

    fetchUser()

    // האזנה לשינויי auth
    const { data: { subscription } } = supabase.auth.onAuthStateChange(() => {
      fetchUser()
    })

    return () => subscription.unsubscribe()
  }, [])

  return state
}
