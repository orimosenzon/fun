-- An invitation is now sent to an email address, and may carry the name the admin knows the
-- person by. The address stays on the account, so the member can come back with Google.
ALTER TABLE login_tokens ADD COLUMN name TEXT;

-- Invitations by Google address become ordinary invitations. Their hash matches no link, so
-- they can only be taken by signing in with Google at that address, exactly as before.
INSERT INTO login_tokens (token_hash, email, created_at, expires_at, invited_by)
  SELECT 'google-invite:' || email, email, created_at, (unixepoch() + 60 * 86400) * 1000, invited_by FROM google_invites;
DROP TABLE google_invites;
