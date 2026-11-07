-- An invitation by Google address. It is deleted once used: the address itself is never kept on
-- the account, only Google's anonymous id.
CREATE TABLE google_invites (
  email TEXT PRIMARY KEY,
  invited_by INTEGER NOT NULL REFERENCES users (id),
  created_at INTEGER NOT NULL
);
