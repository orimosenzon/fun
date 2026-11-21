-- The admin may let the network send invitations from their own Gmail. Google is asked only for
-- permission to send, and the key it gives is kept sealed: it opens only with a secret that is
-- not in the database.
CREATE TABLE mailers (
  user_id INTEGER PRIMARY KEY REFERENCES users (id),
  address TEXT NOT NULL,
  refresh_token TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
