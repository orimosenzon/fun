-- All timestamps are milliseconds since the epoch.
-- Posts and comments use increasing integer ids, so ordering by id is ordering by time.

CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL DEFAULT '',
  created_at INTEGER NOT NULL
);

-- Only hashes are stored, so a leaked database cannot be used to log in.
CREATE TABLE login_tokens (
  token_hash TEXT PRIMARY KEY,
  email TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  used_at INTEGER
);
CREATE INDEX login_tokens_email ON login_tokens (email, created_at);

CREATE TABLE sessions (
  token_hash TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users (id),
  expires_at INTEGER NOT NULL
);

CREATE TABLE friendships (
  requester_id INTEGER NOT NULL REFERENCES users (id),
  addressee_id INTEGER NOT NULL REFERENCES users (id),
  status TEXT NOT NULL CHECK (status IN ('pending', 'accepted')),
  created_at INTEGER NOT NULL,
  PRIMARY KEY (requester_id, addressee_id),
  CHECK (requester_id <> addressee_id)
);
CREATE INDEX friendships_addressee ON friendships (addressee_id, status);

CREATE TABLE posts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  author_id INTEGER NOT NULL REFERENCES users (id),
  body TEXT NOT NULL,
  youtube_id TEXT,
  audience TEXT NOT NULL CHECK (audience IN ('friends', 'public')),
  created_at INTEGER NOT NULL,
  deleted_at INTEGER
);
CREATE INDEX posts_author ON posts (author_id, id);

-- A removed comment keeps its row and place in the thread. Only its body is erased.
CREATE TABLE comments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  post_id INTEGER NOT NULL REFERENCES posts (id),
  author_id INTEGER NOT NULL REFERENCES users (id),
  body TEXT,
  created_at INTEGER NOT NULL,
  removed_by TEXT CHECK (removed_by IN ('author', 'post_owner'))
);
CREATE INDEX comments_post ON comments (post_id, id);
