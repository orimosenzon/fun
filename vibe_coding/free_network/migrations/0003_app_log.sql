-- Errors from the server and from browsers. Never holds post text, emails or tokens.
CREATE TABLE app_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  at INTEGER NOT NULL,
  source TEXT NOT NULL CHECK (source IN ('server', 'client')),
  level TEXT NOT NULL CHECK (level IN ('warn', 'error')),
  request_id TEXT,
  user_id INTEGER,
  method TEXT,
  path TEXT,
  status INTEGER,
  message TEXT,
  stack TEXT
);
CREATE INDEX app_log_request ON app_log (request_id);
