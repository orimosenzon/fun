-- The journal: what happened around getting in, so that "the app did not know me" can be
-- answered from what is written and not guessed. One line per event: the network was opened,
-- someone logged in or out, Google answered. It never holds words that members wrote, an
-- address, a token or an IP. Lines are erased after 30 days.
--
-- session is the first characters of the session's hash: enough to tell that two lines came
-- from the same device, and useless for getting in. device is the kind of phone or computer
-- and its browser, as the browser names itself. detail is a few words of this code's own.
CREATE TABLE journal (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  at INTEGER NOT NULL,
  kind TEXT NOT NULL,
  user_id INTEGER,
  session TEXT,
  device TEXT,
  detail TEXT,
  request_id TEXT
);
CREATE INDEX journal_user ON journal (user_id, at);

-- When a session began, when it was last used (kept to the hour), and on what kind of device.
-- The sessions that exist already began 90 days before they end.
ALTER TABLE sessions ADD COLUMN created_at INTEGER;
ALTER TABLE sessions ADD COLUMN seen_at INTEGER;
ALTER TABLE sessions ADD COLUMN device TEXT;
UPDATE sessions SET created_at = expires_at - 90 * 24 * 60 * 60 * 1000;
