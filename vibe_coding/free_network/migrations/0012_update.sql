-- "What's up": one line a member writes for their friends. It lasts a week and is then erased.
ALTER TABLE users ADD COLUMN status_text TEXT;
ALTER TABLE users ADD COLUMN status_at INTEGER;
-- When the shared place last changed, so friends can be told someone moved.
ALTER TABLE users ADD COLUMN place_at INTEGER;

-- News from outside the network (events, building plans, Wikipedia) is fetched by the server
-- a few times a day and kept here, so reading the update never contacts anyone else.
CREATE TABLE digest_cache (
  key TEXT PRIMARY KEY,
  body TEXT NOT NULL,
  fetched_at INTEGER NOT NULL,
  tried_at INTEGER NOT NULL
);
