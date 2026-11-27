-- What the member chose about how the network behaves for them, on every device: a small JSON
-- object. Only what they changed is kept, and the rest follows the defaults in src/index.js.
ALTER TABLE users ADD COLUMN prefs TEXT NOT NULL DEFAULT '{}';
