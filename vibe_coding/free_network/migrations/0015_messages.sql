-- Private messages between two members. Each pair has one conversation, kept with the smaller
-- id first. Message ids only grow, so ordering by id is ordering by time.
CREATE TABLE conversations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  low_id INTEGER NOT NULL REFERENCES users (id),
  high_id INTEGER NOT NULL REFERENCES users (id),
  created_at INTEGER NOT NULL,
  -- The newest message each side has seen. It is what counts the unread messages on every
  -- device, and it is never shown to the other side.
  low_read INTEGER NOT NULL DEFAULT 0,
  high_read INTEGER NOT NULL DEFAULT 0,
  UNIQUE (low_id, high_id),
  CHECK (low_id < high_id)
);
CREATE INDEX conversations_high ON conversations (high_id);

-- A removed message keeps its row and its place in the conversation. Only its body is erased.
CREATE TABLE messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id INTEGER NOT NULL REFERENCES conversations (id),
  sender_id INTEGER NOT NULL REFERENCES users (id),
  body TEXT,
  created_at INTEGER NOT NULL,
  removed_at INTEGER
);
CREATE INDEX messages_conversation ON messages (conversation_id, id);
CREATE INDEX messages_removed ON messages (conversation_id, removed_at) WHERE removed_at IS NOT NULL;

-- A message can carry one image. Only the two people in the conversation can see it.
ALTER TABLE images ADD COLUMN message_id INTEGER REFERENCES messages (id);
CREATE INDEX images_message ON images (message_id) WHERE message_id IS NOT NULL;
