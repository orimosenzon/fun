-- Tagging a friend: "@name" in the words of a post or a comment, pointing at a member. The
-- words keep the name as it was written (label), so a tag still reads right after the member
-- changed their name. post_id is always the post; comment_id is set when the tag is in a comment.
-- The order of the rows is the order of the tags in the words.
CREATE TABLE mentions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  post_id INTEGER NOT NULL REFERENCES posts (id),
  comment_id INTEGER REFERENCES comments (id),
  user_id INTEGER NOT NULL REFERENCES users (id),
  label TEXT NOT NULL
);
CREATE INDEX mentions_post ON mentions (post_id);
CREATE INDEX mentions_user ON mentions (user_id);

-- Whoever was tagged is told: a new kind of notice. The kinds are listed in a CHECK, which
-- cannot be changed in place, so the table is built again with the same rows.
CREATE TABLE notices_new (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES users (id),
  actor_id INTEGER NOT NULL REFERENCES users (id),
  kind TEXT NOT NULL CHECK (kind IN ('comment', 'reply', 'thread', 'felt', 'share', 'reshare', 'request', 'accepted', 'joined', 'mention')),
  at INTEGER NOT NULL,
  post_id INTEGER REFERENCES posts (id),
  comment_id INTEGER REFERENCES comments (id),
  share_id INTEGER REFERENCES posts (id)
);
INSERT INTO notices_new (id, user_id, actor_id, kind, at, post_id, comment_id, share_id)
  SELECT id, user_id, actor_id, kind, at, post_id, comment_id, share_id FROM notices;
DROP TABLE notices;
ALTER TABLE notices_new RENAME TO notices;
CREATE INDEX notices_user ON notices (user_id, at);
-- A tag is told once too: taking it out of the words and putting it back does not ring again.
CREATE UNIQUE INDEX notices_once ON notices (user_id, actor_id, kind, IFNULL(post_id, 0), IFNULL(comment_id, 0))
  WHERE kind IN ('felt', 'share', 'reshare', 'request', 'accepted', 'joined', 'mention');
