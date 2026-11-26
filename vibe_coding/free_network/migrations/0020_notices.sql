-- A notice tells a member (user_id) what someone else (actor_id) did around them: a comment on
-- their post, a reply to their comment, a feeling, a share, a friend request. It holds only
-- pointers, never the words: the page reads those from the comment or the post itself, so a
-- notice about something that was deleted, or that the member may no longer read, is not shown.
--
-- post_id is the post it happened on. comment_id is the comment that was written, or for a
-- feeling, the member's own comment that it is about. share_id is the post that shares.
CREATE TABLE notices (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES users (id),
  actor_id INTEGER NOT NULL REFERENCES users (id),
  kind TEXT NOT NULL CHECK (kind IN ('comment', 'reply', 'thread', 'felt', 'share', 'reshare', 'request', 'accepted', 'joined')),
  at INTEGER NOT NULL,
  post_id INTEGER REFERENCES posts (id),
  comment_id INTEGER REFERENCES comments (id),
  share_id INTEGER REFERENCES posts (id)
);
CREATE INDEX notices_user ON notices (user_id, at);
-- Every comment is a notice of its own. The rest are told once: whoever takes a feeling back
-- and gives it again, or asks for friendship again, does not ring the same person twice.
CREATE UNIQUE INDEX notices_once ON notices (user_id, actor_id, kind, IFNULL(post_id, 0), IFNULL(comment_id, 0))
  WHERE kind IN ('felt', 'share', 'reshare', 'request', 'accepted', 'joined');

-- A device that asked to be told also when the network is closed (Web Push). It belongs to the
-- session that asked, so logging out, or a session that ran out, silences it.
CREATE TABLE push_devices (
  endpoint TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users (id),
  session_hash TEXT NOT NULL,
  p256dh TEXT NOT NULL,
  auth TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
CREATE INDEX push_devices_user ON push_devices (user_id);

-- Up to when the member has seen their notices, on any device, and the kinds they asked not
-- to be pushed about (comma separated: messages, comments, feelings, shares, friends).
ALTER TABLE users ADD COLUMN notices_seen_at INTEGER NOT NULL DEFAULT 0;
ALTER TABLE users ADD COLUMN push_muted TEXT NOT NULL DEFAULT '';

-- What happened before there were notices, so the page starts with its history.
INSERT INTO notices (user_id, actor_id, kind, at, post_id, comment_id)
SELECT parent.author_id, m.author_id, 'reply', m.created_at, m.post_id, m.id
FROM comments m JOIN comments parent ON parent.id = m.parent_id JOIN posts p ON p.id = m.post_id
WHERE m.removed_by IS NULL AND p.deleted_at IS NULL AND parent.author_id <> m.author_id;

INSERT INTO notices (user_id, actor_id, kind, at, post_id, comment_id)
SELECT p.author_id, m.author_id, 'comment', m.created_at, m.post_id, m.id
FROM comments m JOIN posts p ON p.id = m.post_id LEFT JOIN comments parent ON parent.id = m.parent_id
WHERE m.removed_by IS NULL AND p.deleted_at IS NULL AND p.author_id <> m.author_id
  AND (parent.id IS NULL OR parent.author_id <> p.author_id);

INSERT INTO notices (user_id, actor_id, kind, at, post_id, comment_id)
SELECT DISTINCT earlier.author_id, m.author_id, 'thread', m.created_at, m.post_id, m.id
FROM comments m JOIN comments earlier ON earlier.parent_id = m.parent_id AND earlier.id < m.id AND earlier.removed_by IS NULL
JOIN comments parent ON parent.id = m.parent_id JOIN posts p ON p.id = m.post_id
WHERE m.removed_by IS NULL AND p.deleted_at IS NULL AND earlier.author_id <> m.author_id
  AND earlier.author_id <> parent.author_id AND earlier.author_id <> p.author_id;

INSERT INTO notices (user_id, actor_id, kind, at, post_id)
SELECT p.author_id, r.user_id, 'felt', r.created_at, p.id
FROM reactions r JOIN posts p ON p.id = r.post_id
WHERE p.deleted_at IS NULL AND p.author_id <> r.user_id;

INSERT INTO notices (user_id, actor_id, kind, at, post_id, comment_id)
SELECT m.author_id, r.user_id, 'felt', r.created_at, m.post_id, m.id
FROM reactions r JOIN comments m ON m.id = r.comment_id JOIN posts p ON p.id = m.post_id
WHERE p.deleted_at IS NULL AND m.removed_by IS NULL AND m.author_id <> r.user_id;

INSERT INTO notices (user_id, actor_id, kind, at, post_id, share_id)
SELECT o.author_id, s.author_id, 'share', s.created_at, o.id, s.id
FROM posts s JOIN posts o ON o.id = s.shared_id
WHERE s.deleted_at IS NULL AND o.deleted_at IS NULL AND o.author_id <> s.author_id;

INSERT INTO notices (user_id, actor_id, kind, at, post_id, share_id)
SELECT via.author_id, s.author_id, 'reshare', s.created_at, via.id, s.id
FROM posts s JOIN posts via ON via.id = s.shared_via JOIN posts o ON o.id = s.shared_id
WHERE s.deleted_at IS NULL AND via.deleted_at IS NULL AND o.deleted_at IS NULL
  AND via.author_id <> s.author_id AND via.author_id <> o.author_id;

INSERT INTO notices (user_id, actor_id, kind, at)
SELECT addressee_id, requester_id, 'request', created_at FROM friendships WHERE status = 'pending';

INSERT INTO notices (user_id, actor_id, kind, at)
SELECT t.invited_by, u.id, 'joined', MIN(t.used_at)
FROM login_tokens t JOIN users u ON u.email = t.email
WHERE t.invited_by IS NOT NULL AND t.used_at IS NOT NULL AND t.invited_by <> u.id
GROUP BY t.invited_by, u.id;

-- All of that is history: nobody opens the network to a bell that counts the past.
UPDATE users SET notices_seen_at = CAST(strftime('%s', 'now') AS INTEGER) * 1000;
