-- Invented people and posts for the public demo. Safe to run again: it starts from empty.
DELETE FROM comments;
DELETE FROM posts;
DELETE FROM friendships;
DELETE FROM sessions;
DELETE FROM users;

INSERT INTO users (id, email, name, created_at) VALUES
  (1, 'guest@demo.invalid', 'אורח בדמו', 0),
  (2, 'noa@demo.invalid', 'נועה הדמיונית', 0),
  (3, 'yoav@demo.invalid', 'יואב הבדוי', 0),
  (4, 'tamar@demo.invalid', 'תמר הדוגמה', 0),
  (5, 'amit@demo.invalid', 'עמית לדוגמה', 0),
  (6, 'shira@demo.invalid', 'שירה מהדמו', 0);

-- The hash of the fixed demo session token. It opens only the guest account, which cannot write.
INSERT INTO sessions (token_hash, user_id, expires_at) VALUES
  ('2a97516c354b68848cdbd8f54a226a0a55b21ed138e207ad6c5cbb9c00aa5aea', 1, 4102444800000);

INSERT INTO friendships (requester_id, addressee_id, status, created_at) VALUES
  (1, 2, 'accepted', 0),
  (1, 3, 'accepted', 0),
  (4, 1, 'accepted', 0),
  (5, 1, 'pending', 0),
  (1, 6, 'pending', 0);

INSERT INTO posts (id, author_id, body, youtube_id, audience, created_at) VALUES
  (1, 3, 'מישהו מכיר חשמלאי טוב באזור? נשרף לנו הלוח בבית.', NULL, 'friends', (unixepoch() - 86400 * 3) * 1000),
  (2, 4, 'סרט קצר וחמוד לשבת בבוקר עם הילדים:
https://www.youtube.com/watch?v=aqz-KE-bpKQ', 'aqz-KE-bpKQ', 'public', (unixepoch() - 86400 * 2) * 1000),
  (3, 2, 'מחר בבוקר פוגשים בגינה הקהילתית לשתילה. מביאים כפפות, יהיה קפה.', NULL, 'public', (unixepoch() - 86400) * 1000),
  (4, 3, 'נמצא חתול ג''ינג''י ליד תחנת האוטובוס. רגוע, עם קולר כחול. מי מכיר?', NULL, 'friends', (unixepoch() - 3 * 3600) * 1000);

INSERT INTO comments (id, post_id, author_id, body, created_at, removed_by, parent_id) VALUES
  (1, 1, 2, 'עבד אצלנו מישהו מצוין, אשלח לך את הטלפון בפרטי.', (unixepoch() - 86400 * 3 + 1800) * 1000, NULL, NULL),
  (2, 3, 3, 'אגיע עם האחיינים!', (unixepoch() - 80000) * 1000, NULL, NULL),
  (3, 3, 4, NULL, (unixepoch() - 79000) * 1000, 'author', NULL),
  (4, 3, 5, 'כאן יש הסבר טוב על שתילה בחורף, גם מי שלא כתב את הפוסט יכול לשים קישור לחיץ: https://he.wikipedia.org/wiki/גינה_קהילתית', (unixepoch() - 78000) * 1000, NULL, NULL),
  (5, 3, 2, 'תודה! מעולה.', (unixepoch() - 77000) * 1000, NULL, 4),
  (6, 4, 4, 'נראה לי שזה החתול של השכנים ממול, בודקת.', (unixepoch() - 2 * 3600) * 1000, NULL, NULL);
