-- Invented people and posts for the public demo. Safe to run again: it starts from empty.
DELETE FROM messages;
DELETE FROM conversations;
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
  (6, 'shira@demo.invalid', 'שירה מהדמו', 0),
  (7, 'dana@demo.invalid', 'דנה הבדויה', 0),
  (8, 'ron@demo.invalid', 'רון מהדוגמה', 0),
  (9, 'maya@demo.invalid', 'מאיה הדמיונית', 0),
  (10, 'gil@demo.invalid', 'גיל לדוגמה', 0),
  (11, 'yael@demo.invalid', 'יעל מהסיפור', 0);

-- Where the invented friends appear on the globe. Those left out rest on the cloud.
UPDATE users SET lat = 32.47, lng = 34.97 WHERE id = 1;
UPDATE users SET lat = 32.48, lng = 34.99 WHERE id = 2;
UPDATE users SET lat = 32.52, lng = 34.95 WHERE id = 4;
UPDATE users SET lat = 32.08, lng = 34.78 WHERE id = 7;
UPDATE users SET lat = 52.52, lng = 13.4 WHERE id = 8;
UPDATE users SET lat = 40.71, lng = -74.01 WHERE id = 9;
UPDATE users SET lat = -33.87, lng = 151.21 WHERE id = 11;

-- What the update has to tell: a line from two friends, one who moved, one who just joined.
UPDATE users SET status_text = 'חזרנו מהצפון, הכול ירוק. מי בא לקפה בשבת?', status_at = (unixepoch() - 3600 * 5) * 1000 WHERE id = 2;
UPDATE users SET status_text = 'שבוע ראשון בעבודה החדשה, מתרגלת לרכבת התחתית', status_at = (unixepoch() - 86400 * 2) * 1000 WHERE id = 9;
UPDATE users SET place_at = (unixepoch() - 86400 * 3) * 1000 WHERE id = 8;
UPDATE users SET created_at = (unixepoch() - 86400 * 4) * 1000 WHERE id = 10;

-- The hash of the fixed demo session token. It opens only the guest account, which cannot write.
INSERT INTO sessions (token_hash, user_id, expires_at) VALUES
  ('2a97516c354b68848cdbd8f54a226a0a55b21ed138e207ad6c5cbb9c00aa5aea', 1, 4102444800000);

INSERT INTO friendships (requester_id, addressee_id, status, created_at) VALUES
  (1, 2, 'accepted', 0),
  (1, 3, 'accepted', 0),
  (4, 1, 'accepted', 0),
  (1, 7, 'accepted', 0),
  (8, 1, 'accepted', 0),
  (1, 9, 'accepted', 0),
  (10, 1, 'accepted', 0),
  (1, 11, 'accepted', 0),
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

-- Private messages of the guest, all of them read: the demo cannot store that a visitor read one.
INSERT INTO conversations (id, low_id, high_id, created_at, low_read, high_read) VALUES
  (1, 1, 2, 0, 5, 5),
  (2, 1, 3, 0, 8, 8);

INSERT INTO messages (id, conversation_id, sender_id, body, created_at, removed_at) VALUES
  (1, 1, 2, 'היי! באים מחר לשתילה בגינה הקהילתית?', (unixepoch() - 86400 - 7200) * 1000, NULL),
  (2, 1, 1, 'בטח. מה להביא?', (unixepoch() - 86400 - 7000) * 1000, NULL),
  (3, 1, 2, 'כפפות, ואם יש לכם מעדר קטן. קפה עליי.', (unixepoch() - 86400 - 6900) * 1000, NULL),
  (4, 1, 2, 'זה המקום, ליד שער הכניסה: https://he.wikipedia.org/wiki/גינה_קהילתית', (unixepoch() - 86400 - 6800) * 1000, NULL),
  (5, 1, 1, 'מעולה, נתראה בשמונה.', (unixepoch() - 86400 - 3600) * 1000, NULL),
  (6, 2, 1, 'ראיתי את הפוסט על החתול. הוא עדיין אצלך?', (unixepoch() - 2 * 3600) * 1000, NULL),
  (7, 2, 3, NULL, (unixepoch() - 2 * 3600 + 300) * 1000, (unixepoch() - 2 * 3600 + 360) * 1000),
  (8, 2, 3, 'כן, יושב אצלנו במרפסת ואוכל טונה. תמר חושבת שהוא של השכנים שלה, מחכים לתשובה.', (unixepoch() - 3000) * 1000, NULL);
