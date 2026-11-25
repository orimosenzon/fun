-- A feeling about a post or about a comment: one of the five drawn ones (love, funny, surprise,
-- bored, anger) or any single emoji. Each member has at most one per post and one per comment,
-- and choosing again replaces it. Feelings never change the order of anything.
CREATE TABLE reactions (
  post_id INTEGER REFERENCES posts (id),
  comment_id INTEGER REFERENCES comments (id),
  user_id INTEGER NOT NULL REFERENCES users (id),
  kind TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  CHECK ((post_id IS NULL) <> (comment_id IS NULL))
);
CREATE UNIQUE INDEX reactions_post ON reactions (post_id, user_id) WHERE post_id IS NOT NULL;
CREATE UNIQUE INDEX reactions_comment ON reactions (comment_id, user_id) WHERE comment_id IS NOT NULL;
