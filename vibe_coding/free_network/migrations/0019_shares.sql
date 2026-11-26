-- A share is a post that quotes another post, under the name of whoever shared it.
-- shared_id is always the original post, also when the share was made from someone else's
-- share, and then shared_via is that share. So the writer of a post finds everyone who
-- shared it in one place. A member shares a post once, and deleting the share frees it.
ALTER TABLE posts ADD COLUMN shared_id INTEGER REFERENCES posts (id);
ALTER TABLE posts ADD COLUMN shared_via INTEGER REFERENCES posts (id);
CREATE UNIQUE INDEX posts_share_once ON posts (author_id, shared_id) WHERE shared_id IS NOT NULL AND deleted_at IS NULL;
CREATE INDEX posts_shared ON posts (shared_id) WHERE shared_id IS NOT NULL AND deleted_at IS NULL;
