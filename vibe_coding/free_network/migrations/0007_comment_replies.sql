-- A reply points at a top-level comment. Replies to replies join the same thread, so the tree
-- never goes deeper than two levels.
ALTER TABLE comments ADD COLUMN parent_id INTEGER REFERENCES comments (id);
