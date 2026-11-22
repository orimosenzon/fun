-- A post or a comment can be rewritten by whoever wrote it. The time of the last change is kept,
-- so readers are told that the words are not the first ones.
ALTER TABLE posts ADD COLUMN edited_at INTEGER;
ALTER TABLE comments ADD COLUMN edited_at INTEGER;
