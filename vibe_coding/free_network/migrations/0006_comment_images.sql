-- A comment can carry one image. Whoever can see the post can see it.
ALTER TABLE images ADD COLUMN comment_id INTEGER REFERENCES comments (id);
CREATE INDEX images_comment ON images (comment_id);
