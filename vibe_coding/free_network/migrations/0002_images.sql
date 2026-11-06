-- An image belongs to its uploader until a post claims it. After that, whoever can see the
-- post can see the image.
CREATE TABLE images (
  id TEXT PRIMARY KEY,
  owner_id INTEGER NOT NULL REFERENCES users (id),
  post_id INTEGER REFERENCES posts (id),
  position INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL
);
CREATE INDEX images_post ON images (post_id, position);
CREATE INDEX images_owner ON images (owner_id, created_at);
