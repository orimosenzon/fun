-- Profile pictures are public, like the name beside them. An image used here belongs to no post.
ALTER TABLE users ADD COLUMN avatar_id TEXT REFERENCES images (id);
ALTER TABLE users ADD COLUMN cover_id TEXT REFERENCES images (id);
