-- Google's anonymous id for someone who chose to link their account. Nothing else from Google is kept.
ALTER TABLE users ADD COLUMN google_sub TEXT;
CREATE UNIQUE INDEX users_google_sub ON users (google_sub);
