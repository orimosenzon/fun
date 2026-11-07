-- The size of each stored image, so the server can stop before storage starts to cost money.
ALTER TABLE images ADD COLUMN bytes INTEGER NOT NULL DEFAULT 0;
