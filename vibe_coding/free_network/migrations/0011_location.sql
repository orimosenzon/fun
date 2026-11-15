-- Where a member chose to appear on their friends' globe. NULL means they are not sharing:
-- switching sharing off erases the place, it is not kept hidden.
ALTER TABLE users ADD COLUMN lat REAL;
ALTER TABLE users ADD COLUMN lng REAL;
