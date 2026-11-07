-- Who sent an invitation. Someone who joins through it starts as their friend.
ALTER TABLE login_tokens ADD COLUMN invited_by INTEGER REFERENCES users (id);
