-- The account of the network's own bot: the AI that builds the network and answers members in
-- its own name. It is marked as a bot wherever its name leads, and it may exchange private
-- messages with every member, so that an answer does not have to wait for a friendship. It reads
-- nothing more than any member who is not a friend: posts for friends stay with friends.
ALTER TABLE users ADD COLUMN bot INTEGER NOT NULL DEFAULT 0;
