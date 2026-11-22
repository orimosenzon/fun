-- Whether Google also let the network read the admin's contacts, to complete an address while
-- an invitation is typed. The contacts themselves are never stored.
ALTER TABLE mailers ADD COLUMN contacts INTEGER NOT NULL DEFAULT 0;
