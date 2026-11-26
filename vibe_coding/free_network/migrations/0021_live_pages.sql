-- A live page: a web page that runs inside a post, in a frame of its own. The server reads the
-- page once, when the post is written or its page changes, so that readers see its title and
-- picture without their browser contacting the site. The page itself loads only on a click.
--
-- page_framable is 0 when the site says it may not be shown inside another page, and then
-- readers get only a link that opens it in a window of its own.
ALTER TABLE posts ADD COLUMN page_url TEXT;
ALTER TABLE posts ADD COLUMN page_title TEXT;
ALTER TABLE posts ADD COLUMN page_text TEXT;
ALTER TABLE posts ADD COLUMN page_picture TEXT;
ALTER TABLE posts ADD COLUMN page_framable INTEGER;
