#!/usr/bin/env python3
"""
Pardespedia CLI — manage pages in pardespedia.info wiki.

Usage:
  pardespedia.py get <title>
  pardespedia.py edit <title> <content_file> [--summary "..."]
  pardespedia.py create <title> <content_file> [--summary "..."]
  pardespedia.py search <query>
  pardespedia.py list [--prefix <prefix>]
  pardespedia.py cats <title>
"""

import argparse
import sys
from wiki_client import WikiClient


def cmd_get(client, args):
    page = client.get_page(args.title)
    if not page["exists"]:
        print(f"Page does not exist: {args.title}")
        print(f"URL: {page['url']}")
        return
    print(f"=== {page['title']} ===")
    print(f"URL: {page['url']}\n")
    print(page["wikitext"])


def cmd_edit(client, args):
    client.login()
    with open(args.file) as f:
        content = f.read()
    summary = args.summary or ""
    client.edit_page(args.title, content, summary=summary)


def cmd_search(client, args):
    results = client.search(args.query)
    if not results:
        print("No results found.")
        return
    for r in results:
        print(f"• {r['title']}")
        print(f"  {r['url']}")


def cmd_list(client, args):
    pages = client.list_pages(prefix=args.prefix or "")
    for p in pages:
        print(f"• {p['title']}")
        print(f"  {p['url']}")


def cmd_cats(client, args):
    cats = client.get_categories(args.title)
    if not cats:
        print("No categories.")
        return
    for c in cats:
        print(f"• {c}")


def main():
    parser = argparse.ArgumentParser(description="Pardespedia wiki manager")
    sub = parser.add_subparsers(dest="command")

    p_get = sub.add_parser("get", help="Get page content")
    p_get.add_argument("title")

    p_edit = sub.add_parser("edit", help="Edit a page from a file")
    p_edit.add_argument("title")
    p_edit.add_argument("file", help="Path to file with wikitext content")
    p_edit.add_argument("--summary", default="", help="Edit summary")

    p_create = sub.add_parser("create", help="Create a new page from a file")
    p_create.add_argument("title")
    p_create.add_argument("file", help="Path to file with wikitext content")
    p_create.add_argument("--summary", default="", help="Edit summary")

    p_search = sub.add_parser("search", help="Search pages")
    p_search.add_argument("query")

    p_list = sub.add_parser("list", help="List pages")
    p_list.add_argument("--prefix", default="", help="Filter by title prefix")

    p_cats = sub.add_parser("cats", help="List categories of a page")
    p_cats.add_argument("title")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = WikiClient()

    dispatch = {
        "get": cmd_get,
        "edit": cmd_edit,
        "create": cmd_edit,  # same logic as edit
        "search": cmd_search,
        "list": cmd_list,
        "cats": cmd_cats,
    }
    dispatch[args.command](client, args)


if __name__ == "__main__":
    main()
