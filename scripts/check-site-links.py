"""Check local links and fragments in a built Hugo site."""
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class PageLinks(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids = set()
        self.links = set()
        self.base = ""
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a":
            if "href" in attrs:
                self.links.add(attrs["href"])
            if "name" in attrs:
                self.ids.add(attrs["name"])
        if tag == "base":
            self.base = attrs.get("href", "")


def check(site):
    pages = {p: PageLinks(p.read_text()) for p in site.rglob("*.html")}
    origin = urlsplit(pages[site / "index.html"].base).netloc
    broken = []
    checked = 0
    for path, page in pages.items():
        for href in sorted(page.links):
            url = urlsplit(urljoin(page.base, href))
            if url.scheme not in ("http", "https") or url.netloc != origin:
                continue
            target = site / unquote(url.path).lstrip("/")
            if target.is_dir():
                target /= "index.html"
            checked += 1
            if not target.is_file():
                broken.append((path, href))
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                broken.append((path, href))
    for path, href in broken:
        print(f"{path.relative_to(site)} -> {href}")
    print(f"{len(pages)} HTML pages, {checked} local references, {len(broken)} broken")
    return bool(broken)


if __name__ == "__main__":
    sys.exit(check(Path(sys.argv[1] if len(sys.argv) > 1 else "public")))
