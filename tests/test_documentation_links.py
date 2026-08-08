import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
import unittest


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
IGNORED_DIRECTORIES = {".git", ".venv", "data"}


def markdown_files():
    for path in ROOT.rglob("*.md"):
        if not IGNORED_DIRECTORIES.intersection(path.relative_to(ROOT).parts):
            yield path


class DocumentationLinkTests(unittest.TestCase):
    def test_relative_markdown_links_exist(self):
        missing = []
        for document in markdown_files():
            source = document.read_text(encoding="utf-8")
            for raw_target in MARKDOWN_LINK.findall(source):
                target = raw_target.strip().strip("<>")
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith(("#", "/")):
                    continue
                relative_path = unquote(parsed.path)
                if not relative_path:
                    continue
                resolved = (document.parent / relative_path).resolve()
                if not resolved.exists():
                    missing.append(
                        f"{document.relative_to(ROOT)} -> {target}"
                    )
        self.assertEqual(missing, [], "깨진 문서 링크:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
