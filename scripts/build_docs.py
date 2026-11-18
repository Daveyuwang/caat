"""Build the documentation and its downloadable fixtures from the checkout."""

import logging
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from mkdocs.commands.build import build
from mkdocs.config import load_config
from mkdocs.exceptions import MkDocsException


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "replay/docs/site"
SOURCES = (
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "LICENSE",
    "action.yml",
    "docs/**/*.md",
    "docs/site.css",
    "docs/figures/*.svg",
    "examples/**/*.csv",
    "examples/**/*.jsonl",
    "examples/**/*.toml",
    "examples/**/*.ipynb",
    "results/**/*.csv",
    "results/**/*.json",
    "scripts/prepare_digit_example.py",
)


def stage_sources(directory, repo_url):
    # Mirror the checkout layout so links to examples and benchmarks work as written.
    for pattern in SOURCES:
        for source in ROOT.glob(pattern):
            relative = source.relative_to(ROOT)
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == Path("README.md"):
                # The PyPI README uses GitHub URLs for the same bundled files.
                text = source.read_text(encoding="utf-8")
                for kind in ("blob", "raw"):
                    text = text.replace(f"]({repo_url}/{kind}/main/", "](")
                badge_prefix = f"[![CI]({repo_url}/actions/"
                text = "\n".join(
                    line
                    for line in text.split("\n")
                    if not line.startswith(badge_prefix)
                )
                target.write_text(text, encoding="utf-8")
            else:
                copyfile(source, target)


def main():
    logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
    with TemporaryDirectory(prefix="caat-docs-") as temporary:
        directory = Path(temporary)
        config = load_config(str(ROOT / "docs/site.yml"), docs_dir=str(directory))
        if Path(config.site_dir).resolve() != SITE:
            raise SystemExit(f"Documentation output must resolve to {SITE}")
        stage_sources(directory, config.repo_url.rstrip("/"))
        build(config)


if __name__ == "__main__":
    try:
        main()
    except MkDocsException as exc:
        raise SystemExit(str(exc)) from None
