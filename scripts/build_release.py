"""Build and check the release archives from a clean checkout."""

import gzip
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


def normalize_sdist(path: Path, epoch: int) -> None:
    staged = path.with_suffix(".tmp")
    with (
        tarfile.open(path, "r:gz") as source,
        staged.open("wb") as destination,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=destination, mtime=epoch
        ) as packed,
        tarfile.open(fileobj=packed, mode="w|", format=tarfile.PAX_FORMAT) as archive,
    ):
        for member in source:
            member.mtime = epoch
            member.uid = member.gid = 0
            member.uname = member.gname = "root"
            for name in ("mtime", "atime", "ctime", "uid", "gid", "uname", "gname"):
                member.pax_headers.pop(name, None)
            if member.isfile():
                content = source.extractfile(member)
                assert content is not None
                with content:
                    archive.addfile(member, content)
            else:
                archive.addfile(member)
    staged.replace(path)


def main() -> None:
    timestamp = (
        os.environ.get("SOURCE_DATE_EPOCH")
        or subprocess.check_output(
            ["git", "log", "-1", "--format=%ct"], cwd=ROOT, text=True
        ).strip()
    )
    epoch = int(timestamp)
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch))
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output, prefix=".release-") as directory:
        build = [
            sys.executable,
            "-m",
            "build",
            "--dependency-constraints-txt",
            str(ROOT / "scripts/release-constraints.txt"),
            "--outdir",
            directory,
        ]
        subprocess.run([*build, "--sdist"], cwd=ROOT, env=env, check=True)
        (sdist,) = Path(directory).glob("*.tar.gz")
        normalize_sdist(sdist, epoch)
        subprocess.run([*build, "--wheel", str(sdist)], cwd=ROOT, env=env, check=True)
        (wheel,) = Path(directory).glob("*.whl")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "twine",
                "check",
                "--strict",
                str(wheel),
                str(sdist),
            ],
            cwd=ROOT,
            check=True,
        )
        for artifact in (wheel, sdist):
            artifact.replace(output / artifact.name)


if __name__ == "__main__":
    main()
