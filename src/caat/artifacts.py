"""Prepare related report files before replacing their destinations."""

from contextlib import contextmanager
from os.path import samestat
from pathlib import Path
from shutil import copy2
from stat import S_IREAD, S_IWRITE
from tempfile import NamedTemporaryFile


def _temporary(target):
    with NamedTemporaryFile(prefix=".caat-", dir=target.parent, delete=False) as handle:
        return Path(handle.name)


def _restore(published, backups):
    failures = []
    for target, identity in reversed(published):
        backup = backups.get(target)
        try:
            try:
                current = target.lstat()
            except FileNotFoundError:
                continue
            if not samestat(current, identity):
                continue
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                backup.replace(target)
        except OSError as error:
            message = f"{target}: {error}"
            if backup is not None:
                message += f". Previous output saved at {backups.pop(target)}"
            failures.append(message)
    return failures


def _publish(outputs):
    backups = {}
    published = []
    try:
        # The last replacement has no later file that could fail.
        for target, _ in outputs[:-1]:
            if target.exists():
                backup = backups[target] = _temporary(target)
                copy2(target, backup)
        for target, draft in outputs:
            identity = draft.stat()
            draft.replace(target)
            published.append((target, identity))
    except OSError as error:
        failures = _restore(published, backups)
        if failures:
            raise OSError(
                f"{error}. Could not restore: {' | '.join(failures)}"
            ) from error
        raise
    finally:
        for backup in backups.values():
            try:
                backup.unlink(missing_ok=True)
            except PermissionError:
                # A private copy of a read-only file can be read-only on Windows.
                backup.chmod(S_IREAD | S_IWRITE)
                backup.unlink(missing_ok=True)


@contextmanager
def staged_outputs(paths):
    """Yield temporary paths and publish them in order after successful writes."""
    drafts = {}
    outputs = []
    try:
        for path in paths:
            target = path.resolve()
            if target.exists() and not target.is_file():
                raise ValueError(f"output needs a regular file: {path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            draft = drafts[path] = _temporary(target)
            outputs.append((target, draft))
        yield drafts
        _publish(outputs)
    finally:
        for draft in drafts.values():
            draft.unlink(missing_ok=True)
