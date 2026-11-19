"""Small shared helpers: paths, hashing, JSON IO with immutability guards."""
import datetime
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(value):
    if isinstance(value, str):
        value = value.encode()
    elif not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(ROOT / path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def write(path, value, immutable=False):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False) + '\n'
    if immutable and p.exists():
        if p.read_text() != content:
            raise RuntimeError(f'Refusing to overwrite immutable artifact {p}')
        return
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(content)
    tmp.replace(p)


def source_hash():
    """Hash of the implementation that produces model outputs (cache key component)."""
    names = ['common.py', 'data.py', 'model.py', 'scoring.py', 'edits.py', 'evaluate.py']
    return digest(b''.join(n.encode() + (ROOT / 'src' / n).read_bytes() for n in names if (ROOT / 'src' / n).exists()))


def git_revision(directory=ROOT):
    try:
        return subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None
