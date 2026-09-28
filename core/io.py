"""Local artifact I/O and checksums. Inference adapters may use network services."""
import hashlib
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def verify_manifest(root):
    expected = load_json(root/'MANIFEST.json')['files']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and not {'.git', '__pycache__'}.intersection(p.relative_to(root).parts)
              and p.name != '.DS_Store'}
    if actual != set(expected) | {'MANIFEST.json'}:
        raise ValueError('Package file set changed: ' + str(sorted(actual ^ (set(expected) | {'MANIFEST.json'}))))
    for name, digest in expected.items():
        if sha256(root/name) != digest:
            raise ValueError('Checksum mismatch: ' + name)


def external_output(path, root):
    """Keep new runs and reports out of the immutable reference package."""
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root in path.parents:
        raise ValueError('Choose an output path outside the CORE reference folder.')
    return path
