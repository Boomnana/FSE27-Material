from pathlib import Path
import json
import sys


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    failures = []
    for item in manifest['files']:
        path = root / item['path']
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            failures.append({'path': item['path'], 'error': 'outside artifact'})
            continue
        if not path.is_file():
            failures.append({'path': item['path'], 'error': 'missing'})
            continue
        if path.stat().st_size != item['bytes']:
            failures.append({'path': item['path'], 'error': 'file size mismatch'})
    print(json.dumps({'checked_files': len(manifest['files']), 'failures': failures}, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
