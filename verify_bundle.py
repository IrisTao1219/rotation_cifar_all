"""Verify the packaged files without third-party dependencies or a GPU."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / '文件清单SHA256.json').read_text(encoding='utf-8'))
failed = []
for record in manifest['files']:
    path = root / record['path']
    if not path.is_file():
        failed.append(f"缺失: {record['path']}")
        continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if path.stat().st_size != record['size'] or digest.hexdigest() != record['sha256']:
        failed.append(f"内容改变: {record['path']}")

if failed:
    print('\n'.join(failed))
    print(f'校验失败: {len(failed)} 个文件。')
    sys.exit(1)
print(f"校验通过: {len(manifest['files'])} 个文件，大小与 SHA256 全部一致。")
