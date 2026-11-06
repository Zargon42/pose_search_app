"""Install the official HumanArt detector and RTMPose body model for local inference."""
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.model_assets import MANGA_MODEL_DIR, MANGA_MODELS


def install(item):
    name, url = item
    destination = MANGA_MODEL_DIR / name
    if not destination.exists():
        archive_path = destination.with_suffix('.zip.download')
        pending = destination.with_suffix('.onnx.part')
        try:
            with urllib.request.urlopen(url, timeout=60) as source, archive_path.open('wb') as output:
                size, reported = 0, 0
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
                    size += len(chunk)
                    if size - reported >= 16 * 1024 * 1024:
                        print(f'{name}: downloaded {size // (1024 * 1024)} MiB', flush=True)
                        reported = size
            with zipfile.ZipFile(archive_path) as archive:
                models = [entry for entry in archive.infolist() if entry.filename.endswith('.onnx') and not entry.is_dir()]
                if len(models) != 1 or not 0 < models[0].file_size < 500 * 1024 * 1024:
                    raise ValueError(f'Unexpected model archive contents: {name}')
                with archive.open(models[0]) as source, pending.open('wb') as output:
                    shutil.copyfileobj(source, output)
            pending.replace(destination)
        finally:
            archive_path.unlink(missing_ok=True)
            pending.unlink(missing_ok=True)
    with destination.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    print(f'{name}: ready ({destination.stat().st_size // (1024 * 1024)} MiB)', flush=True)
    return name, {'source': url, 'sha256': digest}


def main():
    MANGA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        manifest = dict(executor.map(install, MANGA_MODELS.items()))
    (MANGA_MODEL_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Local manga models installed in {MANGA_MODEL_DIR}')


if __name__ == '__main__':
    main()
