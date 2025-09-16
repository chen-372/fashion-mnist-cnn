import argparse, hashlib
from pathlib import Path
import requests

FILES = [
    ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
]
BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='./data_raw')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {out.resolve()} ...")

    for fname, expected_md5 in FILES:
        url = BASE + fname
        dest = out / fname
        print(f"* {fname} <- {url}")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with dest.open('wb') as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk:
                    f.write(chunk)
        got_md5 = md5(dest)
        ok = (got_md5 == expected_md5)
        print(f"  MD5: {got_md5} {'OK' if ok else 'MISMATCH!'}")
        if not ok:
            raise SystemExit(f"Checksum mismatch for {fname}. Expected {expected_md5}, got {got_md5}")
    print("All files downloaded and verified.")

if __name__ == '__main__':
    main()
