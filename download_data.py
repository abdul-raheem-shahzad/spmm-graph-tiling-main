"""Download the public graph datasets used in the experiments.

All files are plain-text edge lists (no pickles). Run once:

    python download_data.py
"""
from __future__ import annotations

import gzip
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"

SOURCES = {
    # name: (url, kind)
    "cora": ("https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz", "tgz"),
    "citeseer": ("https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz", "tgz"),
    "pubmed": ("https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz", "tgz"),
    "ca-HepPh": ("https://snap.stanford.edu/data/ca-HepPh.txt.gz", "gz"),
    "email-Enron": ("https://snap.stanford.edu/data/email-Enron.txt.gz", "gz"),
}


def fetch(url: str, dest: Path) -> None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.URLError:
        # Python's certificate bundle can miss the system/proxy root certificate.
        # curl uses the operating system's certificate store (TLS is still verified).
        subprocess.run(["curl", "-sSfL", "-m", "300", "-o", str(dest), url], check=True)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, kind) in SOURCES.items():
        out_dir = RAW / name
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[skip] {name} already present")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        archive = out_dir / url.rsplit("/", 1)[1]
        print(f"[get ] {name}: {url}")
        fetch(url, archive)
        if kind == "tgz":
            with tarfile.open(archive) as t:
                t.extractall(out_dir, filter="data")
        else:
            with gzip.open(archive, "rb") as src, open(archive.with_suffix(""), "wb") as dst:
                shutil.copyfileobj(src, dst)
        archive.unlink()
        print(f"[ ok ] {name}")


if __name__ == "__main__":
    main()
