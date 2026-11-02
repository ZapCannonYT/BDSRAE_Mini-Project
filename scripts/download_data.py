"""Download NYC TLC Yellow Taxi monthly Parquet files and the zone lookup into data/raw.

Usage:
    python scripts/download_data.py                       # months from configs/default.yaml
    python scripts/download_data.py --start 2019-01 --end 2019-03
    python scripts/download_data.py --dry-run             # list what would be downloaded

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page (public data).
Files already present are skipped, so the script can be re-run after an interruption.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from taxidrift.config import load_config, resolve  # noqa: E402

BASE_URL = "https://d37ci6vzurychx.cloudfront.net"
TRIP_URL = BASE_URL + "/trip-data/yellow_tripdata_{month}.parquet"
ZONE_URL = BASE_URL + "/misc/taxi_zone_lookup.csv"


def months(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def main() -> None:
    cfg = load_config()["data"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=cfg["start_month"])
    p.add_argument("--end", default=cfg["end_month"])
    p.add_argument("--out", default=cfg["raw_dir"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(ZONE_URL, out / "taxi_zone_lookup.csv")]
    jobs += [(TRIP_URL.format(month=m), out / f"yellow_tripdata_{m}.parquet") for m in months(args.start, args.end)]

    for url, dest in jobs:
        if dest.exists():
            print(f"skip  {dest.name} (exists)")
        elif args.dry_run:
            print(f"would download {url}")
        else:
            print(f"get   {url}")
            download(url, dest)
    print(f"done: {len(jobs)} files in {out}")


if __name__ == "__main__":
    main()
