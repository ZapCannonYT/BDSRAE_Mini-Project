"""Taxi zone lookup: zone id -> borough."""

from __future__ import annotations

import pandas as pd

from taxidrift.config import load_config, resolve

# Zones 264 and 265 are "Unknown" / "N/A" in the lookup and are dropped from all analysis.
UNKNOWN_ZONES = {264, 265}


def load_zone_lookup(path: str | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns zone, borough, zone_name, service_zone."""
    path = path or resolve(load_config()["data"]["zone_lookup"])
    df = pd.read_csv(path)
    df = df.rename(columns={"LocationID": "zone", "Borough": "borough", "Zone": "zone_name",
                            "service_zone": "service_zone"})
    return df[~df["zone"].isin(UNKNOWN_ZONES)].reset_index(drop=True)


def zone_to_borough(path: str | None = None) -> dict[int, str]:
    df = load_zone_lookup(path)
    return dict(zip(df["zone"], df["borough"]))
