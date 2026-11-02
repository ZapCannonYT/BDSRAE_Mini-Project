"""Message formats shared by the producer, the Spark aggregator and the monitor.

Agree on changes here before changing any one service; all three depend on these fields.
All messages are JSON. Timestamps are ISO-8601 strings in New York local time, as in the TLC data.
"""

# Topic `trips`: one message per trip, keyed by pickup zone id.
TRIP_FIELDS = {
    "pickup_datetime": "string",  # e.g. "2019-01-01T00:46:40"
    "pickup_zone": "int",         # TLC PULocationID, 1-265
}

# Topic `zone_demand`: one message per (zone, hour) window, emitted by the aggregator.
ZONE_DEMAND_FIELDS = {
    "window_start": "string",  # start of the 1-hour window
    "zone": "int",
    "pickups": "int",
}

# Topic `drift_events`: one message per detector alarm, emitted by the monitor.
DRIFT_EVENT_FIELDS = {
    "timestamp": "string",   # stream time of the alarm (window_start of the hour that fired it)
    "detector": "string",    # e.g. "adwin", "psi"
    "scope": "string",       # "city" or a borough name
    "level": "string",       # "warning" or "drift"
    "statistic": "float",    # detector statistic at alarm time (e.g. PSI value)
}

BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "EWR"]
