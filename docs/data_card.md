# Data Card: NYC Yellow Taxi Zone-Hour Demand

Owner: Raghav U (with input from Rahul Saran Sharma on processing).

## Source
- NYC Taxi and Limousine Commission (TLC) Trip Record Data, Yellow Taxi, monthly Parquet files,
  January 2019 to December 2021: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Taxi Zone Lookup Table (265 zones, 6 boroughs including EWR) from the same page.
- Public data published by the City of New York.

## What we use
- Only `tpep_pickup_datetime` and `PULocationID` from each trip.
- Aggregated to pickups per zone per hour. No trip-level records leave the aggregation step.

## Processing
- Drop pickups outside 2019-01-01 to 2022-01-01 (the raw files contain some wrong timestamps).
- Drop unknown zones 264 and 265.
- Fill zone-hours with no trips with 0.
- TODO: rows in / rows out per year, storage sizes.

## Known shifts in the data
- COVID-19: demand collapse from about 15 March 2020; NY PAUSE from 22 March 2020.
- Phased reopening June-July 2020; Omicron wave December 2021; holidays and seasonality.

## Privacy
- The TLC files contain no names or other direct identifiers. We further reduce detail to zone-hour counts.
- TODO: number of zone-hours with very small counts (fewer than 5 trips) and how they are handled.

## Biases and limitations
- Yellow taxis serve Manhattan far more than other boroughs; outer-borough demand is under-represented
  and increasingly served by green taxis and ride-hailing apps, which are not in this dataset.
- Demand recorded here is completed pickups, not unmet demand.
