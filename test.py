#! /usr/bin/env python3

from pyspi.utils.query import SelectData
from pyspi.clustering.pointing import PointingClusters
from astropy.coordinates import SkyCoord
from datetime import datetime, timedelta
from astropy.time import Time

if __name__ == "__main__":
    sd = SelectData(SkyCoord.from_name("G 21.5-0.9"))
    start = datetime(2008, 1, 1)
    sd.filter(
        time_range=(
            float(Time(start).mjd),
            float(Time(start + timedelta(days=31 * 2)).mjd),
        )
    )
    print(sd.table)

    pc = PointingClusters(sd)
