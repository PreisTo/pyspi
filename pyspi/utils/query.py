from dataclasses import dataclass
from datetime import datetime

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astroquery.heasarc import Heasarc


class SelectData:
    def __init__(self, position: SkyCoord, radius: u.Quantity = 5 * u.deg):
        self._heasarc = Heasarc()
        self._position = position
        self._radius = radius
        self._min_good_spi = 100

        self._query_region()

    def filter(
        self,
        min_good_spi: float = 100,
        scw_id_range: (int, int) = (None, None),
        time_range: (float, float) = (None, None),
        obs_type: str = None,
        radius: u.Quantity = None,
    ):
        """
        Filter the data based on the given criteria.

        :param min_good_spi: Minimum number of good SPI time required (default: 100)
        :type min_good_spi: float
        :param scw_id_range: Tuple of (min_scw_id, max_scw_id) to filter by SCW ID range
        (default: (None, None))
        :type scw_id_range: (int, int)
        :param time_range: Tuple of (start_time, end_time) to filter by time range must
        be in MJD (default: (None, None))
        :type time_range: (float, float)
        :param obs_type: Observation type to filter by (default: None)
        :type obs_type: str
        :param radius: Radius to filter by (default: None, uses the initial radius)
        :type radius: u.Quantity
        """
        self._min_good_spi = min_good_spi
        if radius is not None:
            self._radius = radius

        self._query_region()  # we requery to ensure radius and min_good_spi are correct

        tab = self._table.copy()

        mask = np.ones(len(tab), dtype=bool)

        scw_id = np.asarray(tab["scw_id"].filled("-1"), dtype=np.int64)

        if scw_id_range[0] is not None:
            mask &= scw_id >= scw_id_range[0]
        if scw_id_range[1] is not None:
            mask &= scw_id <= scw_id_range[1]

        if time_range[0] is not None:
            mask &= tab["end_date"] >= time_range[0]
        if time_range[1] is not None:
            mask &= tab["start_date"] <= time_range[1]

        if obs_type is not None:
            mask &= tab["obs_type"] == obs_type

        tab = tab[mask]
        self._table = tab

    def _query_region(self, **kwargs):
        tab = self._heasarc.query_region(
            self._position, catalog="intscw", radius=self._radius, **kwargs
        )
        tab = self._default_selections(tab)
        self._table = tab

    def _default_selections(self, tab):
        tab = tab[tab["scw_type"] == "pointing"]
        mask = tab["good_spi"] >= self._min_good_spi
        tab = tab[mask]
        return tab

    @property
    def table(self):
        return self._table


@dataclass
class Pointing:
    """
    Dataclass that represents a single Pointing. Contains its coordinates, scw_id,
    index and its cluster may be added later
    """

    scw_id: str
    sky_coords: SkyCoord
    start_time: datetime
    index: int
    cluster: Cluster = None

    def angle_between_three_pointings(
        self,
        pointing2: "Pointing",
        pointing3: "Pointing",
        angle_weight: float,
        time_weight: float,
    ) -> float:
        """Returns the angle between three different poitings
        Angle between the vector self -> poiting2 and vector self -> pointing3
        """

        origin = SkyCoord(
            self.sky_coords.ra.deg,
            (self.sky_coords.dec.deg % 180.0) - 90.0,
            frame="icrs",
            unit="deg",
        )
        origin_Frame = SkyOffsetFrame(origin=origin)

        p1 = self.sky_coords.transform_to(origin_Frame)
        p2 = pointing2.sky_coords.transform_to(origin_Frame)
        p3 = pointing3.sky_coords.transform_to(origin_Frame)

        a_a = p2.lon.deg - p3.lon.deg
        a_d2 = abs(p1.lat.deg - p2.lat.deg) * angle_weight
        a_d3 = abs(p1.lat.deg - p3.lat.deg) * angle_weight

        t_d2 = (
            (pointing2.start_time - self.start_time).total_seconds()
            / 86400
            * time_weight
        )
        t_d3 = (
            (pointing3.start_time - self.start_time).total_seconds()
            / 86400
            * time_weight
        )

        return np.arccos(
            np.clip(
                (a_d2 * a_d3 * np.cos(np.deg2rad(a_a)) + t_d2 * t_d3)
                / np.linalg.norm([a_d2, t_d2])
                / np.linalg.norm([a_d3, t_d3]),
                -1.0,
                1.0,
            )
        )
