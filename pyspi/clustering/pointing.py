import numpy as np
from astropy.time import Time
from pyspi.clustering.cluster_query import ClusteredQuery
import os
import pickle


def save_clusters(pointings, folder):
    if not os.path.exists(f"./{folder}") and not os.path.exists(
        folder
    ):  # I hope this doesnt break anything
        os.mkdir(folder)
    with open(f"{folder}/pointings.pickle", "wb") as f:
        pickle.dump(pointings, f)


def load_clusters(folder, outliers=None):
    with open(f"{folder}/pointings.pickle", "rb") as f:
        pointings = pickle.load(f)

    if outliers is not None:
        pointings = remove_outlier_clusters(pointings, outliers)

    return pointings


def remove_outlier_clusters(pointings, outliers):
    new_pointings = []
    for cluster in pointings:
        was_in = False
        for cluster2 in outliers:
            assert (
                cluster2[1] != cluster[0][0]
            ), "Outliers do not correspond to Clusters"
            if cluster2[0] == cluster[0][0]:
                was_in = True
                assert (
                    cluster2[1] == cluster[1][0]
                ), "Outliers do not correspond to Clusters"
                # continue
        if not was_in:
            new_pointings.append(cluster)
    return tuple(new_pointings)


def load_outliers(path, id):
    with open(path, "rb") as f:
        outliers = pickle.load(f)[id]
    return outliers


class PointingClusters:  # add min time diff
    def __init__(
        self,
        query_table,
        min_angle_dif,
        max_angle_dif,
        max_time_dif,
        cluster_size_range,
        random_angle_dif_range=None,
        center_ra=83.6333,
        center_dec=22.0144,
    ):
        self._query_table = query_table
        self._min_angle_dif = min_angle_dif
        self._max_angle_dif = max_angle_dif
        self._max_time_dif = max_time_dif
        self._cluster_size_range = cluster_size_range
        self._random_angle_dif_range = random_angle_dif_range

        pointings = []
        self._get_scw_ids(print_results=False)
        cq = ClusteredQuery(
            self._scw_ids,
            angle_weight=0.0,
            time_weight=1.0 / self._max_time_dif,
            max_distance=1.0,
            min_ang_distance=self._min_angle_dif,
            max_ang_distance=self._max_angle_dif,
            cluster_size_range=self._cluster_size_range,
            failed_improvements_max=3,
            suboptimal_cluster_size=min(
                max(1, self._cluster_size_range[0]), self._cluster_size_range[1] - 1
            ),
            close_suboptimal_cluster_size=min(
                max(1, self._cluster_size_range[0]), self._cluster_size_range[1] - 1
            ),
        ).get_clustered_scw_ids()

        for size in range(self._cluster_size_range[0], self._cluster_size_range[1] + 1):
            for cluster in cq[size]:
                pointings_cluster = []
                for pointing in cluster:
                    for i, scw in enumerate(self._scw_ids):
                        if scw[0] == pointing:
                            pointings_cluster.append((pointing, self._scw_paths[i]))
                            break
                pointings.append(tuple(pointings_cluster))

        self.pointings = tuple(pointings)

    def _get_scw_ids(self, print_results=False):

        self._query_table.sort("scw_id")
        scw_ids_all = np.array(self._query_table["scw_id"])
        ras = np.array(self._query_table["ra"])
        decs = np.array(self._query_table["dec"])
        start_dates = np.array(
            Time(self._query_table["start_date"], format="mjd").datetime
        )

        self._scw_ids = np.array([scw_ids_all, ras, decs, start_dates]).T
