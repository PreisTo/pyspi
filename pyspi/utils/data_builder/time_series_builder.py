import numpy as np
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion
from threeML.utils.time_series.event_list import EventListWithLiveTime

from .science_window import SPISWFile, SPISWFileGRB


class TimeSeriesBuilderSPI(TimeSeriesBuilder):
    def __init__(
        self,
        name,
        time_series,
        response=None,
        poly_order=-1,
        verbose=True,
        restore_poly_fit=None,
        container_type=BinnedSpectrumWithDispersion,
        **kwargs,
    ):
        """
        Class to build the time_series for SPI. Inherited from the 3ML
        TimeSeriesBuilder with added class methods to build the object
        for SPI datafiles.
        :param name: Name of the tsb
        :param time_series: Timeseries with the data
        :param response: Response object
        :param poly_order: poly order for the polynominal fitting
        :param verbose: Verbose?
        :param restore_poly_fit: Path to a file with the poly bkg fits
        :param containter_type: ContainerType for spectrum
        :returns: Object
        """

        super().__init__(
            name,
            time_series,
            response=response,
            poly_order=poly_order,
            unbinned=False,
            verbose=verbose,
            restore_poly_fit=restore_poly_fit,
            container_type=container_type,
            **kwargs,
        )

    @classmethod
    def from_spi_grb(
        cls,
        name,
        det,
        time_of_grb,
        ebounds=None,
        response=None,
        sgl_type=None,
        restore_background=None,
        poly_order=0,
        verbose=True,
    ):
        """
        Class method to build the time_series_builder for a given GRB time

        :param name: Name of object
        :param det: Which det?
        :param ebounds: Output ebounds for analysis.
        :param time_of_grb: Astropy time object with the time of the GRB (t0)
        :param response: InstrumenResponse Object
        :param sgl_type: What kind of sinlge events? Standard single events?
                         PSD events? Or both?
        :param restore_background: File to restore bkg
        :param poly_order: Which poly_order? -1 gives automatic determination
        :param verbose: Verbose?

        :returns: Initalized TimeSeriesBuilder object
        """

        assert (
            ebounds is not None or response is not None
        ), "You have to either specify ebounds or input a response object."

        if ebounds is not None and response is not None:
            assert np.all(
                response.ebounds == ebounds
            ), "Ebounds do not match the ones in the response object!"
        elif response is not None:
            ebounds = response.ebounds

        spi_grb_setup = SPISWFileGRB(det, ebounds, time_of_grb, sgl_type=sgl_type)

        event_list = EventListWithLiveTime(
            arrival_times=spi_grb_setup.gti_times,
            measurement=spi_grb_setup.gti_energy_bins,
            n_channels=spi_grb_setup.n_channels,
            start_time=spi_grb_setup.time_start,
            stop_time=spi_grb_setup.time_stop,
            live_time=spi_grb_setup.livetimes_per_interval,
            live_time_starts=spi_grb_setup.deadtime_bin_starts,
            live_time_stops=spi_grb_setup.deadtime_bin_stops,
            first_channel=0,
            instrument=spi_grb_setup.det_name,
            mission=spi_grb_setup.mission,
            verbose=verbose,
            edges=spi_grb_setup.ebounds,
        )

        return cls(
            name,
            event_list,
            poly_order=poly_order,
            verbose=verbose,
            restore_poly_fit=restore_background,
            response=response,
            container_type=BinnedSpectrumWithDispersion,
        )

    @classmethod
    def from_spi_constant_pointing(
        cls,
        det=0,
        pointing_id="118900570010",
        ebounds=None,
        response=None,
    ):
        """
        Class method to build the time_series_builder for a given pointing id

        :param det: Which det?
        :param ebounds: Output ebounds for analysis.
        :param pointing_id: Pointing ID
        :param response: InstrumenResponse Object

        :returns: Initalized TimeSeriesBuilder object
        """

        assert (
            ebounds is not None or response is not None
        ), "You have to either specify ebounds or input a response object."

        if ebounds is not None and response is not None:
            assert np.all(
                response.ebounds == ebounds
            ), "Ebounds do not match the ones in the response object!"
        elif response is not None:
            ebounds = response.ebounds

        spi_grb_setup1 = SPISWFile(det, pointing_id, ebounds)

        event_list = EventListWithLiveTime(
            arrival_times=spi_grb_setup1.times,
            measurement=spi_grb_setup1.energy_bins,
            n_channels=spi_grb_setup1.n_channels,
            start_time=spi_grb_setup1.time_start,
            stop_time=spi_grb_setup1.time_stop,
            live_time=spi_grb_setup1.livetimes_per_interval.reshape(-1),
            live_time_starts=spi_grb_setup1.deadtime_bin_starts,
            live_time_stops=spi_grb_setup1.deadtime_bin_stops,
            first_channel=0,
            instrument=spi_grb_setup1.det_name,
            mission=spi_grb_setup1.mission,
            verbose=False,
            edges=spi_grb_setup1.ebounds,
        )

        tsb = cls(
            f"SPIID{pointing_id}D{det}",
            event_list,
            verbose=True,
            response=response,
            container_type=BinnedSpectrumWithDispersion,
        )

        tsb.set_active_time_interval(
            f"{spi_grb_setup1.times[0]}-{spi_grb_setup1.times[-1]}"
        )

        return tsb
