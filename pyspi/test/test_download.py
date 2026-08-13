# Test download from isdarc for given pointin_id
import os
import urllib.request
from tempfile import NamedTemporaryFile
from urllib.error import URLError

import pytest

try:
    urllib.request.urlopen(
        "ftp://isdcarc.unige.ch/arc/rev_3/scw/"
        "1189/118900580010.001/sc_orbit_param.fits.gz"
    )
    skip_test = False
except URLError:
    skip_test = True

try:
    tempfile = NamedTemporaryFile(delete=False)
    file_path = os.path.join(
        "isdcarc.unige.ch::arc",
        "rev_3",
        "scw",
        1189,
        "118900580010.001",
        "sc_orbit_param.fits.gz",
    )
    save_path = tempfile.name
    os.system(f"rsync -ltv {file_path} {save_path}")
    tempfile.close()
    skip_test = False
except Exception as e:
    print("Also failed rsync from isdc - will skip test for good " + str(e))


@pytest.mark.skipif(skip_test, reason="ISDC data archiev is broken")
def test_download():
    import os
    import shutil

    from pyspi.io.file_utils import file_existing_and_readable
    from pyspi.io.get_files import get_files
    from pyspi.io.package_data import get_path_of_external_data_dir

    pointing_id = 169600130010
    base_path = os.path.join(
        get_path_of_external_data_dir(), "pointing_data", str(pointing_id)
    )

    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    get_files(pointing_id)

    geom_save_path = os.path.join(
        get_path_of_external_data_dir(),
        "pointing_data",
        str(pointing_id),
        "sc_orbit_param.fits.gz",
    )
    data_save_path = os.path.join(
        get_path_of_external_data_dir(),
        "pointing_data",
        str(pointing_id),
        "spi_oper.fits.gz",
    )

    assert file_existing_and_readable(geom_save_path), "Download test failed"
    assert file_existing_and_readable(data_save_path), "Download test failed"
