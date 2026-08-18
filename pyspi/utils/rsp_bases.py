from pyspi.utils.response.spi_response_data import ResponseDataRMF

rsp_bases = {}


def get_base_rsp(version):
    global rsp_bases
    if version not in rsp_bases.keys():
        rsp_bases[version] = ResponseDataRMF.from_version(version)
    return rsp_bases[version]
