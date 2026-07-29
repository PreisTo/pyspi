from pyspi.utils.response.spi_response_data import ResponseDataRMF

rsp_bases = tuple(ResponseDataRMF.from_version(i) for i in range(5))
