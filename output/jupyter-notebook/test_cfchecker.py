import nc_check


def main():
    fname = "/Users/luke/Downloads/GCB-2025_dataprod_CSIR-ML6_1982-2024.nc"
    data = nc_check.CanonicalDataset.from_file(fname, chunks={}, decode_times=True)
    report = nc_check.cfchecker_report_suite.run(data)

    report.to_json("cfchecker_report.json")


if __name__ == "__main__":
    main()
