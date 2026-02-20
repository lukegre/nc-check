from __future__ import annotations

import xarray as xr

from netcdf_cf_coercer import cli


def test_run_check_uses_pretty_print_mode(monkeypatch, tmp_path, capsys) -> None:
    source = tmp_path / "in.nc"
    xr.Dataset(data_vars={"v": (("time",), [1.0])}, coords={"time": [0]}).to_netcdf(
        source
    )

    seen: dict[str, bool] = {}

    def _fake_check(
        ds: xr.Dataset,
        *,
        pretty_print: bool = False,
        **kwargs: object,
    ) -> None:
        seen["pretty_print"] = pretty_print
        print("report-output")

    monkeypatch.setattr(cli, "check_dataset_compliant", _fake_check)

    status = cli.run_check([str(source)])
    out = capsys.readouterr().out

    assert status == 0
    assert seen["pretty_print"] is True
    assert "report-output" in out


def test_run_comply_writes_output_file(tmp_path) -> None:
    source = tmp_path / "in.nc"
    target = tmp_path / "out.nc"
    xr.Dataset(
        data_vars={"temp": (("lat", "lon"), [[280.0]])},
        coords={"lat": ["10"], "lon": ["20"]},
    ).to_netcdf(source)

    status = cli.run_comply([str(source), str(target)])

    assert status == 0
    assert target.exists()

    with xr.open_dataset(target) as out:
        assert out.attrs["Conventions"] == "CF-1.12"
        assert out["lat"].attrs["standard_name"] == "latitude"
        assert out["lon"].attrs["standard_name"] == "longitude"
