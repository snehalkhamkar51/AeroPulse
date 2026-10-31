import numpy as np
import pandas as pd
import pytest

from aeropulse import data_loader
from aeropulse.data_loader import DatasetError, load_cmapss, add_train_rul, detect_subset_files


def _make_cmapss_files(tmp_path, n_engines=3, cycles_per=5):
    rng = np.random.default_rng(0)
    cols = ["unit_number", "time_cycles"] + [f"x{i}" for i in range(24)]
    # Actually need 26 columns: unit, cycle, 3 settings, 21 sensors
    from aeropulse.config import COLUMN_NAMES

    rows = []
    for u in range(1, n_engines + 1):
        for c in range(1, cycles_per + 1):
            vals = [u, c] + [float(rng.normal()) for _ in range(24)]
            rows.append(vals)
    df = pd.DataFrame(rows, columns=COLUMN_NAMES)
    train_path = tmp_path / "train_FD001.txt"
    df.to_csv(train_path, sep=" ", header=False, index=False)

    # test: one fewer cycle per engine
    trows = []
    for u in range(1, n_engines + 1):
        for c in range(1, cycles_per):
            vals = [u, c] + [float(rng.normal()) for _ in range(24)]
            trows.append(vals)
    tdf = pd.DataFrame(trows, columns=COLUMN_NAMES)
    test_path = tmp_path / "test_FD001.txt"
    tdf.to_csv(test_path, sep=" ", header=False, index=False)

    rul_path = tmp_path / "RUL_FD001.txt"
    rul_path.write_text("\n".join(str(10 + i) for i in range(n_engines)) + "\n")
    return tmp_path


def test_detect_missing_files(tmp_path):
    with pytest.raises(DatasetError):
        detect_subset_files(tmp_path, "FD001")


def test_load_cmapss_and_rul(tmp_path):
    d = _make_cmapss_files(tmp_path)
    train_df, test_df, rul_series = load_cmapss(d, "FD001")
    assert len(train_df) == 3 * 5
    assert train_df.shape[1] == 26
    assert test_df["unit_number"].nunique() == 3
    assert len(rul_series) == 3
    assert list(rul_series.index) == [1, 2, 3]


def test_add_train_rul(tmp_path):
    d = _make_cmapss_files(tmp_path)
    train_df, _, _ = load_cmapss(d, "FD001")
    labeled = add_train_rul(train_df, clip=3)
    # engine 1 max cycle 5: row cycle1 rul 4 clipped to 3
    rul_vals = labeled.loc[labeled["unit_number"] == 1, "RUL"].tolist()
    assert rul_vals[0] == 3  # 5-1=4 clipped to 3
    assert rul_vals[-1] == 0


def test_invalid_missing_values(tmp_path):
    from aeropulse.config import COLUMN_NAMES
    import pandas as pd
    d = tmp_path
    rows = [[1, 1] + [1.0] * 24]
    df = pd.DataFrame(rows, columns=COLUMN_NAMES)
    df.loc[0, "sensor_1"] = float("nan")
    (d / "train_FD001.txt").write_text(df.to_csv(sep=" ", header=False, index=False))
    # need test and rul too
    (d / "test_FD001.txt").write_text(df.to_csv(sep=" ", header=False, index=False))
    (d / "RUL_FD001.txt").write_text("5\n")
    with pytest.raises(DatasetError):
        load_cmapss(d, "FD001")
