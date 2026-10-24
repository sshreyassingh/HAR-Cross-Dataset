from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_folders_exist():
    for d in ["data", "src", "configs", "experiments", "results",
              "figures", "tests", "docs", "splits"]:
        assert (ROOT / d).is_dir(), f"missing folder: {d}"


def test_configs_parse():
    files = list((ROOT / "configs").glob("*.yaml"))
    assert files, "no yaml configs found"
    for f in files:
        assert isinstance(yaml.safe_load(f.read_text()), dict), f.name


def test_base_config_keys():
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    for key in ["seed", "data", "train", "eval"]:
        assert key in cfg


def test_uci_har_present_if_downloaded():
    d = ROOT / "data" / "raw" / "uci_har" / "UCI HAR Dataset"
    if not d.exists():
        pytest.skip("UCI-HAR not downloaded yet")
    assert (d / "train" / "X_train.txt").exists()
