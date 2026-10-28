"""推荐基线只允许使用训练期可见的数据。"""

import csv
import json

import pytest

from recagent.engine.store import Catalog


@pytest.fixture
def tiny_catalog_dir(tmp_path):
    movies = [
        {"item_id": 1, "title": "One", "genres": ["Comedy"]},
        {"item_id": 2, "title": "Two", "genres": ["Drama"]},
        {"item_id": 3, "title": "Three", "genres": ["Comedy"]},
        {"item_id": 4, "title": "Future", "genres": ["Drama"]},
    ]
    (tmp_path / "items.json").write_text(json.dumps(movies), encoding="utf-8")
    rows = [(1, 1, 10), (1, 2, 11), (2, 1, 12), (2, 3, 13), (3, 2, 14), (3, 3, 15)]
    with (tmp_path / "interactions_train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "item_id", "timestamp", "rating"])
        writer.writerows((*row, "") for row in rows)
    with (tmp_path / "interactions_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "item_id", "timestamp", "rating"])
        writer.writerow([1, 4, 20, ""])
    return tmp_path


def test_catalog_uses_train_only(tiny_catalog_dir):
    catalog = Catalog.load(tiny_catalog_dir)
    assert catalog.history[1] == (1, 2)
    assert catalog.popularity == {1: 2, 2: 2, 3: 2}
    assert 4 not in catalog.item_users


def test_catalog_requires_training_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        Catalog.load(tmp_path)
