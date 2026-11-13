"""Label-map parsing, including the woodsapp app-config shape."""

from __future__ import annotations

import json

import pytest

from modelwidget.labels import labels_from_id2label, parse_label_map, resolve_labels


def test_list_and_dict_maps():
    listed = parse_label_map(["tench", "goldfish"], 3)
    assert [item.name for item in listed] == ["tench", "goldfish", "class_2"]

    mapped = parse_label_map({"1": {"name": "goldfish", "description": "a fish"}}, 2)
    assert mapped[0].name == "class_0"
    assert mapped[1].name == "goldfish"
    assert mapped[1].description == "a fish"


def test_woodsapp_label_map_wrapper():
    data = {"label_map": {"0": {"name": "Oak", "description": "N/A"}}}
    labels = parse_label_map(data, 1)
    assert labels[0].name == "Oak"
    assert labels[0].description is None


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        parse_label_map("nope", 2)


def test_resolve_labels_reads_a_file(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(["alpha", "beta"]), encoding="utf-8")
    labels = resolve_labels(path, 2)
    assert [item.name for item in labels] == ["alpha", "beta"]


def test_id2label_fills_from_the_map_when_num_labels_is_missing():
    labels = labels_from_id2label({"0": "NEGATIVE", "1": "POSITIVE"}, None)
    assert [item.name for item in labels] == ["NEGATIVE", "POSITIVE"]


def test_missing_label_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_labels(tmp_path / "missing.json", 2)
