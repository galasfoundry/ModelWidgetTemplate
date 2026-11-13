"""Path and weight-lookup tests. These do not download a model."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from paths import (
    ensure_weights,
    find_local_snapshot,
    load_dotenv,
    model_dir_name,
    project_path,
    select_filenames,
    snapshot_from_download,
)


def _snapshot(root: Path, repo_id: str, revision: str) -> Path:
    folder = root / model_dir_name(repo_id) / "snapshots" / revision
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"weights")
    return folder


def test_dotenv_keeps_existing_values_and_reads_new_ones(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "export MODEL_ID=from-file",
                "NEW_KEY_FOR_DOTENV='quoted'",
                "SKIP",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_ID", "from-shell")
    monkeypatch.delenv("NEW_KEY_FOR_DOTENV", raising=False)
    try:
        load_dotenv(env_file)
        assert os.environ["MODEL_ID"] == "from-shell"
        assert os.environ["NEW_KEY_FOR_DOTENV"] == "quoted"
    finally:
        os.environ.pop("NEW_KEY_FOR_DOTENV", None)


def test_snapshot_from_download_keeps_the_snapshot_directory(tmp_path):
    revision = "a" * 40
    snapshot = tmp_path / "models--org--name" / "snapshots" / revision
    blob = tmp_path / "models--org--name" / "blobs" / "deadbeef"
    snapshot.mkdir(parents=True)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"weights")
    link = snapshot / "model.safetensors"
    link.symlink_to(blob)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    found = snapshot_from_download(link)
    assert found == snapshot.resolve()
    assert (found / "config.json").is_file()


def test_project_path_is_relative_to_the_app(tmp_path):
    resolved = project_path("weights/model.safetensors")
    assert resolved.is_absolute()
    assert resolved.name == "model.safetensors"
    assert resolved.parent.name == "weights"


def test_select_filenames_prefers_safetensors_and_skips_other_frameworks():
    files = [
        "README.md",
        ".gitattributes",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "flax_model.msgpack",
        "tf_model.h5",
        "onnx/model.onnx",
        "tokenizer.json",
    ]
    chosen = select_filenames(files)
    assert "model.safetensors" in chosen
    assert "config.json" in chosen
    assert "tokenizer.json" in chosen
    assert "pytorch_model.bin" not in chosen
    assert "flax_model.msgpack" not in chosen
    assert "tf_model.h5" not in chosen
    assert "onnx/model.onnx" not in chosen
    assert "README.md" not in chosen


def test_select_filenames_keeps_bin_when_that_is_the_checkpoint():
    chosen = select_filenames(["config.json", "pytorch_model.bin", "vocab.json"])
    assert chosen == ["config.json", "pytorch_model.bin", "vocab.json"]


def test_exact_revision_in_weights_wins_over_the_hub_cache(tmp_path):
    repo = "timm/example"
    revision = "a" * 40
    weights = _snapshot(tmp_path / "weights", repo, revision)
    _snapshot(tmp_path / "hub", repo, "b" * 40)
    found = find_local_snapshot(repo, revision, tmp_path / "weights", hub_dir=tmp_path / "hub")
    assert found == weights.resolve()


def test_weights_copy_wins_when_the_hub_cache_has_the_same_revision(tmp_path):
    repo = "org/name"
    revision = "b" * 40
    weights = _snapshot(tmp_path / "weights", repo, revision)
    cached = _snapshot(tmp_path / "hub", repo, revision)
    # Make the cache copy newer. The project weights folder still wins.
    os.utime(cached / "model.safetensors", (2_000_000_000, 2_000_000_000))
    found = find_local_snapshot(repo, revision, tmp_path / "weights", hub_dir=tmp_path / "hub")
    assert found == weights.resolve()


def test_hub_cache_is_reused_when_weights_dir_is_empty(tmp_path):
    repo = "org/name"
    revision = "c" * 40
    cached = _snapshot(tmp_path / "hub", repo, revision)
    found = find_local_snapshot(repo, revision, tmp_path / "weights", hub_dir=tmp_path / "hub")
    assert found == cached.resolve()


def test_branch_ref_maps_to_the_snapshot_sha(tmp_path):
    repo = "org/name"
    sha = "d" * 40
    cached = _snapshot(tmp_path / "hub", repo, sha)
    ref = tmp_path / "hub" / model_dir_name(repo) / "refs"
    ref.mkdir(parents=True)
    (ref / "main").write_text(sha, encoding="utf-8")
    found = find_local_snapshot(repo, "main", tmp_path / "weights", hub_dir=tmp_path / "hub")
    assert found == cached.resolve()


def test_model_weights_override_missing_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_WEIGHTS", str(tmp_path / "missing.safetensors"))

    def boom(*_args, **_kwargs):
        raise AssertionError("download should not run")

    monkeypatch.setattr("paths.download_weights", boom)
    with pytest.raises(FileNotFoundError):
        ensure_weights("org/name", "e" * 40, tmp_path / "weights")


def test_model_weights_directory_is_used_as_given(tmp_path, monkeypatch):
    folder = tmp_path / "snap"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"weights")
    (folder / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MODEL_WEIGHTS", str(folder))

    def boom(*_args, **_kwargs):
        raise AssertionError("download should not run")

    monkeypatch.setattr("paths.download_weights", boom)
    found = ensure_weights("org/name", "", tmp_path / "weights")
    assert found == folder.resolve()


def test_relative_model_weights_resolve_from_the_project(monkeypatch, tmp_path):
    relative = Path("weights") / "local-snapshot"
    folder = project_path(relative)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pytorch_model.bin").write_bytes(b"weights")
    monkeypatch.setenv("MODEL_WEIGHTS", relative.as_posix())
    try:
        found = ensure_weights("org/name", "", tmp_path)
        assert found == folder.resolve()
    finally:
        (folder / "pytorch_model.bin").unlink(missing_ok=True)
        folder.rmdir()
