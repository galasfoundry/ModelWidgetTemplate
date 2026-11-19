# The First Book library

All offline content lives here (or wherever `LIBRARY_DIR` points).

| Folder | What goes in it | Committed? |
| --- | --- | --- |
| `maps/` | `*.mbtiles` and `*.pmtiles` UK map packs | No, only the README |
| `videos/` | `*.mp4`, `*.webm`, `*.mkv`, `*.m4v` plus `videos.json` | No, only the README and example |
| `guides/` | `*.md` and `*.txt` guides | Yes |

Model weights are not here; they stay in `weights/` (see the main README).
Each folder can also be moved on its own with `MAPS_DIR`, `VIDEOS_DIR` and
`GUIDES_DIR`. See `docs/ARCHITECTURE.md` for the full picture.
