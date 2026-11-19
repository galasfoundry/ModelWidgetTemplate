# Vendored front-end libraries

Copied in once at build time so the app never needs a CDN. Do not edit the
minified files; replace them with a newer release instead.

| Folder | Package / source | Version | Licence |
| --- | --- | --- | --- |
| `maplibre-gl/` | npm `maplibre-gl` (`dist/maplibre-gl.js`, `dist/maplibre-gl.css`) | 5.24.1 | BSD-3-Clause (`LICENSE.txt`) |
| `pmtiles/` | npm `pmtiles` (`dist/pmtiles.js`, IIFE global `pmtiles`) | 4.5.0 | BSD-3-Clause (`LICENSE`, from github.com/protomaps/PMTiles) |
| `protomaps-basemaps/` | npm `@protomaps/basemaps` (`dist/basemaps.js`, global `basemaps`) | 5.7.2 | code BSD-3-Clause, visual design CC0 (`LICENSE.md`) |
| `basemaps-assets/fonts/` | github.com/protomaps/basemaps-assets `fonts/`, Noto Sans Regular / Medium / Italic, ranges 0-255, 256-511, 8192-8447 | commit 028c18f | SIL Open Font License 1.1 (`OFL.txt`) |
| `basemaps-assets/sprites/v4/` | same repo, `sprites/v4/light*` | commit 028c18f | derived from MIT-licensed tangrams/icons (`LICENSE-tangrams-icons.md`) |

MapLibre 5.x is used rather than 6.x because 5.x still ships a single UMD
file that works with a plain `<script>` tag; 6.x is ES-module only.

To update:

```bash
npm pack maplibre-gl@5 pmtiles @protomaps/basemaps
# unpack each .tgz and copy the dist files listed above
```

`tests/test_offline.py` checks each folder still has its files and licence.
