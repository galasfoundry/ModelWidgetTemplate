# Map packs

Drop UK map packs here. The filename (without extension) becomes the pack
name in URLs, so keep it to letters, digits, `.`, `_` and `-`.

* `name.mbtiles` is served tile by tile at `/tiles/name/{z}/{x}/{y}`.
* `name.pmtiles` is served whole, with Range support, at `/maps/files/name.pmtiles`.

`config/maps_catalog.json` lists the packs we know about, where to get each one
while online, the expected filename, and its licence and attribution. Always
show the attribution (for OpenStreetMap data: "© OpenStreetMap contributors").
