// Offline map viewer for /maps. Everything loads from this server:
// MapLibre, pmtiles.js and the Protomaps style come from static/vendor/,
// tiles from /tiles/... (MBTiles) or /maps/files/... (PMTiles by byte range).
(function () {
  "use strict";
  const el = document.getElementById("map");
  if (!el || typeof maplibregl === "undefined") return;

  const origin = window.location.origin;
  const assets = origin + el.dataset.assets;
  const select = document.getElementById("map-pack");
  const status = document.getElementById("map-status");
  const UK = [-8.65, 49.86, 1.77, 60.86];

  const protocol = new pmtiles.Protocol();
  maplibregl.addProtocol("pmtiles", protocol.tile);

  const map = new maplibregl.Map({
    container: el,
    style: { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#e9eef2" } }] },
    bounds: UK,
    fitBoundsOptions: { padding: 20 },
    attributionControl: { compact: false },
  });
  map.addControl(new maplibregl.NavigationControl(), "top-right");
  map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");

  let packs = [];

  function say(text) { if (status) status.textContent = text; }

  // A plain style for vector packs whose schema we don't know (e.g. OS Open
  // Zoomstack or OpenMapTiles): every layer drawn as fill / line / point.
  function genericVectorLayers(source, layerIds) {
    const out = [{ id: "bg", type: "background", paint: { "background-color": "#f4f1ea" } }];
    layerIds.forEach(function (id) {
      out.push({ id: id + "-fill", type: "fill", source: source, "source-layer": id,
        filter: ["==", ["geometry-type"], "Polygon"], paint: { "fill-color": "#dcd8cc", "fill-opacity": 0.6 } });
      out.push({ id: id + "-line", type: "line", source: source, "source-layer": id,
        filter: ["==", ["geometry-type"], "LineString"], paint: { "line-color": "#8a8a8a", "line-width": 1 } });
      out.push({ id: id + "-point", type: "circle", source: source, "source-layer": id,
        filter: ["==", ["geometry-type"], "Point"], paint: { "circle-radius": 2, "circle-color": "#555" } });
    });
    return out;
  }

  function isProtomaps(layerIds) {
    return layerIds.indexOf("earth") >= 0 && layerIds.indexOf("roads") >= 0;
  }

  function vectorStyle(source, layerIds) {
    const style = { version: 8, sources: { pack: source }, layers: [] };
    if (isProtomaps(layerIds) && typeof basemaps !== "undefined") {
      // Protomaps basemap schema: use the vendored style, local glyphs and sprites.
      style.glyphs = assets + "/fonts/{fontstack}/{range}.pbf";
      style.sprite = assets + "/sprites/v4/light";
      style.layers = basemaps.layers("pack", basemaps.namedFlavor("light"), { lang: "en" });
    } else {
      style.layers = genericVectorLayers("pack", layerIds);
    }
    return style;
  }

  function rasterStyle(source) {
    return { version: 8, sources: { pack: source }, layers: [{ id: "raster", type: "raster", source: "pack" }] };
  }

  async function pmtilesStyle(pack) {
    const url = origin + pack.file_url;
    const archive = new pmtiles.PMTiles(url);
    protocol.add(archive);
    const header = await archive.getHeader();
    const meta = (await archive.getMetadata()) || {};
    const bounds = [header.minLon, header.minLat, header.maxLon, header.maxLat];
    const base = { url: "pmtiles://" + url, attribution: pack.attribution || meta.attribution || "" };
    if (header.tileType === pmtiles.TileType.Mvt) {
      const ids = (meta.vector_layers || []).map(function (l) { return l.id; });
      return { style: vectorStyle(Object.assign({ type: "vector" }, base), ids), bounds: bounds };
    }
    return { style: rasterStyle(Object.assign({ type: "raster", tileSize: 256 }, base)), bounds: bounds };
  }

  function mbtilesStyle(pack) {
    const meta = pack.metadata || {};
    const fmt = String(pack.format || "pbf").toLowerCase();
    const source = {
      tiles: [origin + pack.tile_url],
      minzoom: meta.minzoom || 0,
      maxzoom: meta.maxzoom || 14,
      attribution: pack.attribution || "",
    };
    if (meta.bounds) source.bounds = meta.bounds;
    if (fmt === "pbf" || fmt === "mvt") {
      const ids = (meta.vector_layers || []).map(function (l) { return l.id; });
      return { style: vectorStyle(Object.assign({ type: "vector" }, source), ids), bounds: meta.bounds };
    }
    return { style: rasterStyle(Object.assign({ type: "raster", tileSize: 256 }, source)), bounds: meta.bounds };
  }

  async function show(name) {
    const pack = packs.find(function (p) { return p.name === name; });
    if (!pack) return;
    say("Loading " + pack.title + "…");
    try {
      const built = pack.kind === "pmtiles" ? await pmtilesStyle(pack) : mbtilesStyle(pack);
      map.setStyle(built.style);
      const b = built.bounds && built.bounds.every(isFinite) ? built.bounds : UK;
      map.fitBounds(b, { padding: 20, animate: false });
      say("");
    } catch (err) {
      console.error(err);
      say("Could not open " + pack.title + ": " + err.message);
    }
  }

  fetch(el.dataset.packsUrl)
    .then(function (r) { return r.json(); })
    .then(function (data) {
      packs = data.packs || [];
      if (!packs.length) { say("No map packs found."); return; }
      if (select) {
        select.addEventListener("change", function () { show(select.value); });
        show(select.value || packs[0].name);
      } else {
        show(packs[0].name);
      }
    })
    .catch(function (err) { say("Could not list map packs: " + err.message); });

  window.firstBookMap = map; // handy for debugging in the console
})();
