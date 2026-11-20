# Videos

Drop practical how-to videos here (repair, electronics, power, radio, tools,
3D printing, and so on). Sub-folders are fine.

Metadata goes in `videos.json` (copy `videos.example.json`), or in a sidecar
`<file>.json` next to one video. Build or refresh the manifest with:

```bash
python scripts/build_video_manifest.py            # this folder
python scripts/build_video_manifest.py /media/usb/videos --topic electronics
```

Existing titles, tags and notes are kept; new files get an entry titled from
their filename. Durations are filled in when `ffprobe` is installed.
Only add videos you are allowed to keep a copy of, and record the source and
licence.

## Creative Commons starter set

`config/videos_catalog.json` lists videos that may be kept offline because
their creators published them under Creative Commons Attribution (CC BY).
Fetch them while online (needs `pip install yt-dlp`; ffmpeg recommended):

```bash
python scripts/fetch_videos.py --list            # what is in the catalog and on disk
python scripts/fetch_videos.py                   # download the missing ones (360p MP4, ~290 MB)
python scripts/fetch_videos.py --max-height 480  # sharper copies, ~500 MB
```

The script re-checks each licence on YouTube before downloading, saves
`<topic>/<id>.mp4` here, and rebuilds `videos.json` with each creator's
attribution line. The watch page shows the credit under the player. If
YouTube answers "HTTP 403" or "confirm you're not a bot", wait a few minutes
and run it again; half-finished files in `.downloading/` resume.

### Credits

These videos are by their creators, used under the Creative Commons
Attribution 3.0 licence (CC BY 3.0, https://creativecommons.org/licenses/by/3.0/).
They are stored at reduced resolution for offline viewing; nothing else is
changed. Licences checked on YouTube on 19 Nov 2026.

**DIY off-grid hydro**

- [Real World Results From The $50 Water Turbine](https://www.youtube.com/watch?v=lBl4Sdw0dXo) by [OpenSourceLowTech](https://www.youtube.com/@OpenSourceLowTech), 15:13, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). File: `hydro/50-dollar-water-turbine-results.mp4`.
- [Motorbike Alternators With Wind & Water Turbines](https://www.youtube.com/watch?v=jLVcbd0C9Ns) by [OpenSourceLowTech](https://www.youtube.com/@OpenSourceLowTech), 20:01, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). File: `hydro/motorbike-alternators-wind-water-turbines.mp4`.
- [Micro hydro power, Pump As Turbine](https://www.youtube.com/watch?v=kiH02Gfj_k8) by [Leif Roger Innerdal](https://www.youtube.com/@TheHuggern), 1:42, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). File: `hydro/pump-as-turbine-micro-hydro.mp4`.

**Battery storage**

- [Assembling 18650 fused battery pack.](https://www.youtube.com/watch?v=KNKmogsoXN8) by [Open Sun Power](https://www.youtube.com/@opensunpower8507), 12:09, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). File: `batteries/assembling-fused-18650-pack.mp4`.
- [18650 Salvage, Testing, Sorting, & Making Pack](https://www.youtube.com/watch?v=E_11eyYvp2M) by [richardhroth](https://www.youtube.com/@richardhroth), 9:49, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). File: `batteries/18650-salvage-test-sort-pack.mp4`.
- [Salvaged Tesla Batteries for Off-Grid Power?](https://www.youtube.com/watch?v=4veRBtg6rno) by [Overkill Solar Channel](https://www.youtube.com/@OverkillSolarLLC), 11:19, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). File: `batteries/salvaged-tesla-module-bms.mp4`.

**Starlink off-grid**

- [How much power does Starlink really use [SURPRISE]](https://www.youtube.com/watch?v=HPMgruSja8E) by [Programmer Offgrid](https://www.youtube.com/@ProgrammerOffgrid), 11:52, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). File: `starlink-off-grid/starlink-power-use-12h-test.mp4`.

## First Book originals

Short explainers made for The First Book live in `originals/`. They are our
own work, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
credited to The First Book / galasfoundry, and fact-checked by Murder before
they go into the library.

The sidecar (`originals/001-head-and-flow.mp4.json`) and subtitles
(`originals/001-head-and-flow.srt`) are in git. The video file is not
(episode 001 is about 6.7 MB at 720p). Copy it into place:

```bash
mkdir -p library/videos/originals
cp /path/to/001-head-and-flow.mp4 library/videos/originals/
```

It then appears on `/videos` under Power, with the credit line and
subtitles. Later episodes use the same layout: `originals/<id>-<slug>.mp4`
plus a `<file>.json` sidecar and an optional `.srt`.
