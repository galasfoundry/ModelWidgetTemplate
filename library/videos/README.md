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
