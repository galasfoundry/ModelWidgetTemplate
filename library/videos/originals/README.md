# First Book originals

This folder holds short animated explainers made for The First Book
(CC BY 4.0, credited to The First Book / galasfoundry).

The `.mp4.json` sidecar and `.srt` subtitles for each episode are in git.
The `.mp4` itself is not. Download library fetches both episodes from the
live GitHub Release `library-v1`. To place a file by hand, copy it here
next to those sidecars:

    # from a checkout of the rendered originals (example):
    cp 001-head-and-flow.mp4 library/videos/originals/
    cp 002-watts-vs-watt-hours.mp4 library/videos/originals/

Episode 001, "Head and flow: how much power is in your stream?", is about
6.7 MB at 720p. Episode 002, "Watts vs watt-hours: power, energy and your
daily budget", is about 6.4 MB at 720p. Once a file is in place it appears
on `/videos` under Power, with credits and subtitles. Episodes follow the
same layout: `<id>-<slug>.mp4`, `<id>-<slug>.mp4.json` and optional
`<id>-<slug>.srt`.
