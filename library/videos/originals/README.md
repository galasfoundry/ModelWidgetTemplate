# First Book originals

This folder holds short animated explainers made for The First Book
(CC BY 4.0, credited to The First Book / galasfoundry).

The `.mp4.json` sidecar and `.srt` subtitles for each episode are in git.
The `.mp4` itself is not. Download library fetches episodes 001 to 004 from
the live GitHub Release `episodes-v1`. To place a file by hand, copy it here
next to those sidecars:

    cp 001-head-and-flow.mp4 library/videos/originals/

Episode 001, "Head and flow", is about 6.7 MB at 720p. Episode 002, "Watts
vs watt-hours", is about 6.5 MB. Episode 003, "Volts, amps and ohms", is
about 5.4 MB. Episode 004, "The fire triangle", is about 4.8 MB. Once a file
is in place it appears on `/videos` in the Fundamentals lane, with credits
and subtitles. Episodes follow the same layout: `<id>-<slug>.mp4`,
`<id>-<slug>.mp4.json` and optional `<id>-<slug>.srt`.
