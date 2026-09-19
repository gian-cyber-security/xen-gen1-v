# Dataset template (not training data)

This directory is intentionally empty. Provide a real video-caption JSONL manifest and the referenced files. Supported video formats are MP4, WEBM, MOV, AVI, and MKV. Videos must be readable by the installed imageio/ffmpeg stack.

Conceptual record (EXAMPLE ONLY; do not use as a dataset):
`{"video":"videos/example.mp4","caption":"EXAMPLE caption"}`

Validate with:
`python tools/validate_dataset.py --kind video --data datasets/video_data.jsonl`

The validator is read-only and reports invalid JSONL, missing files, unsupported formats, unreadable media, empty captions, and duplicate references.
