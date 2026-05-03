# Data Directory Layout

This directory is organized by artifact lifecycle.

```text
data/
├── raw/
│   ├── urfd_videos/                  # downloaded URFD source videos
│   └── voxel_huggingface_data/       # downloaded Hugging Face source videos
├── manifests/
│   └── video_manifest.csv            # compact metadata and split assignment
├── splits/
│   └── videos/                       # train/valid/test video symlinks or copies
├── processed/
│   ├── frames/                       # sampled frames and frame manifests
│   ├── poses/                        # per-video pose JSON and pose manifests
│   ├── windows/                      # pre-LLM window CSV/JSONL files
│   ├── window_poses/                 # compact one-file-per-window pose exports
│   └── window_poses_readable/        # descriptive one-file-per-window pose exports
└── uploads/
    └── windows/                      # one-off uploaded/local video outputs
```

Git tracks compact manifests and JSON/JSONL artifacts when useful. Raw videos, split video folders, sampled frame JPGs, and caches are ignored because they are large or reproducible.
