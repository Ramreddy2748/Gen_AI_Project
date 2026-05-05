# Handoff: Single-Video Fall Detection Demo

**For:** teammate running the demo on her Mac
**Author:** Harshitha
**Date:** 2026-05-04

## What this is

A small CLI script that takes one video file as input and returns a `FALL` or `NO_FALL` verdict, using our existing pipeline:

1. Extract frames with OpenCV
2. Run MediaPipe pose on each frame
3. Build 3-frame sliding windows in the same schema as our training data
4. Classify each window with **GPT-4o** using the safety-first **Best-Prompt** strategy (our highest-recall classifier — 97.9% video recall on the test set)
5. Aggregate window predictions to a single video-level verdict

Script location: `final_pipeline/demo/2026-05-04-video_fall_demo.py`

The script does **not** modify any existing pipeline file. It only imports from `final_pipeline/scripts/balanced_pipeline.py` (pose + window construction) and `final_pipeline/prompting/best_prompt.py` (classifier).

## Prereqs

- macOS with Python 3.9 or newer
- A video clip to test (`.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`)
- Your **OpenAI API key**
- The repo cloned somewhere on your Mac

## Step 1 — Pull the latest code

```bash
cd ~/path/to/Gen_AI_Project        # wherever you cloned it
git pull origin main
```

(Use the branch name we agreed on if it's not `main`.)

## Step 2 — Activate the virtualenv

If you already have a `venv/` in the project:

```bash
source venv/bin/activate
```

If not, create one:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The demo needs `opencv-python`, `mediapipe`, and `openai`. All three are listed in `requirements.txt`.

## Step 3 — Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

This sets the key for the current terminal session only. If you close the terminal you'll need to re-export. To make it permanent, add the same line to `~/.zshrc` and run `source ~/.zshrc`.

## Step 4 — Run the demo

```bash
python final_pipeline/demo/2026-05-04-video_fall_demo.py
```

A native macOS file picker will pop up. Select your video and click Open. The script will then:

- Sample frames + run MediaPipe pose (a few seconds)
- Compute velocity / acceleration / posture features
- Build 3-frame sliding windows
- Send each window to GPT-4o
- Print a per-window timeline and a final verdict

You'll see output like this:

```
Video        : /Users/<you>/Desktop/test_clip.mp4
Output dir   : /Users/<you>/Gen_AI_Project/demo_out/test_clip
[1/4] Extracting frames and running MediaPipe pose...
      fps=30.00, frames sampled=42, pose detected on=37
[2/4] Computing velocity, acceleration, and posture features...
[3/4] Building 3-frame sliding windows (stride=1)...
      40 windows built
[4/4] Classifying 40 windows with gpt-4o (Best-Prompt strategy)...

   window   1/40  t=  0.00s–  0.33s          no_fall
   window   2/40  t=  0.17s–  0.50s          no_fall
   ...
   window  17/40  t=  2.83s–  3.17s   FALL    fall
   ...

================================================================
  VIDEO VERDICT : FALL
  9 of 40 windows predicted FALL (threshold = 25%)
================================================================
```

## Where the results go

By default everything lands in `./demo_out/<video_stem>/`:

```
demo_out/
  test_clip/
    frames/         # extracted JPEGs
    windows/        # one JSON per 3-frame window with full pose data
    result.json     # final verdict + per-window timeline
```

You can change the destination with `--output-dir <path>`.

## Useful flags

| Flag | Default | What it does |
|------|---------|---|
| `--video PATH` | (opens picker) | Skip the picker, pass a path directly |
| `--output-dir DIR` | `./demo_out/<video_stem>` | Where to write results |
| `--sample-rate N` | `5` | Sample every Nth frame (lower = denser, slower) |
| `--max-frames N` | `200` | Cap on total frames sampled |
| `--stride N` | `1` | Sliding-window stride |
| `--fall-threshold F` | `0.25` | Fraction of windows predicting FALL needed to flag the video |
| `--no-llm` | off | Skip GPT-4o calls (free dry run, just builds windows) |

Examples:

```bash
# Pass the path directly, no picker
python final_pipeline/demo/2026-05-04-video_fall_demo.py --video ~/Desktop/clip.mp4

# More sensitive — flag the video as FALL if any 10% of windows predict FALL
python final_pipeline/demo/2026-05-04-video_fall_demo.py --fall-threshold 0.10

# Cheap dry run — verify pipeline works without burning API credits
python final_pipeline/demo/2026-05-04-video_fall_demo.py --no-llm
```

## Cost estimate

GPT-4o is roughly **$0.07 per 10-second video** (about 30 windows × ~$0.0025 per call). A 60-second video runs about $0.40. Keep an eye on usage at platform.openai.com.

## Performance / what to expect

- **Not real-time.** A 10-second clip takes 30–90 seconds end-to-end, mostly because of serial GPT-4o calls.
- **Pose detection quality matters.** If MediaPipe can't see the person clearly (poor lighting, far away, occluded), windows show "pose not detected" and the classifier is unreliable. The script will warn you when 0 frames had a detected pose.
- **Conservative bias.** Best-Prompt is tuned safety-first ("when in doubt, choose FALL"). It will occasionally flag a no-fall video as a fall — that's the trade-off for high recall, which is what you want for elderly care.

## Troubleshooting

| Error | Fix |
|---|---|
| `Error: OPENAI_API_KEY not set` | Re-run step 3 in the same terminal |
| `ModuleNotFoundError: No module named 'openai'` | `pip install openai` (also `mediapipe`, `opencv-python` if missing) |
| `Could not open video: ...` | File path is wrong, or codec not supported. Try converting to `.mp4` |
| File picker doesn't open | Run with `--video <path>` instead — your Python build may not have tkinter |
| `RateLimitError 429` | The script auto-retries with back-off; if persistent, lower `--sample-rate` to make fewer calls |
| Pose detection rate < 50% | Make sure the person is well-lit, fully visible in frame, and reasonably close |

## What this demo does NOT need

You don't need to worry about these for the demo to run:

- **`final_pipeline/data/windows_balanced/`** — empty in git, only used by the bulk evaluation scripts
- **`ground_truth.csv` paths** — those still point to Abhishek's machine but don't affect the demo
- **`fix_metadata_paths.py`** — only needed if you want to run the bulk prompting scripts (`few_shot.py`, `chain_of_thought.py`, etc.), not the demo

## Files involved

| File | Role |
|---|---|
| `final_pipeline/demo/2026-05-04-video_fall_demo.py` | The demo script (this is what you run) |
| `final_pipeline/scripts/balanced_pipeline.py` | Imported for `extract_pose`, `enhance_frames`, `create_windows` |
| `final_pipeline/prompting/best_prompt.py` | Imported for `BEST_PROMPT`, `extract_features`, `format_data`, `parse_response`, `aggregate_video` |
| `pose_landmarker_lite.task` | MediaPipe model — auto-downloaded on first run if missing |

## Where to get a test video

**The repo does not contain any video files** — only the processed window JSONs. You need to provide your own video to run the demo. Three options:

1. **Record a short clip on your phone** (easiest). Have someone simulate a stumble or sit-down. ~15 seconds is plenty. AirDrop to your Mac.
2. **Download from URFD** (publicly available): http://fenix.ur.edu.pl/~mkepski/ds/uf.html — grab one fall clip and one ADL (no-fall) clip, save them anywhere on your Mac.
3. **YouTube clip** with `yt-dlp` — Creative Commons or fair-use only.

## Sanity test before the live demo

Once you have at least one fall clip and one no-fall clip, run the demo on each:

```bash
# Run on a known fall video
python final_pipeline/demo/2026-05-04-video_fall_demo.py --video ~/Desktop/fall_test.mp4

# Run on a known no-fall video
python final_pipeline/demo/2026-05-04-video_fall_demo.py --video ~/Desktop/nofall_test.mp4
```

The fall clip should report `VIDEO VERDICT: FALL`; the no-fall clip should report `NO_FALL`. If they're flipped, something's wrong with the pipeline — ping Harshitha.

## Contact

If anything breaks, ping Harshitha. Include:

- The exact command you ran
- The full terminal output (error message + traceback)
- The video file (or describe it: how long, fall or no-fall, how good the lighting/visibility is)
