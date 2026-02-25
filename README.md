### python analyze_video.py --video "path" --sample-rate 1 --max-faces 1 --device -1


### python analyze_video.py --webcam --webcam-duration 0 --sample-rate 1 --max-faces 1 --device -1 --debug

CUDA
### python analyze_video.py --webcam --webcam-duration 0 --sample-rate 1 --max-faces 1 --device 0 --debug



# Classroom Video Emotion MVP

Tiny CLI MVP that analyzes a recorded classroom video (`.mp4`) or webcam feed by sampling frames, detecting faces with OpenCV Haar cascade, classifying face crops locally with a Hugging Face Transformers model (`nihar245/Expression-Detection-BEIT-Large`), and producing a simple teaching recommendation from aggregated classroom states.

## Files

- `analyze_video.py`
- `requirements.txt`
- `.env.example`
- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`
- Outputs generated at runtime:
  - `outputs/summary.csv`
  - `outputs/report.json`

## Setup

```bash
pip install -r requirements.txt
```

## Model Download

On first run, `transformers` downloads model files from Hugging Face and caches them locally.
No `HF_TOKEN` is required for this public model.

## CLI Usage

Video file mode:

```bash
python analyze_video.py --video path/to/video.mp4
```

Webcam mode (shows live preview, press `q` to stop):

```bash
python analyze_video.py --webcam
```

Webcam mode without display window:

```bash
python analyze_video.py --webcam --no-display
```

Optional flags:

- `--sample-rate` (default `1.0`): sampled frames per second
- `--max-faces` (default `3`): largest faces per sampled frame
- `--output-dir` (default `outputs`)
- `--webcam-device` (default `0`)
- `--webcam-duration` (default `30`, `0` = run until stopped)
- `--device` (default `-1`): `-1` for CPU, `0` for first CUDA GPU
- `--debug`: print raw model labels and scores per face

## Static Frontend (Webcam + Emotion)

A tiny local static demo is in `frontend/`.

From project root:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/frontend/` in your browser, enter your HF token, and click **Start Webcam**.

Notes:

- Browser will ask for camera permission.
- The frontend sends captured JPEG frames directly to Hugging Face from your browser.
- Your HF token is used client-side in that page.

## Emotion Mapping and Recommendation

Model labels -> classroom state:

- `engaged` -> `engaged`
- `neutral` -> `neutral`
- `sad` -> `bored`
- `fear`, `surprise` -> `confused`
- `angry`, `disgust` -> `frustrated`

Final recommendation rules (overall distribution):

- if `bored >= 50%`: `Do a quick interactive poll or ask a direct question.`
- else if `confused >= 35%`: `Re-explain with a concrete example and slow down.`
- else if `frustrated >= 25%`: `Pause briefly, clarify steps, and address issues.`
- else: `Keep going, students look engaged.`

## Output

The script prints:

- total frames sampled
- total face crops sent
- emotion distribution (%)
- classroom distribution (%)
- final recommendation

`outputs/summary.csv` has one row per sampled second:

- `timestamp_sec`
- `engaged_pct`
- `neutral_pct`
- `bored_pct`
- `confused_pct`
- `frustrated_pct`
- `top_state`
- `recommendation`

`outputs/report.json` contains overall aggregated stats.

## Error Handling

The script fails gracefully with helpful errors when:

- source input is missing or invalid
- CLI values are invalid (e.g., non-positive sample rate)
- video/webcam cannot be opened

## Privacy, Consent, and Limitations

- Obtain explicit consent before analyzing classroom recordings.
- Handle videos/results as sensitive data; follow school and legal policy.
- Performance depends on lighting, camera angle, occlusion, and video quality.
- Face detection and emotion models can be biased and may be inaccurate across demographics and contexts.
- Recommendation is a coarse signal, not a pedagogical diagnosis.
