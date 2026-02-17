import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

HF_MODEL = "nihar245/Expression-Detection-BEIT-Large"
EMOTION_TO_STATE = {
    "engaged": "engaged",
    "bored": "bored",
    "confused": "confused",
    "happy": "engaged",
    "neutral": "neutral",
    "sad": "bored",
    "fear": "confused",
    "surprise": "confused",
    "angry": "frustrated",
    "disgust": "frustrated",
}
STATE_ORDER = ["engaged", "neutral", "bored", "confused", "frustrated"]


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze classroom emotions from video file or webcam."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", help="Path to input MP4 video")
    source_group.add_argument("--webcam", action="store_true", help="Use webcam input")

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Frames sampled per second (default: 1.0)",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=3,
        help="Max number of largest faces per sampled frame (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for summary.csv and report.json (default: outputs)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Transformers pipeline device. Use -1 for CPU, 0 for first CUDA GPU (default: -1).",
    )
    parser.add_argument(
        "--webcam-device",
        type=int,
        default=0,
        help="Webcam device index for --webcam (default: 0)",
    )
    parser.add_argument(
        "--webcam-duration",
        type=float,
        default=30.0,
        help="Webcam run duration in seconds. 0 means run until 'q' or Ctrl+C (default: 30)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV webcam preview window",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw model labels/scores for each detected face",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists() or not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be > 0")
    if args.max_faces <= 0:
        raise ValueError("--max-faces must be > 0")
    if args.device < -1:
        raise ValueError("--device must be -1 (CPU) or >= 0 (GPU index)")
    if args.webcam_duration < 0:
        raise ValueError("--webcam-duration must be >= 0")


def make_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")
    return detector


def get_recommendation(overall_states_pct: Dict[str, float]) -> str:
    bored = overall_states_pct.get("bored", 0.0)
    confused = overall_states_pct.get("confused", 0.0)
    frustrated = overall_states_pct.get("frustrated", 0.0)

    if bored >= 50.0:
        return "Do a quick interactive poll or ask a direct question."
    if confused >= 35.0:
        return "Re-explain with a concrete example and slow down."
    if frustrated >= 25.0:
        return "Pause briefly, clarify steps, and address issues."
    return "Keep going, students look engaged."


def load_local_classifier(device: int) -> Any:
    return pipeline(
        task="image-classification",
        model=HF_MODEL,
        device=device,
    )


def classify_face_crop(
    image_bgr: np.ndarray,
    classifier: Any,
    debug: bool = False,
) -> Tuple[Optional[str], Optional[float]]:
    if image_bgr.size == 0:
        return None, None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    try:
        payload = classifier(pil_image)
    except Exception:
        return None, None

    if not isinstance(payload, list) or not payload:
        return None, None

    top = max(
        (item for item in payload if isinstance(item, dict)),
        key=lambda x: float(x.get("score", 0.0)),
        default=None,
    )
    if not top:
        return None, None

    label = normalize_label(str(top.get("label", "")))
    score = float(top.get("score", 0.0))
    if debug:
        print(f"debug: raw_label={label}, score={score:.4f}")
    if label in EMOTION_TO_STATE:
        return label, score
    return None, None


def pct_distribution(counter: Counter, labels: List[str]) -> Dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {label: 0.0 for label in labels}
    return {label: round((counter.get(label, 0) / total) * 100.0, 2) for label in labels}


def top_state(state_counter: Counter) -> str:
    if not state_counter:
        return "no_data"
    return max(STATE_ORDER, key=lambda s: state_counter.get(s, 0))


def annotate_frame(
    frame: np.ndarray,
    face_labels: List[Tuple[int, int, int, int, str, float]],
    latest_recommendation: str,
) -> np.ndarray:
    output = frame.copy()
    for (x, y, w, h, emotion, score) in face_labels:
        cv2.rectangle(output, (x, y), (x + w, y + h), (25, 200, 90), 2)
        tag = f"{emotion} ({score:.2f})"
        cv2.putText(
            output,
            tag,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 200, 90),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(output, (8, 8), (output.shape[1] - 8, 42), (10, 10, 10), -1)
    cv2.putText(
        output,
        latest_recommendation,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return output


def process_sample(
    frame: np.ndarray,
    timestamp_sec: float,
    detector: cv2.CascadeClassifier,
    classifier: Any,
    args: argparse.Namespace,
) -> Tuple[dict, Counter, Counter, List[Tuple[int, int, int, int, str, float]], int]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[: args.max_faces]

    second_emotions = Counter()
    second_states = Counter()
    face_labels: List[Tuple[int, int, int, int, str, float]] = []
    faces_sent = 0

    for (x, y, w, h) in faces_sorted:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        faces_sent += 1
        emotion, score = classify_face_crop(
            crop,
            classifier,
            debug=args.debug,
        )
        if emotion is None:
            continue

        state = EMOTION_TO_STATE[emotion]
        second_emotions[emotion] += 1
        second_states[state] += 1
        face_labels.append((x, y, w, h, emotion, score or 0.0))

    state_pct = pct_distribution(second_states, STATE_ORDER)
    rec = get_recommendation(state_pct)
    row = {
        "timestamp_sec": timestamp_sec,
        "engaged_pct": state_pct["engaged"],
        "neutral_pct": state_pct["neutral"],
        "bored_pct": state_pct["bored"],
        "confused_pct": state_pct["confused"],
        "frustrated_pct": state_pct["frustrated"],
        "top_state": top_state(second_states),
        "recommendation": rec,
    }

    return row, second_emotions, second_states, face_labels, faces_sent


def analyze_video_file(
    args: argparse.Namespace, classifier: Any, detector: cv2.CascadeClassifier
) -> Tuple[dict, List[dict]]:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    step_frames = max(1, int(round(fps / args.sample_rate)))

    frame_index = 0
    sampled_frames = 0
    total_faces_sent = 0
    overall_emotions = Counter()
    overall_states = Counter()
    rows: List[dict] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % step_frames == 0:
            sampled_frames += 1
            timestamp_sec = round(frame_index / fps, 2)
            row, second_emotions, second_states, _, faces_sent = process_sample(
                frame, timestamp_sec, detector, classifier, args
            )
            rows.append(row)
            total_faces_sent += faces_sent
            overall_emotions.update(second_emotions)
            overall_states.update(second_states)

        frame_index += 1

    cap.release()
    return finalize_summary(
        source_name=str(Path(args.video).resolve()),
        args=args,
        sampled_frames=sampled_frames,
        total_faces_sent=total_faces_sent,
        overall_emotions=overall_emotions,
        overall_states=overall_states,
        rows=rows,
    )


def analyze_webcam(
    args: argparse.Namespace, classifier: Any, detector: cv2.CascadeClassifier
) -> Tuple[dict, List[dict]]:
    cap = cv2.VideoCapture(args.webcam_device)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam device index: {args.webcam_device}."
        )

    rows: List[dict] = []
    overall_emotions = Counter()
    overall_states = Counter()
    sampled_frames = 0
    total_faces_sent = 0

    sample_interval = 1.0 / args.sample_rate
    next_sample_at = 0.0
    start_ts = time.time()
    last_overlay_frame: Optional[np.ndarray] = None
    last_recommendation = "Waiting for emotion samples..."

    print("Webcam mode started. Press 'q' in the preview window to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_ts

        if elapsed >= next_sample_at:
            timestamp_sec = round(elapsed, 2)
            row, second_emotions, second_states, face_labels, faces_sent = process_sample(
                frame, timestamp_sec, detector, classifier, args
            )

            rows.append(row)
            sampled_frames += 1
            total_faces_sent += faces_sent
            overall_emotions.update(second_emotions)
            overall_states.update(second_states)
            last_recommendation = row["recommendation"]
            last_overlay_frame = annotate_frame(frame, face_labels, last_recommendation)

            print(
                f"t={timestamp_sec}s | top_state={row['top_state']} | recommendation={row['recommendation']}"
            )

            next_sample_at += sample_interval

        if not args.no_display:
            frame_to_show = last_overlay_frame if last_overlay_frame is not None else frame
            cv2.imshow("Classroom Emotion Webcam", frame_to_show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if args.webcam_duration > 0 and elapsed >= args.webcam_duration:
            break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    return finalize_summary(
        source_name=f"webcam:{args.webcam_device}",
        args=args,
        sampled_frames=sampled_frames,
        total_faces_sent=total_faces_sent,
        overall_emotions=overall_emotions,
        overall_states=overall_states,
        rows=rows,
    )


def finalize_summary(
    source_name: str,
    args: argparse.Namespace,
    sampled_frames: int,
    total_faces_sent: int,
    overall_emotions: Counter,
    overall_states: Counter,
    rows: List[dict],
) -> Tuple[dict, List[dict]]:
    emotion_labels = list(EMOTION_TO_STATE.keys())
    overall_emotion_pct = pct_distribution(overall_emotions, emotion_labels)
    overall_state_pct = pct_distribution(overall_states, STATE_ORDER)
    final_recommendation = get_recommendation(overall_state_pct)

    summary = {
        "source": source_name,
        "model": HF_MODEL,
        "sample_rate_fps": args.sample_rate,
        "max_faces": args.max_faces,
        "sampled_frames": sampled_frames,
        "face_crops_sent": total_faces_sent,
        "overall_emotion_counts": dict(overall_emotions),
        "overall_emotion_pct": overall_emotion_pct,
        "overall_state_counts": dict(overall_states),
        "overall_state_pct": overall_state_pct,
        "final_recommendation": final_recommendation,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    return summary, rows


def write_outputs(output_dir: Path, summary: dict, rows: List[dict]) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "report.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_sec",
                "engaged_pct",
                "neutral_pct",
                "bored_pct",
                "confused_pct",
                "frustrated_pct",
                "top_state",
                "recommendation",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return csv_path, json_path


def print_summary(summary: dict, csv_path: Path, json_path: Path) -> None:
    print(f"Model: {summary['model']}")
    print(f"Total frames sampled: {summary['sampled_frames']}")
    print(f"Total face crops sent: {summary['face_crops_sent']}")
    print("Emotion distribution (%):")
    for label, pct in summary["overall_emotion_pct"].items():
        print(f"  - {label}: {pct}")

    print("Classroom distribution (%):")
    for label, pct in summary["overall_state_pct"].items():
        print(f"  - {label}: {pct}")

    print(f"Final recommendation: {summary['final_recommendation']}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved report: {json_path}")


def main() -> int:
    try:
        args = parse_args()
        validate_args(args)
        print(f"Loading local model: {HF_MODEL} (device={args.device})")
        classifier = load_local_classifier(args.device)
        detector = make_face_detector()

        if args.video:
            summary, rows = analyze_video_file(args, classifier, detector)
        else:
            summary, rows = analyze_webcam(args, classifier, detector)

        csv_path, json_path = write_outputs(Path(args.output_dir), summary, rows)
        print_summary(summary, csv_path, json_path)
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
