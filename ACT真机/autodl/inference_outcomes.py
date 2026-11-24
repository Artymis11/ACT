"""Offline inference outcome labels and success-rate accounting.

F8 contributes an operator-reported failure. Timeout and finish events remain
unknown until reviewed. Manual labels are appended beside the source events.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid


ACT_ROOT = Path(__file__).resolve().parent.parent
MAX_EVENT_BYTES = 32 * 1024 * 1024
MAX_RECORDS = 50000
MAX_LINE_BYTES = 1024 * 1024


def checked_events_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ACT_ROOT.resolve()):
        raise ValueError("Inference outcome files must stay inside D:\\act.")
    if not path.is_file():
        raise ValueError(f"Inference event file does not exist: {path}")
    return path


def labels_path(events_path):
    path = events_path.with_name(events_path.stem + ".manual_labels.jsonl").resolve()
    if not path.is_relative_to(ACT_ROOT.resolve()):
        raise ValueError("Manual labels must stay inside D:\\act.")
    return path


def read_records(stream, source):
    total_bytes = 0
    for line_number in range(1, MAX_RECORDS + 2):
        line = stream.readline(MAX_LINE_BYTES + 1)
        if not line:
            return
        total_bytes += len(line)
        if len(line) > MAX_LINE_BYTES or total_bytes > MAX_EVENT_BYTES or line_number > MAX_RECORDS:
            raise ValueError(f"Outcome input exceeds its size/record limit: {source}")
        if not line.strip():
            continue
        try:
            record = json.loads(line.decode("utf-8-sig"))
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(f"Invalid JSON at {source} line {line_number}. Finish the run before editing labels.") from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at {source} line {line_number}.")
        yield record


def valid_trial_id(value):
    if type(value) is not int or not 1 <= value <= 100:
        raise ValueError("trial_id must be an integer from 1 to 100.")
    return value


def load_trials(events_path):
    trials = {}
    with events_path.open("rb") as stream:
        for record in read_records(stream, events_path):
            event = record.get("event")
            if event not in {"trial_start", "trial_end"}:
                continue
            trial_id = valid_trial_id(record.get("trial_id"))
            trial = trials.setdefault(trial_id, {"trial_id": trial_id, "start": None, "end": None})
            key = "start" if event == "trial_start" else "end"
            if trial[key] is not None:
                raise ValueError(f"Duplicate {event} for trial {trial_id}. Review the source events.")
            trial[key] = record
    # The inference supervisor numbers attempts consecutively from one. A
    # later ID proves an earlier attempt slot existed even if its logs dropped.
    if trials:
        for trial_id in range(1, max(trials) + 1):
            trials.setdefault(trial_id, {"trial_id": trial_id, "start": None, "end": None})
    return trials


def automatic_outcome(trial):
    end = trial["end"]
    if end is not None and end.get("outcome") == "failed" and end.get("reason") == "operator_failed":
        return "failed"
    return "unknown"


@contextmanager
def locked_labels(path, *, create):
    if not create and not path.exists():
        yield None
        return
    with path.open("a+b" if create else "r+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Manual outcome labels are busy. Retry after the other label command finishes.") from error
            try:
                yield stream
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("Manual outcome labels are busy. Retry after the other label command finishes.") from error
            try:
                yield stream
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_labels(stream, path, trials):
    labels = {}
    if stream is None:
        return labels
    stream.seek(0)
    for record in read_records(stream, path):
        trial_id = valid_trial_id(record.get("trial_id"))
        if (record.get("schema") != "act-inference-manual-outcome-v1"
                or record.get("outcome") not in {"success", "failed"}
                or type(record.get("corrects_prior")) is not bool):
            raise ValueError(f"Invalid manual outcome record for trial {trial_id}.")
        trial = trials.get(trial_id)
        if trial is None or trial["start"] is None or trial["end"] is None:
            raise ValueError(f"Manual label trial {trial_id} has no complete source trial.")
        if record.get("episode_id") != trial["start"].get("episode_id"):
            raise ValueError(f"Manual label trial {trial_id} belongs to another cloud episode.")
        if (trial_id in labels or automatic_outcome(trial) != "unknown") and not record["corrects_prior"]:
            raise ValueError(f"Duplicate outcome for trial {trial_id} requires an explicit correction.")
        labels[trial_id] = record
    return labels


def summarize(events):
    events = checked_events_path(events)
    trials = load_trials(events)
    label_file = labels_path(events)
    with locked_labels(label_file, create=False) as stream:
        labels = load_labels(stream, label_file, trials)
    details = []
    for trial_id, trial in sorted(trials.items()):
        label = labels.get(trial_id)
        outcome = label["outcome"] if label is not None else automatic_outcome(trial)
        details.append({"trial_id": trial_id, "started": trial["start"] is not None,
                        "ended": trial["end"] is not None, "outcome": outcome,
                        "source": "manual_label" if label else ("F8_operator_failure" if outcome == "failed" else "unlabelled"),
                        "reason": trial["end"].get("reason") if trial["end"] else None,
                        "note": label.get("note", "") if label else ""})
    attempted = len(details)
    completed = sum(item["ended"] for item in details)
    successes = sum(item["outcome"] == "success" for item in details)
    failed = sum(item["outcome"] == "failed" for item in details)
    unknown = attempted - successes - failed
    known = successes + failed
    missing_start = [item["trial_id"] for item in details if not item["started"]]
    definitive = attempted > 0 and completed == attempted and unknown == 0 and not missing_start
    return {"events": str(events), "manual_labels": str(label_file),
            "attempted": attempted, "completed": completed, "success": successes,
            "failed": failed, "unknown": unknown, "manual_labelled": len(labels),
            "success_rate_complete": definitive,
            "success_rate": successes / attempted if definitive else None,
            "success_rate_denominator": attempted if definitive else None,
            "labelled_success_rate": successes / known if known else None,
            "labelled_rate_denominator": known,
            "success_fraction_lower_bound": successes / attempted if attempted else None,
            "success_fraction_upper_bound": (successes + unknown) / attempted if attempted else None,
            "missing_trial_start": missing_start,
            "unfinished_trials": [item["trial_id"] for item in details if not item["ended"]],
            "count_scope": "Sequential trial IDs through the maximum recorded ID. Missing earlier IDs remain unknown. Rates use fractions from 0 to 1.",
            "trials": details}


def label(events, trial_id, outcome, *, note="", correct=False):
    events = checked_events_path(events)
    trial_id = valid_trial_id(trial_id)
    if outcome not in {"success", "failed"}:
        raise ValueError("Manual outcome must be success or failed.")
    if len(note) > 4000:
        raise ValueError("Keep the manual outcome note within 4000 characters.")
    trials = load_trials(events)
    trial = trials.get(trial_id)
    if trial is None or trial["start"] is None or trial["end"] is None:
        raise ValueError(f"Trial {trial_id} must have both trial_start and trial_end before labelling.")
    label_file = labels_path(events)
    with locked_labels(label_file, create=True) as stream:
        labels = load_labels(stream, label_file, trials)
        prior = labels.get(trial_id)
        has_prior = prior is not None or automatic_outcome(trial) != "unknown"
        if has_prior and not correct:
            raise ValueError(f"Trial {trial_id} already has an outcome. Use --correct to append an explicit correction.")
        record = {"schema": "act-inference-manual-outcome-v1", "label_id": uuid.uuid4().hex,
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "trial_id": trial_id, "episode_id": trial["start"].get("episode_id"),
                  "outcome": outcome, "note": note, "corrects_prior": bool(correct),
                  "supersedes_label_id": prior.get("label_id") if prior else None,
                  "supersedes_automatic_outcome": automatic_outcome(trial) if has_prior and prior is None else None}
        encoded = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        if size:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                raise ValueError("Manual labels have an incomplete final line. Review it before appending.")
            stream.seek(0, os.SEEK_END)
        if stream.tell() + len(encoded) > MAX_EVENT_BYTES:
            raise ValueError("Manual label file exceeds its size limit.")
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    video_index = events.parent / "trial_videos" / f"trial_{trial_id:03d}.json"
    if video_index.is_file():
        try:
            from inference_trial_video import classify_trial_video
            folder = classify_trial_video(events.parent, trial_id, record["episode_id"], outcome)
            record["video_folder"] = str(folder) if folder else None
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            record["video_archive_error"] = str(error)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize", help="Count observed trials and reviewed outcomes.")
    summary.add_argument("--events", required=True, type=Path)
    manual = commands.add_parser("label", help="Append a reviewed success/failed label.")
    manual.add_argument("--events", required=True, type=Path)
    manual.add_argument("--trial-id", required=True, type=int)
    manual.add_argument("--outcome", required=True, choices=["success", "failed"])
    manual.add_argument("--note", default="")
    manual.add_argument("--correct", action="store_true", help="Explicitly supersede a prior outcome and retain its history.")
    args = parser.parse_args(argv)
    try:
        if args.command == "label":
            result = label(args.events, args.trial_id, args.outcome, note=args.note, correct=args.correct)
        else:
            result = summarize(args.events)
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
