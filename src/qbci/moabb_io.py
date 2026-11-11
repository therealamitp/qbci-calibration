"""Audited MOABB export, intentionally requiring explicit cue-marker offsets.

MOABB release changes can alter raw event conventions. Never silently assume a
default offset based only on a dataset name. Pass one offset per session after
checking the pinned loader. This exporter reads raw data, not MOABB's standard
CrossSessionEvaluation, which does not implement this study's prefix protocol.
"""
import hashlib
import inspect
import json
from pathlib import Path
import re
import numpy as np
from .data import Session

DATASETS = ("BNCI2014_001", "BNCI2014_004", "Lee2019_MI")


def natural_key(value):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(value))]


def export_subject(name, subject, output, cue_offsets):
    import mne
    import moabb
    from moabb import datasets
    if name not in DATASETS:
        raise ValueError(name)
    constructor = getattr(datasets, name)
    kwargs = {}
    if name == "Lee2019_MI":
        # Fail on incompatible loaders instead of silently including online runs.
        kwargs = {"train_run": True, "test_run": False, "resting_state": False}
    dataset = constructor(**kwargs)
    raw_sessions = dataset.get_data(subjects=[int(subject)])[int(subject)]
    session_names = sorted(raw_sessions, key=natural_key)
    if len(cue_offsets) != len(session_names):
        raise ValueError("Supply one audited cue offset for each session in order: " + str(session_names))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_text = inspect.getsource(constructor)
    audit = {"dataset": name, "subject": subject, "moabb_version": moabb.__version__,
             "mne_version": mne.__version__, "loader_class_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
             "loader_interval": list(dataset.interval), "sessions": [],
             "test_run": False if name == "Lee2019_MI" else "not_applicable"}
    saved = []
    for number, (session_name, offset) in enumerate(zip(session_names, cue_offsets), 1):
        parts, labels, identifiers, run_audits = [], [], [], []
        channel_names, sfreq = None, None
        for run_name in sorted(raw_sessions[session_name], key=natural_key):
            raw = raw_sessions[session_name][run_name]
            if name == "Lee2019_MI" and "train" not in run_name.lower():
                raise ValueError("Unexpected OpenBMI run: " + run_name)
            picks = mne.pick_types(raw.info, eeg=True, eog=False, emg=False, stim=False, exclude=[])
            channels = tuple(raw.ch_names[i] for i in picks)
            current_sfreq = float(raw.info["sfreq"])
            if channel_names is not None and (channels != channel_names or current_sfreq != sfreq):
                raise ValueError("Run montage/rate mismatch")
            channel_names, sfreq = channels, current_sfreq
            stim = mne.pick_types(raw.info, stim=True)
            if len(stim):
                events = mne.find_events(raw, shortest_event=1, verbose=False)
                mapping = dataset.event_id
            else:
                events, annotation_map = mne.events_from_annotations(raw, verbose=False)
                mapping = {}
                for label in ("left_hand", "right_hand"):
                    candidates = (label, str(dataset.event_id[label]))
                    matches = [annotation_map[c] for c in candidates if c in annotation_map]
                    if len(matches) != 1:
                        raise ValueError("Unrecognized annotation mapping, audit manually: " + str(annotation_map))
                    mapping[label] = matches[0]
            run_counts = {"left": 0, "right": 0}
            for marker, _, event in events:
                if event not in (mapping["left_hand"], mapping["right_hand"]):
                    continue
                label = 0 if event == mapping["left_hand"] else 1
                start = int(marker - raw.first_samp + round(offset*sfreq))
                stop = start + int(round(4*sfreq))
                if start < 0 or stop > raw.n_times:
                    raise ValueError("Incomplete epoch; do not silently change a chronological prefix")
                epoch = raw.get_data(picks=picks, start=start, stop=stop)
                if not np.isfinite(epoch).all():
                    raise ValueError("Nonfinite epoch needs a prespecified quality-policy decision")
                parts.append(epoch)
                labels.append(label)
                identifiers.append("%s:%s" % (run_name, marker))
                run_counts["left" if label == 0 else "right"] += 1
            run_audits.append({"run": run_name, "counts": run_counts})
        data = Session(name, str(subject), session_name, number, np.stack(parts), np.asarray(labels),
                       sfreq, channel_names, np.asarray(identifiers), name == "BNCI2014_004" and number >= 3)
        expected = {"BNCI2014_001": 144, "Lee2019_MI": 100}.get(name, 120 if number <= 2 else 160)
        if len(labels) != expected or any(np.count_nonzero(data.labels == c) != expected//2 for c in (0, 1)):
            raise ValueError("Unexpected trial/label counts; audit the release before analysis")
        file_path = output / ("%s_sub-%s_session-%d.npz" % (name, subject, number))
        data.save(file_path)
        saved.append(str(file_path))
        audit["sessions"].append({"name": session_name, "order": number, "cue_offset_seconds": offset,
                                  "trials": len(labels), "channels": channels, "runs": run_audits,
                                  "export_sha256": hashlib.sha256(file_path.read_bytes()).hexdigest()})
    audit_path = output / ("%s_sub-%s_audit.json" % (name, subject))
    audit_path.write_text(json.dumps(audit, indent=2))
    return saved, str(audit_path)
