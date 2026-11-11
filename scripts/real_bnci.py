"""Run the fixed, chronological development benchmark on official BNCI MATLAB releases."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from qbci.data import Session
from qbci.experiment import DEFAULT_METHODS, ExperimentConfig, run_pair
from qbci.cli import save_run

URL = "https://lampx.tugraz.at/~bci/database/"
CHANNELS = {
    "001": "Fz FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz".split(),
    "004": ["C3", "Cz", "C4"],
}


def source_file(raw, dataset, subject, phase):
    prefix = "A" if dataset == "001" else "B"
    relative = f"{dataset}-2014/{prefix}{subject:02d}{phase}.mat"
    path = raw / relative
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".mat.download")
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                        "--retry", "2", "--max-time", "900", "--output", str(part),
                        URL + relative], check=True)
        part.replace(path)
    return path, {"url": URL + relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_subject(raw, dataset, subject):
    channels = CHANNELS[dataset]
    offset = 2 if dataset == "001" else 3
    sessions, audit = [], []
    for phase in ("T", "E"):
        path, provenance = source_file(raw, dataset, subject, phase)
        runs = np.atleast_1d(loadmat(str(path), variable_names=["data"],
                                    struct_as_record=False, squeeze_me=True)["data"])
        blocks = []
        for run_no, run in enumerate(runs):
            markers = np.asarray(run.trial).astype(np.int64).ravel()
            codes = np.asarray(run.y).ravel()
            if not len(markers) and not len(codes):
                continue  # Native EOG-only calibration recordings.
            fs = int(run.fs)
            if fs != 250 or run.X.shape[1] != len(channels) + 3:
                raise ValueError(f"Unexpected sample rate/montage in {path}")
            names = list(run.classes)
            if names[:2] != ["left hand", "right hand"]:
                raise ValueError(f"Unknown class convention: {names}")
            if len(codes) != len(markers) or np.any(np.diff(markers) <= 0) or min(markers) < 1:
                raise ValueError("Invalid event ordering")
            eeg, labels, ids = [], [], []
            for marker, code in zip(markers, codes):
                if code not in (1, 2):
                    continue
                # MATLAB one-based trial-start sample, followed by the cue offset.
                start = int(marker) - 1 + offset * fs
                block = run.X[start:start + 4*fs, :len(channels)]
                if block.shape != (4*fs, len(channels)) or not np.isfinite(block).all():
                    raise ValueError("Incomplete or nonfinite native epoch")
                eeg.append(block.T * 1e-6)
                labels.append(int(code) - 1)
                ids.append(f"{path.stem}:{run_no}:{marker}")
            blocks.append((eeg, labels, ids))
        # IV-2a: one file/session, six task runs. IV-2b: one task record/session.
        if dataset == "001" and len(blocks) != 6:
            raise ValueError(f"Unexpected task run count in {path}")
        if dataset == "004" and len(blocks) != (3 if phase == "T" else 2):
            raise ValueError(f"Unexpected session count in {path}")
        groups = [blocks] if dataset == "001" else [[b] for b in blocks]
        for group in groups:
            eeg, labels, ids = ([x for block in group for x in block[i]] for i in range(3))
            n = len(sessions)+1
            if dataset == "001" and np.bincount(labels,minlength=2).tolist() != [72,72]:
                raise ValueError("Unexpected IV-2a hand trial counts")
            session = Session("BNCI2014_"+dataset,str(subject),str(n),n,np.stack(eeg),
                              np.array(labels),250,tuple(channels),np.array(ids),
                              dataset == "004" and n >= 3)
            sessions.append(session)
            audit.append(dict(provenance, session=n, trials=len(labels),
                              class_counts=np.bincount(labels,minlength=2).tolist(),
                              first_id=ids[0],last_id=ids[-1],task_runs=len(group),
                              eeg_channels=channels,cue_offset_seconds=offset,
                              feedback=session.feedback, artifact_policy="retain all task trials"))
    return sessions,audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=CHANNELS, required=True)
    p.add_argument("--subjects",type=int,nargs="+",default=list(range(1,10)))
    p.add_argument("--raw",type=Path,default=Path("data/raw"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--config",type=Path,default=Path("configs/development.json"))
    p.add_argument("--methods",nargs="+",default=DEFAULT_METHODS)
    args = p.parse_args()
    config = ExperimentConfig(**json.loads(args.config.read_text()))
    if set(args.subjects)-set(range(1,10)):
        p.error("subjects must be in 1..9")
    with threadpool_limits(limits=1):
        for subject in args.subjects:
            sessions,audit = load_subject(args.raw,args.dataset,subject)
            parent=args.out/f"BNCI2014_{args.dataset}"/f"sub-{subject:02d}"
            parent.mkdir(parents=True,exist_ok=True)
            (parent/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
            for target in sessions[1:]:
                dest=parent/f"s1-to-s{target.session}"
                print("Running",dest,flush=True)
                rows,pred,cost=run_pair(sessions[0],target,config,args.methods)
                save_run(dest,rows,pred,cost,config,"REAL EEG: fixed-config development")
                print("Finished",dest,len(rows),"scores",flush=True)


if __name__ == "__main__":
    main()
