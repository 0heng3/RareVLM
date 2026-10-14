"""Recover report export only; preserve the frozen scientific implementation."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np

from clip_retrieval import analysis, learning
from rarevlm_paths import SOURCE_ROOT
from vlm_probe.evaluate import now, sha_file, write_json


def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg["output_root"])
    owner = (root / "STUDY.lock").open("a+")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = json.loads((root / "STUDY_INPUT_LOCK.json").read_text())
    if sha_file(args.config) != lock["config_sha256"]:
        raise ValueError("Frozen config changed")
    for rel, digest in lock["source_hashes"].items():
        if sha_file(SOURCE_ROOT / rel) != digest:
            raise ValueError(f"Frozen source changed: {rel}")
    if sha_file(root / "DATA_SOURCE_LOCK.json") != lock["data_source_lock_sha256"]:
        raise ValueError("Frozen data source changed")
    for stage in ["feature_queue", "training_queue", "evaluation_queue"]:
        if json.loads((root / stage / "STATUS.json").read_text())["status"] != "completed":
            raise ValueError(f"Compute stage incomplete: {stage}")
    protected = [args.config, root / "STUDY_INPUT_LOCK.json"]
    protected += [SOURCE_ROOT / rel for rel in lock["source_hashes"]]
    protected += sorted((root / "evaluation").rglob("EVALUATION.json"))
    protected += sorted((root / "evaluation").rglob("per_query.npz"))
    protected += sorted((root / "runs").rglob("TRAIN_RESULT.json"))
    before = {str(path): sha_file(path) for path in protected}
    audit_path = root / "analysis" / "REPORT_EXPORT_RECOVERY.json"
    audit_path.parent.mkdir(exist_ok=True)
    if audit_path.exists():
        raise ValueError("Recovery already recorded; inspect existing report first")
    audit = {
        "status": "started",
        "started_at": now(),
        "prior_status": json.loads((root / "STATUS.json").read_text()),
        "failure": "NumPy int64 in seed-direction count was not JSON serializable",
        "controller_log_sha256": sha_file(root / "logs" / "controller.log"),
        "recovery_source_sha256": sha_file(Path(__file__)),
        "scope": "Original analyze/build functions; JSON scalar conversion only; no training, ranking, score, bootstrap, threshold or frozen source changes",
        "protected_sha256": before,
    }
    write_json(audit_path, audit)
    original_writer = analysis.write_json
    analysis.write_json = lambda path, value: original_writer(path, plain(value))
    try:
        result = plain(analysis.analyze(args.config))
        learning.build(root, result)
        after = {str(path): sha_file(path) for path in protected}
        if before != after:
            raise ValueError("Protected study artifacts changed during export")
        audit.update(status="completed", completed_at=now(), protected_unchanged=True,
                     analysis_sha256=sha_file(root / "analysis" / "ANALYSIS.json"))
        write_json(audit_path, audit)
        write_json(root / "COMPLETION.json", {
            "status": "completed", "completed_at": now(),
            "analysis_sha256": audit["analysis_sha256"],
            "report_export_recovery_sha256": sha_file(audit_path),
        })
        write_json(root / "STATUS.json", {
            "status": "completed", "updated_at": now(),
            "source_lock_sha256": sha_file(root / "STUDY_INPUT_LOCK.json"),
            "report_export_recovered": True, "existing_tasks_untouched": True,
            "gates": result["gates"],
        })
        print(json.dumps({"status": "completed", "gates": result["gates"],
                          "comparisons": result["comparisons"],
                          "summaries": result["summaries"],
                          "protected_unchanged": True}, ensure_ascii=False, indent=2))
    except BaseException as error:
        audit.update(status="failed", failed_at=now(), reason=repr(error))
        write_json(audit_path, audit)
        raise
    finally:
        analysis.write_json = original_writer
        owner.close()


if __name__ == "__main__":
    main()
