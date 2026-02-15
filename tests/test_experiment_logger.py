import json
import re

from trainer import ExperimentLogger


def test_experiment_logger_sanitizes_invalid_filename_chars(tmp_path):
    raw_run_id = 'x:test/invalid|name*?'
    logger = ExperimentLogger(run_id=raw_run_id, logdir=str(tmp_path))
    try:
        logger.log({"event": "ok"})
    finally:
        logger.close()

    assert logger.run_id == raw_run_id
    assert logger.filepath.exists()
    assert logger.filepath.suffix == ".jsonl"
    assert logger.filepath.parent == tmp_path
    assert all(ch not in logger.filepath.name for ch in '<>:"/\\|?*')

    lines = logger.filepath.read_text(encoding="utf-8").splitlines()
    assert lines, "expected at least one logged record"
    payload = json.loads(lines[-1])
    assert payload.get("event") == "ok"
    assert "timestamp" in payload


def test_experiment_logger_fallback_filename_for_blank_run_id(tmp_path):
    logger = ExperimentLogger(run_id="   ", logdir=str(tmp_path))
    try:
        logger.log({"event": "blank_id"})
    finally:
        logger.close()

    assert logger.filepath.exists()
    assert re.fullmatch(r"run_\d+\.jsonl", logger.filepath.name) is not None
