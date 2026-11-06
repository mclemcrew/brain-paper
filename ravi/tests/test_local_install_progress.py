"""The installer reports within-file download progress by polling huggingface_hub's .incomplete blob."""
import json
import threading
from pathlib import Path

from ravi.local import worker


def test_incomplete_bytes_sums_in_progress_blobs(tmp_path):
    blobs = tmp_path / "models--stabilityai--stable-audio-3-optimized" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc.incomplete").write_bytes(b"x" * 1000)
    (blobs / "done").write_bytes(b"y" * 5000)          # finished blobs do not count
    assert worker._incomplete_bytes(tmp_path) == 1000


def test_watch_download_reports_fraction_within_file(tmp_path, capsys):
    blobs = tmp_path / "models--stabilityai--stable-audio-3-optimized" / "blobs"
    blobs.mkdir(parents=True)
    part = blobs / "abc.incomplete"
    part.write_bytes(b"x" * 250)
    stop = threading.Event()
    t = threading.Thread(target=worker._watch_download,
                         args=(tmp_path, "dit_medium_f16.npz", 0, 5, 1000, stop, 0.01))
    t.start()
    import time; time.sleep(0.05)
    part.write_bytes(b"x" * 500)
    time.sleep(0.05)
    stop.set(); t.join(timeout=2)
    events = [json.loads(l[5:]) for l in capsys.readouterr().out.splitlines() if l.startswith("RAVI ")]
    assert events, "no progress events"
    assert events[0]["progress"] == (0 + 0.25) / 5
    assert events[-1]["progress"] == (0 + 0.5) / 5
    assert events[-1]["message"] == "Downloading dit_medium_f16.npz · 0.00 of 0.00 GB"   # 500 of 1000 bytes, shown in GB
    # progress is monotone and strictly below the next file's start
    assert all(0 <= e["progress"] < 1 / 5 for e in events)


def test_watch_download_without_known_size_reports_bytes_only(tmp_path, capsys):
    blobs = tmp_path / "models--x--y" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "q.incomplete").write_bytes(b"x" * 10)
    stop = threading.Event()
    t = threading.Thread(target=worker._watch_download, args=(tmp_path, "f.npz", 2, 5, None, stop, 0.01))
    t.start(); import time; time.sleep(0.05); stop.set(); t.join(timeout=2)
    events = [json.loads(l[5:]) for l in capsys.readouterr().out.splitlines() if l.startswith("RAVI ")]
    assert events and events[0]["progress"] == 2 / 5 and "GB" in events[0]["message"]
