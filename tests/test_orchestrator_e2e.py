"""Punta a punta con modelos falsos y un .mp4: archivos, esquemas,
paso continuo entre combinaciones y reanudacion con los minutos faltantes."""
import gzip
import json
import threading
from datetime import datetime

import config
from core.orchestrator import COMPLETED, INTERRUPTED, Orchestrator
from core.threads import compute_budget
from fakes import FakeModel
from test_camera_frame_id import make_video

BUDGET = compute_budget(2, 2, None, None, 1)
DURACION_S = 3


def _prepare(tmp_path, monkeypatch):
    for task, name in (("deteccion", "yolo11n"), ("pose", "yolo11n-pose"),
                       ("segmentacion", "yolo11n-seg")):
        folder = tmp_path / "modelos" / task / "yolo11" / "OPENVINOFP16" / f"{name}_openvino_model"
        folder.mkdir(parents=True)
        (folder / f"{name}.xml").write_text("<xml/>")
    video = tmp_path / "v.mp4"
    make_video(video, n=60)
    overrides = dict(
        BASE_MODELS=tmp_path / "modelos", RESULTS_DIR=tmp_path / "results",
        CAMERA_SOURCE_TYPE="file", VIDEO_FILE_PATH=str(video), DEVICE="cpu",
        WARMUP_FRAMES=2, SUMMARY_WINDOW_S=1, HEARTBEAT_INTERVAL_S=1, STATUS_INTERVAL_S=1,
        HW_SAMPLE_INTERVAL_S=0.5, HW_ENABLE={**config.HW_ENABLE, "gpu_intel_gpu_top": False},
        COMBINATION_ORDER=["det+pose", "det+seg"], DURACION_COMBINACION_MIN=DURACION_S / 60)
    for key, value in overrides.items():
        monkeypatch.setattr(config, key, value)


def _orch(force_resume=False):
    return Orchestrator(config, BUDGET, force_resume=force_resume,
                        model_factory=lambda info: FakeModel(info, config))


def _lines(run_dir, pattern):
    out = []
    for p in run_dir.glob(pattern + "*"):           # .jsonl o .jsonl.gz
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as fh:
            out += [json.loads(l) for l in fh]
    return out


def _status(tmp_path):
    return json.loads(next((tmp_path / "results").glob("run_*_status.json")).read_text())


def test_config_valida(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    assert config.validate_config() == []


def test_corrida_completa_sin_pausa(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    assert _orch().run() == COMPLETED
    status = _status(tmp_path)
    assert status["status"] == "completed"
    assert status["models_completed"] == ["det+pose", "det+seg"]
    for combo in status["combos"].values():
        assert abs(combo["segundos"] - DURACION_S) < 1.5          # corrio lo configurado
    assert json.loads((tmp_path / "results" / "heartbeat.json").read_text())["status"] == "finished"

    run_dir = tmp_path / "results" / status["run_id"]
    events = _lines(run_dir, "*_events.jsonl")
    end_1 = next(e for e in events if e["tipo"] == "combo_end" and e["combinacion"] == "det+pose")
    start_2 = next(e for e in events if e["tipo"] == "combo_start" and e["combinacion"] == "det+seg")
    gap = (datetime.fromisoformat(start_2["ts"]) - datetime.fromisoformat(end_1["ts"])).total_seconds()
    assert gap < 1.0                                              # sin pausa entre combinaciones
    assert sum(e["tipo"] == "camera_disconnected" for e in events) == 0   # camara nunca se cerro

    frames = _lines(run_dir, "*_det+pose_frames_yolo11n-pose_")
    assert {"fps", "lat_ms", "frame_id", "frames_saltados", "edad_frame_ms"} <= frames[0].keys()
    ids = [f["frame_id"] for f in frames]
    assert ids == sorted(set(ids))                                # solo frames unicos
    assert "polygon" in _lines(run_dir, "*_det+seg_detections_yolo11n-seg_")[0]["detecciones"][0]
    summary = _lines(run_dir, "*_det+pose_summary_")
    assert {s["modelo"] for s in summary} == {"yolo11n", "yolo11n-pose"}
    assert _lines(run_dir, "*_hardware_")


def test_reanuda_con_los_segundos_faltantes(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    orch = _orch()
    threading.Timer(1.5, orch.stop_event.set).start()            # simula Ctrl+C a mitad
    assert orch.run() == INTERRUPTED
    corridos = _status(tmp_path)["combos"]["det+pose"]["segundos"]
    assert 0 < corridos < DURACION_S
    assert _orch(force_resume=True).run() == COMPLETED
    status = _status(tmp_path)
    assert status["reinicios"] == 1
    assert abs(status["combos"]["det+pose"]["segundos"] - DURACION_S) < 1.5
