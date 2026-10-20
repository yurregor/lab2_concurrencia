"""
status.json (para reanudar) y heartbeat.json (esta vivo?).

Ninguno crea hilos: el orquestador los actualiza desde el hilo principal.

Escritura ATOMICA: se escribe a un .tmp y luego os.replace(). Si el proceso
muere a mitad de escritura, queda el archivo anterior completo.
"""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def write_json_atomic(path: Path, data: dict, retries: int = 5) -> bool:
    """Escribe JSON de forma atomica. Retorna False si no pudo (no lanza).

    Reintentos: en Windows os.replace falla si otro programa tiene el archivo
    abierto en ese instante (ej. un editor mostrando heartbeat.json).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    for _ in range(retries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.2)
    logger.warning("No se pudo actualizar %s (archivo bloqueado por otro programa)", path.name)
    return False


class StatusManager:
    """Avance del run: por cada combinacion, su estado y los SEGUNDOS que ya corrio.

    Conserva las claves del Lab 1. En el Lab 2, models_completed /
    models_remaining contienen NOMBRES DE COMBINACION.
    """

    def __init__(self, results_dir: Path, run_id: str, combos: list[str],
                 duracion_s: float, schema: str):
        self.path = results_dir / f"{run_id}_status.json"
        self.data = {
            "run_id": run_id, "status": "running",
            "started_at": datetime.now().isoformat(),
            "updated_at": None, "current_model_index": 0, "current_model": "N/A",
            "models_completed": [], "models_remaining": list(combos),
            "last_heartbeat": None, "schema_version": schema,
            "duracion_combinacion_s": duracion_s, "reinicios": 0,
            "combos": {c: {"estado": "pending", "inicio": None, "fin": None,
                           "segundos": 0.0} for c in combos},
        }

    @classmethod
    def load(cls, path: Path) -> "StatusManager":
        obj = cls.__new__(cls)
        obj.path = path
        obj.data = json.loads(path.read_text(encoding="utf-8"))
        return obj

    @property
    def run_id(self) -> str:
        return self.data["run_id"]

    def pending(self) -> list[str]:
        """Combinaciones que faltan, en orden."""
        return list(self.data["models_remaining"])

    def seconds_done(self, combo: str) -> float:
        return float(self.data["combos"][combo]["segundos"])

    def start_combo(self, combo: str) -> None:
        c = self.data["combos"][combo]
        c["estado"] = "running"
        c["inicio"] = c["inicio"] or datetime.now().isoformat()
        names = list(self.data["combos"])
        self.data.update(current_model_index=names.index(combo), current_model=combo)
        self.save("running")

    def add_seconds(self, combo: str, seconds: float) -> None:
        c = self.data["combos"][combo]
        c["segundos"] = round(c["segundos"] + seconds, 1)

    def finish_combo(self, combo: str) -> None:
        c = self.data["combos"][combo]
        c["estado"], c["fin"] = "completed", datetime.now().isoformat()
        self.data["models_completed"].append(combo)
        self.data["models_remaining"].remove(combo)
        self.save("running")

    def mark_resumed(self) -> None:
        self.data["reinicios"] = self.data.get("reinicios", 0) + 1

    def save(self, status: str | None = None) -> None:
        if status:
            self.data["status"] = status
        now = datetime.now().isoformat()
        self.data["updated_at"] = now
        self.data["last_heartbeat"] = now
        write_json_atomic(self.path, self.data)


def find_resumable(results_dir: Path) -> Path | None:
    """status.json mas reciente que quedo 'running' (caida) o 'interrupted' (Ctrl+C)."""
    if not results_dir.exists():
        return None
    for path in sorted(results_dir.glob("run_*_status.json"), reverse=True):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("status") in ("running", "interrupted"):
                return path
        except (OSError, ValueError):
            continue
    return None


class Heartbeat:
    """heartbeat.json con los campos del Lab 1 mas los nuevos."""

    def __init__(self, results_dir: Path, camera_name: str):
        self.path = results_dir / "heartbeat.json"
        self.camera_name = camera_name

    def write(self, status: str, combinacion: str | None = None, modelos: str = "",
              fps_actual: dict | None = None, camara_ok: bool | None = None,
              ultimo_frame_id: int | None = None, minutos_restantes: float | None = None,
              disco_libre_gb: float | None = None) -> None:
        write_json_atomic(self.path, {
            "status": status, "modelo": modelos, "camara": self.camera_name,
            "pid": os.getpid(), "last_update": datetime.now().isoformat(),
            "combinacion": combinacion, "fps_actual": fps_actual or {},
            "camara_ok": camara_ok, "ultimo_frame_id": ultimo_frame_id,
            "minutos_restantes_combo": None if minutos_restantes is None else round(minutos_restantes, 1),
            "disco_libre_gb": disco_libre_gb,
        })
