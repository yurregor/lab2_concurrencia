"""
Presupuesto de hilos del laboratorio.

Por que existe: con 2 modelos en paralelo, cada libreria (OpenVINO-CPU,
torch, OpenMP, OpenCV) intenta usar TODOS los nucleos por su cuenta. Con 8
vCPU eso son 8 + 8 + ... hilos compitiendo -> sobresuscripcion, cambios de
contexto y FPS inestables. Aqui se CUENTAN los hilos disponibles y se reparten.

IMPORTANTE: apply_thread_env() debe llamarse ANTES de importar numpy, cv2,
torch, openvino o ultralytics, porque esas librerias leen las variables de
entorno de hilos una sola vez, al importarse.

Este modulo solo usa la libreria estandar a proposito.
"""
import os
from dataclasses import dataclass, asdict

# Hilos de Python que crea el laboratorio, ademas de los workers (uno por modelo).
#   MainThread: orquestador + heartbeat + status (sin hilos aparte para eso).
#   cam-capture: lectura/decodificacion de la camara.
#   hw-monitor: muestreo de hardware + revision de compresiones.
FIXED_PYTHON_THREADS = ("MainThread", "cam-capture", "hw-monitor")


@dataclass(frozen=True)
class ThreadBudget:
    """Reparto de hilos calculado al iniciar. Se registra en events.jsonl."""
    cpus_disponibles: int
    cpus_reservadas: int
    modelos_simultaneos: int
    hilos_por_modelo: int
    hilos_opencv: int
    hilos_ffmpeg: int
    hilos_python_fijos: int
    hilos_python_workers: int
    hilos_computo_estimados: int
    sobresuscripcion: bool

    def as_dict(self) -> dict:
        return asdict(self)


def available_cpus() -> int:
    """CPUs que el proceso realmente puede usar.

    En Linux respeta la afinidad (taskset, contenedores); os.cpu_count()
    devolveria las CPUs del equipo aunque el proceso tenga menos asignadas.
    En Windows no existe sched_getaffinity y se usa os.cpu_count().
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def compute_budget(n_models: int, reserved: int, per_model: int | None,
                   opencv_threads: int | None, ffmpeg_threads: int) -> ThreadBudget:
    """Calcula cuantos hilos de computo recibe cada modelo.

    Formula: (CPUs disponibles - reservadas) // modelos simultaneos, minimo 1.
    Dentro de un worker el pre-proceso (OpenCV), la inferencia (OpenVINO) y
    el post-proceso (torch) ocurren EN SECUENCIA, por eso pueden compartir el
    mismo numero de hilos sin sumarse entre si.
    """
    cpus = available_cpus()
    usable = max(1, cpus - reserved)
    threads = per_model or max(1, usable // max(1, n_models))
    cv_threads = opencv_threads or threads
    computo = threads * n_models + ffmpeg_threads
    return ThreadBudget(
        cpus_disponibles=cpus,
        cpus_reservadas=reserved,
        modelos_simultaneos=n_models,
        hilos_por_modelo=threads,
        hilos_opencv=cv_threads,
        hilos_ffmpeg=ffmpeg_threads,
        hilos_python_fijos=len(FIXED_PYTHON_THREADS),
        hilos_python_workers=n_models,
        hilos_computo_estimados=computo,
        sobresuscripcion=computo > cpus,
    )


def apply_thread_env(budget: ThreadBudget) -> None:
    """Fija las variables de entorno de hilos (antes de importar librerias).

    setdefault: si el usuario ya exporto un valor en la consola, se respeta.
    """
    n = str(budget.hilos_por_modelo)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, n)
    os.environ.setdefault("OPENCV_FFMPEG_THREADS", str(budget.hilos_ffmpeg))


def apply_library_threads(budget: ThreadBudget) -> None:
    """Limita los hilos de OpenCV y torch (se llama despues de importarlos)."""
    import cv2
    cv2.setNumThreads(budget.hilos_opencv)
    try:
        import torch
        torch.set_num_threads(budget.hilos_por_modelo)
    except ImportError:
        pass
