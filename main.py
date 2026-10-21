"""
Ejecuta el Laboratorio 2 (Windows y Linux).

Normalmente se lanza run_lab2.py (supervisor), que llama a este archivo y lo
relanza si el proceso se cae. Tambien se puede lanzar directo para depurar:
    python main.py
    python main.py --reanudar      # fuerza continuar el run interrumpido

El ORDEN de este archivo importa:
1. Leer config (no importa nada pesado).
2. Calcular el presupuesto de hilos y fijar las variables de entorno.
3. Recien entonces importar cv2 / numpy / torch / openvino (via core.*).
"""
import argparse
import logging
import sys

import config
from core.exit_codes import EXIT_COMPLETED, EXIT_CONFIG_ERROR, EXIT_INTERRUPTED
from core.threads import apply_thread_env, compute_budget


def setup_logging() -> None:
    """Log a consola y a results/lab2.log con el hilo en cada linea."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT, handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.RESULTS_DIR / "lab2.log", encoding="utf-8")])


def fail_config(message: str) -> None:
    """Error que no se arregla relanzando: el supervisor se detiene con este codigo."""
    print(f"\n[CONFIG] {message}\n", file=sys.stderr)
    sys.exit(EXIT_CONFIG_ERROR)


def check_device(logger) -> None:
    """Falla rapido si se pidio GPU y OpenVINO no la ve (o cae a CPU si se permite)."""
    from core.engine import available_openvino_devices
    devices = available_openvino_devices()
    logger.info("Dispositivos OpenVINO disponibles: %s", devices)
    if config.DEVICE == "gpu" and not any(d.startswith("GPU") for d in devices):
        if not config.FALLBACK_TO_CPU_IF_NO_GPU:
            fail_config("DEVICE='gpu' pero OpenVINO no ve ninguna GPU (revisa drivers / /dev/dri).")
        logger.warning("Sin GPU: se continua en CPU (FALLBACK_TO_CPU_IF_NO_GPU=True)")
        config.DEVICE = "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 2: inferencia concurrente OpenVINO")
    grupo = parser.add_mutually_exclusive_group()          # no permite usar los dos a la vez
    grupo.add_argument("--reanudar", action="store_true", help="continuar el run interrumpido")
    grupo.add_argument("--nuevo", action="store_true", help="iniciar un run nuevo")
    args = parser.parse_args()

    errors = config.validate_config()
    if errors:
        fail_config("Configuracion invalida:\n  - " + "\n  - ".join(errors))

    budget = compute_budget(config.max_models_per_combination(), config.RESERVED_CPUS,
                            config.THREADS_PER_MODEL, config.OPENCV_THREADS, config.FFMPEG_THREADS)
    apply_thread_env(budget)          # ANTES de importar librerias pesadas

    setup_logging()
    logger = logging.getLogger("main")
    logger.info("Presupuesto de hilos: %s", budget.as_dict())
    logger.info("Combinaciones: %s | %s min cada una",
                config.COMBINATION_ORDER, config.DURACION_COMBINACION_MIN)
    if budget.sobresuscripcion:
        logger.warning("Hay mas hilos de computo que CPUs: considera bajar THREADS_PER_MODEL")

    from core.threads import apply_library_threads
    apply_library_threads(budget)
    check_device(logger)

    from core.orchestrator import COMPLETED, Orchestrator
    result = Orchestrator(config, budget, force_resume=args.reanudar, force_new=args.nuevo).run()
    return EXIT_COMPLETED if result == COMPLETED else EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
