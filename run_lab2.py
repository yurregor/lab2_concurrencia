"""
Supervisor del Laboratorio 2. Funciona en Windows y Linux. ESTE es el que se lanza.

Para que existe: si main.py se cae por un error inesperado a mitad de los 3
dias (falta de memoria, error de un driver, etc.), nadie estaria ahi para
volver a lanzarlo. Este script lo relanza solo y main.py continua donde iba.

Decide segun el codigo de salida de main.py (core/exit_codes.py):
    0  todas las combinaciones terminaron -> termina
    2  error de configuracion / sin GPU   -> termina (relanzar no lo arregla)
    4  Ctrl+C del usuario                  -> termina
    otro (caida inesperada)                -> espera SUPERVISOR_ESPERA_TRAS_FALLO_S y relanza

Uso:  python run_lab2.py
Solo libreria estandar: no carga nada pesado.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import config
from core.exit_codes import EXIT_COMPLETED, EXIT_CONFIG_ERROR, EXIT_INTERRUPTED

HERE = Path(__file__).resolve().parent
STOP_CODES = {EXIT_COMPLETED: "todas las combinaciones terminaron",
              EXIT_CONFIG_ERROR: "error de configuracion",
              EXIT_INTERRUPTED: "detenido por el usuario"}


def log(message: str) -> None:
    print(f"[supervisor {datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def record_restart(code: int) -> None:
    """Deja rastro de cada relanzamiento en results/reinicios.jsonl."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "reinicios.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": datetime.now().isoformat(), "codigo_salida": code}) + "\n")


def run_child(extra_args: list[str]) -> tuple[int, bool]:
    """Ejecuta main.py y espera a que termine. Retorna (codigo, hubo_ctrl_c).

    Ctrl+C llega a ambos procesos (misma consola). El hijo se detiene
    ordenadamente; el supervisor solo anota que el usuario pidio parar y SIGUE
    esperando. No se usa subprocess.call porque ante Ctrl+C mata al hijo.
    """
    proc = subprocess.Popen([sys.executable, str(HERE / "main.py"), *extra_args], cwd=HERE)
    user_stop = False
    while True:
        try:
            return proc.wait(), user_stop
        except KeyboardInterrupt:
            user_stop = True
            log("Ctrl+C recibido: esperando que main.py cierre ordenadamente...")


def main() -> int:
    args: list[str] = []
    while True:
        log(f"Lanzando main.py {' '.join(args)}")
        code, user_stop = run_child(args)
        if user_stop or code in STOP_CODES:
            log(f"Fin (codigo {code}: {STOP_CODES.get(code, 'Ctrl+C')}).")
            return code if code == EXIT_CONFIG_ERROR else 0
        record_restart(code)
        log(f"main.py se cayo (codigo {code}). Relanzo en "
            f"{config.SUPERVISOR_ESPERA_TRAS_FALLO_S} s y continua donde iba (Ctrl+C cancela).")
        try:
            time.sleep(config.SUPERVISOR_ESPERA_TRAS_FALLO_S)
        except KeyboardInterrupt:
            log("Cancelado por el usuario.")
            return 0
        args = ["--reanudar"]


if __name__ == "__main__":
    sys.exit(main())
