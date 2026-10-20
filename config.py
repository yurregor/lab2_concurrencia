"""
Configuracion del Laboratorio 2: inferencia concurrente OpenVINO FP16.

Es la UNICA fuente de parametros del laboratorio. Ningun modulo de core/
tiene numeros "magicos": todo lo que se puede ajustar vive aqui.

Regla: este archivo NO importa librerias pesadas (cv2, torch, openvino,
ultralytics). Asi main.py y run_lab2.py pueden leerlo ANTES de fijar las
variables de entorno de hilos.

Funciona igual en Windows (pruebas) y Linux (IntelNUC).
"""
import os
from pathlib import Path

# ==============================================================
# 1. RUTAS
# ==============================================================
# LAB_DIR = carpeta donde esta ESTE archivo (.../edge_benchmark/lab2_concurrencia).
# Todas las rutas parten de aqui, asi funciona sin importar desde donde se ejecute.
LAB_DIR = Path(__file__).resolve().parent
#   BASE_MODELS = Path(r"D:\modelos")   (Windows)   |   Path("/home/usuario/modelos")   (Linux)
BASE_MODELS = LAB_DIR.parent / "edge_benchmark" / "modelos" #LAB_DIR.parent / "modelos" #TODO: Modificar ruta de modelos
# Resultados del Lab 2: .../lab2_concurrencia/results
RESULTS_DIR = LAB_DIR / "results"

# ==============================================================
# 2. DISPOSITIVO DE INFERENCIA
# ==============================================================
# "gpu" -> IntelNUC (Intel Iris Xe). "cpu" -> pruebas en el PC sin GPU.
DEVICE = "gpu" #TODO: Modificar a "gpu" para usar la GPU del NUC
# Traduccion al formato que espera Ultralytics para OpenVINO.
OPENVINO_DEVICE_MAP = {"cpu": "intel:cpu", "gpu": "intel:gpu"}
# Si DEVICE="gpu" y OpenVINO no ve la GPU: False = no arrancar (recomendado en la NUC).
FALLBACK_TO_CPU_IF_NO_GPU = False

# ==============================================================
# 3. PRESUPUESTO DE HILOS (ver core/threads.py)
# ==============================================================
# Hilos logicos que NO se entregan a los modelos:
#   1 para decodificar la camara (FFmpeg) + 1 para SO, hilo principal y monitor.
RESERVED_CPUS = 2
# Hilos por modelo para OpenVINO-CPU, torch, OpenMP y OpenCV.
# None = automatico: (CPUs disponibles - RESERVED_CPUS) // modelos simultaneos.
THREADS_PER_MODEL = None
# Hilos internos de OpenCV (resize del preproceso). None = THREADS_PER_MODEL.
OPENCV_THREADS = None
# Hilos de FFmpeg para decodificar. 1 evita el error 'async_lock' del Lab 1.
FFMPEG_THREADS = 1
# Solo en DEVICE="cpu": recompilar el modelo OpenVINO limitando sus hilos.
APPLY_OV_CPU_THREADS = True

# ==============================================================
# 4. CAMARA (una sola fuente para ambos modelos)
# ==============================================================
CAMERA_NAME = "cam1"
# Elige la fuente: "rtsp" (NUC) | "file" (video .mp4 en el PC) | "webcam" (camara del PC)
CAMERA_SOURCE_TYPE = "rtsp" # TODO: Modificar a "rtsp" para usar la camara IP

# --- Si CAMERA_SOURCE_TYPE = "rtsp" ---
# La URL NO se escribe aqui (tiene usuario y clave). Se lee de la variable de
# entorno LAB2_RTSP_URL, que se crea en la consola ANTES de lanzar:
#   Linux / tmux : export LAB2_RTSP_URL='rtsp://usuario:clave@IP:554/ruta'
#   PowerShell   : $env:LAB2_RTSP_URL = "rtsp://usuario:clave@IP:554/ruta"
#   cmd          : set LAB2_RTSP_URL=rtsp://usuario:clave@IP:554/ruta
RTSP_URL = os.environ.get("LAB2_RTSP_URL", "")

# --- Si CAMERA_SOURCE_TYPE = "file" ---
# Ruta COMPLETA al video. En Windows usa r"..." para que las \ no den problemas.
#   Ejemplo Windows: r"C:\Users\yuliurrs\Videos\prueba.mp4"
#   Ejemplo Linux  : "/home/usuario/videos/prueba.mp4"
VIDEO_FILE_PATH = r""
# Entregar los frames al ritmo real del video (como una camara) y repetirlo en bucle.
FILE_SIMULATE_REALTIME = True

# --- Si CAMERA_SOURCE_TYPE = "webcam" ---
WEBCAM_INDEX = 0                     # 0 = primera camara del equipo

# Tiempo maximo que FFmpeg espera al abrir/leer el RTSP (evita bloqueos indefinidos).
CAMERA_OPEN_TIMEOUT_MS = 10000
CAMERA_READ_TIMEOUT_MS = 10000
# Espera progresiva entre reintentos de reconexion: 2, 4, 8 ... hasta el maximo.
RECONNECT_INITIAL_BACKOFF_S = 2
RECONNECT_MAX_BACKOFF_S = 60
# Si no llega un frame nuevo en N s, el heartbeat marca camara_ok = false.
STALE_FRAME_TIMEOUT_S = 10

# ==============================================================
# 5. INFERENCIA (iguales al Lab 1)
# ==============================================================
MODEL_FORMAT = "OPENVINOFP16"        # nombre de la carpeta de formato dentro de modelos/
CONF_THRESHOLD = 0.25
IMGSZ = 640
CLASSES = None                       # None = todas las clases COCO
WARMUP_FRAMES = 15
FRAME_WAIT_TIMEOUT_S = 1.0           # maximo que un worker espera un frame NUEVO
MAX_CONSECUTIVE_ERRORS = 50          # errores seguidos antes de detener ese worker

# ==============================================================
# 6. COMBINACIONES Y DURACION
# ==============================================================
# Cada combinacion = modelos que corren AL MISMO TIEMPO sobre la camara.
# Los nombres son los de las carpetas sin "_openvino_model".
COMBINATIONS = {
    "det+pose": ["yolo11n", "yolo11n-pose"],
    "det+seg": ["yolo11n", "yolo11n-seg"],
    "pose+seg": ["yolo11n-pose", "yolo11n-seg"],
}
# Orden en que se ejecutan, una detras de otra, SIN pausa entre ellas.
COMBINATION_ORDER = ["det+pose", "det+seg", "pose+seg"]
# Minutos que corre CADA combinacion. 1440 = 24 h | 720 = 12 h | 60 = 1 h | 3 = prueba.
# Acepta decimales (0.5 = 30 s). Cuenta tiempo CORRIENDO: si el proceso se cae y se
# reanuda, la combinacion continua solo con los minutos que le faltaban.
DURACION_COMBINACION_MIN = 1440 #TODO: Modificar tiempo
# "auto" -> si hay un run interrumpido, se reanuda.  "ask" -> pregunta.  "never" -> run nuevo.
RESUME_MODE = "never"
# Segundos que espera run_lab2.py antes de relanzar main.py tras una caida.
SUPERVISOR_ESPERA_TRAS_FALLO_S = 30

# ==============================================================
# 7. REGISTRO DE DATOS
# ==============================================================
SCHEMA_VERSION = "2.0"
BATCH_WRITE_FRAMES = 30          # lineas acumuladas antes de escribir a disco
SUMMARY_WINDOW_S = 600           # un resumen cada 10 min (comparable con Lab 1)
ROTATE_EVERY_S = 3600            # un archivo nuevo cada hora
COMPRESS_CLOSED_FILES = True     # comprimir a .gz cada archivo cerrado
SAVE_DETECTIONS = True           # detections.jsonl
WRITE_EMPTY_DETECTIONS = False   # escribir lineas sin detecciones
SAVE_KEYPOINTS = True
SAVE_POLYGONS = True
POLYGON_EPSILON_PX = 2.0         # simplificacion de poligonos; 0 = sin simplificar
COORD_DECIMALS = 0               # decimales de bbox, keypoints y poligonos
MIN_FREE_DISK_GB = 20            # por debajo, se deja de escribir detections.jsonl

# ==============================================================
# 8. EVIDENCIA VISUAL (desactivada por defecto)
# ==============================================================
SAVE_FRAMES = True #TODO: Modificar a False para no guardar frames
SAVE_FRAMES_EVERY_S = 3600 #900

# ==============================================================
# 9. HARDWARE
# ==============================================================
HW_SAMPLE_INTERVAL_S = 2
# Cada metrica se prueba UNA vez al iniciar; si falla, se desactiva toda la corrida.
HW_ENABLE = {
    "cpu": True,
    "cpu_per_core": True,
    "cpu_freq": True,
    "load_avg": True,             # en Windows se desactiva solo (crearia un hilo extra)
    "ram": True,
    "swap": True,
    "process": True,
    "threads_cpu": True,
    "disk_io": True,
    "disk_usage": True,
    "net_io": True,
    "temp": True,                 # Linux; en Windows y en VMs suele no existir
    "gpu_freq_sysfs": True,       # solo Linux
    "gpu_intel_gpu_top": True,    # solo Linux con intel-gpu-tools y sudo
}
INTEL_GPU_TOP_CMD = ["sudo", "-n", "intel_gpu_top", "-J", "-s", "2000"]
HW_HISTORY_S = 3600              # muestras en memoria para promedios del resumen

# ==============================================================
# 10. HEARTBEAT, STATUS Y LOGS
# ==============================================================
HEARTBEAT_INTERVAL_S = 10
STATUS_INTERVAL_S = 30
MAIN_LOOP_TICK_S = 1.0
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(message)s"


def camera_source():
    """Fuente que se entrega a OpenCV segun CAMERA_SOURCE_TYPE."""
    return {"rtsp": RTSP_URL, "file": VIDEO_FILE_PATH, "webcam": WEBCAM_INDEX}[CAMERA_SOURCE_TYPE]


def duracion_combinacion_s() -> float:
    return DURACION_COMBINACION_MIN * 60


def max_models_per_combination() -> int:
    """Mayor numero de modelos simultaneos en cualquier combinacion."""
    return max(len(COMBINATIONS[name]) for name in COMBINATION_ORDER)


def openvino_device() -> str:
    """Dispositivo en formato Ultralytics ('intel:cpu' o 'intel:gpu')."""
    return OPENVINO_DEVICE_MAP[DEVICE]


def validate_config() -> list[str]:
    """Revisa la configuracion y devuelve TODOS los errores juntos (vacia = OK)."""
    errors = []
    if DEVICE not in OPENVINO_DEVICE_MAP:
        errors.append(f"DEVICE debe ser 'cpu' o 'gpu' (es {DEVICE!r})")
    if CAMERA_SOURCE_TYPE not in ("rtsp", "file", "webcam"):
        errors.append(f"CAMERA_SOURCE_TYPE invalido: {CAMERA_SOURCE_TYPE!r}")
    elif CAMERA_SOURCE_TYPE == "rtsp" and not RTSP_URL:
        errors.append("Falta la variable de entorno LAB2_RTSP_URL (ver seccion 4 de config.py)")
    elif CAMERA_SOURCE_TYPE == "file" and not Path(VIDEO_FILE_PATH).is_file():
        errors.append(f"VIDEO_FILE_PATH no existe: {VIDEO_FILE_PATH!r}")
    if RESUME_MODE not in ("auto", "ask", "never"):
        errors.append(f"RESUME_MODE invalido: {RESUME_MODE!r}")
    if not COMBINATION_ORDER or len(set(COMBINATION_ORDER)) != len(COMBINATION_ORDER):
        errors.append("COMBINATION_ORDER debe tener combinaciones sin repetir")
    for name in COMBINATION_ORDER:
        if name not in COMBINATIONS:
            errors.append(f"COMBINATION_ORDER usa una combinacion inexistente: {name}")
    if not BASE_MODELS.is_dir():
        errors.append(f"No existe la carpeta de modelos BASE_MODELS = {BASE_MODELS}")
    positives = {
        "DURACION_COMBINACION_MIN": DURACION_COMBINACION_MIN, "SUMMARY_WINDOW_S": SUMMARY_WINDOW_S,
        "ROTATE_EVERY_S": ROTATE_EVERY_S, "HW_SAMPLE_INTERVAL_S": HW_SAMPLE_INTERVAL_S,
        "BATCH_WRITE_FRAMES": BATCH_WRITE_FRAMES, "HEARTBEAT_INTERVAL_S": HEARTBEAT_INTERVAL_S,
        "SAVE_FRAMES_EVERY_S": SAVE_FRAMES_EVERY_S, "FRAME_WAIT_TIMEOUT_S": FRAME_WAIT_TIMEOUT_S,
    }
    for name, value in positives.items():
        if value <= 0:
            errors.append(f"{name} debe ser > 0")
    if RESERVED_CPUS < 0:
        errors.append("RESERVED_CPUS debe ser >= 0")
    if THREADS_PER_MODEL is not None and THREADS_PER_MODEL < 1:
        errors.append("THREADS_PER_MODEL debe ser None o >= 1")
    return errors
