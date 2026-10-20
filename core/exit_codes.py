"""
Codigos de salida de main.py: el "idioma" entre main.py y el supervisor
(run_lab2.py). Solo libreria estandar.
"""
EXIT_COMPLETED = 0          # todas las combinaciones terminaron: el supervisor se detiene
EXIT_CONFIG_ERROR = 2       # config invalida / sin GPU: relanzar no sirve, se detiene
EXIT_INTERRUPTED = 4        # Ctrl+C del usuario: se detiene (se puede reanudar a mano)
# Cualquier otro codigo (1, -9, ...) = caida inesperada: el supervisor relanza y continua.
