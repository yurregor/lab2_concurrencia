"""
Registro de eventos operativos (events.jsonl).

Un evento es algo que PASA (se cayo la camara, se cargo un modelo, cambio la
combinacion). Se separa de frames.jsonl para poder dibujarlos como marcas
sobre las series de tiempo sin mezclar tipos de linea.
"""
import logging
import os
from datetime import datetime

from core.writers import JsonlWriter

logger = logging.getLogger(__name__)


class EventLog:
    """Escribe eventos inmediatamente (lote de 1): son pocos y valiosos."""

    def __init__(self, writer: JsonlWriter, run_id: str, schema_version: str):
        self._writer = writer
        self._run_id = run_id
        self._schema = schema_version

    def emit(self, tipo: str, **datos) -> None:
        # pid: tras una caida y reanudacion, permite saber que proceso lo emitio.
        entry = {"schema_version": self._schema, "run_id": self._run_id,
                 "ts": datetime.now().isoformat(), "pid": os.getpid(), "tipo": tipo, **datos}
        self._writer.write(entry)
        self._writer.flush()
        logger.info("EVENTO %s %s", tipo, datos if datos else "")

    def close(self) -> None:
        self._writer.close()
