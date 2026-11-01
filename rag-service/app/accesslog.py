"""One JSON line per /ask request: who (key name, never the secret), how long, which mode, tokens and cost.

The question and the answer text are not logged: questions can describe a person's immigration situation.
"""
import json
import logging
import sys
from datetime import datetime, timezone

logger = logging.getLogger("opt_compass.access")


def configure() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def log_request(**fields) -> None:
    logger.info(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **fields}))
