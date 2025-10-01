import logging

# --- Logging Setup (No changes) ---
def setup_logging(log_level: str):
    log_format = f"[%(levelname)s] [PID:%(process)d] %(message)s"
    logging.basicConfig(level=log_level.upper(), format=log_format)
    # Suppress verbose Uvicorn logs from showing up in the main logger
    for name in logging.root.manager.loggerDict:
        if name.startswith("uvicorn"):
            logging.getLogger(name).handlers = []
    return logging.getLogger(__name__)


logger = setup_logging("INFO")
