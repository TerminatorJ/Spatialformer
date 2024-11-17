__version__ = "0.0.1"
import logging
import sys

logger = logging.getLogger("Spatialformer")
# check if logger has been initialized
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    

# from . import preprocess as pp
from . import tools as tl
from .data_loader import create_dataloader_eval
# from .GraphSAGE import *
