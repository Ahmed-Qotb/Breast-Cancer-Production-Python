import logging

from production_app.config import config

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)
