import sys

from loguru import logger


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level="INFO",
    )


def main():
    _configure_logging()

    from brahe_mcp.db import init_db
    from brahe_mcp.server import mcp

    logger.info("Initializing brahe-mcp server")
    init_db()
    mcp.run()


if __name__ == "__main__":
    main()
