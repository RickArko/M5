"""``python -m m5.serve`` — boot uvicorn with env-driven settings."""

from __future__ import annotations

import uvicorn

from m5.serve.config import ServeSettings


def main() -> None:
    s = ServeSettings()
    uvicorn.run(
        "m5.serve.app:create_app",
        factory=True,
        host=s.host,
        port=s.port,
        workers=s.workers,
        # We configure logging via loguru in `configure_logging`; suppress uvicorn's default.
        log_config=None,
    )


if __name__ == "__main__":
    main()
