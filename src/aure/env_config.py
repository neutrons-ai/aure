"""Environment-variable loading for AuRE.

AuRE reads its LLM / fit configuration from environment variables (see
``.env.example``).  Those can be supplied three ways, in decreasing
precedence:

1. the real shell environment (whatever is already in ``os.environ``);
2. a project-local ``.env`` file (found by walking up from the CWD);
3. a per-user ``~/.aure`` file — same ``KEY=value`` format as ``.env``,
   acting as machine-wide defaults.

``load_env()`` is the single entry point; both the CLI and the MCP server
call it instead of bare ``dotenv.load_dotenv()`` so the ``~/.aure`` fallback
is honoured everywhere.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

#: Per-user config file, same format as ``.env``.
USER_ENV_PATH = Path.home() / ".aure"


def load_env() -> None:
    """Populate ``os.environ`` from ``.env`` then ``~/.aure``.

    Neither source overrides variables already present in the real
    environment, and the project-local ``.env`` takes precedence over
    ``~/.aure`` (loaded first, ``override=False`` leaves its values in
    place).  Both files are optional.
    """
    # Project-local .env first — found by walking up from the CWD.
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
        logger.debug("[ENV] Loaded %s", dotenv_path)

    # Per-user ~/.aure fills in anything still unset.
    if USER_ENV_PATH.is_file():
        load_dotenv(USER_ENV_PATH)
        logger.debug("[ENV] Loaded %s", USER_ENV_PATH)
