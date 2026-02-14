"""
ALCF inference endpoints provider.

Supports the Sophia (vLLM) and Metis (SambaNova) clusters at
``https://inference-api.alcf.anl.gov``.

Authentication is via a Globus access token:

1. ``ALCF_ACCESS_TOKEN`` environment variable, **or**
2. Automatic invocation of ``inference_auth_token.py get_access_token``.

See https://docs.alcf.anl.gov/services/inference-endpoints/
"""

import logging
import os
import subprocess

from ..config import get_llm_timeout

logger = logging.getLogger(__name__)

_CLUSTER_PATHS: dict[str, str] = {
    "sophia": "/resource_server/sophia/vllm/v1",
    "metis": "/resource_server/metis/api/v1",
}


def create_llm(config: dict, temperature: float):
    """Create a ``ChatOpenAI`` instance pointed at an ALCF cluster."""
    from langchain_openai import ChatOpenAI

    cluster = config.get("alcf_cluster") or "sophia"
    path = _CLUSTER_PATHS.get(cluster)
    if path is None:
        raise ValueError(
            f"Unknown ALCF cluster '{cluster}'. "
            f"Supported: {', '.join(sorted(_CLUSTER_PATHS))}"
        )

    base_url = f"https://inference-api.alcf.anl.gov{path}"
    api_key = _get_token()

    return ChatOpenAI(
        model=config["model"],
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        max_retries=0,
        timeout=float(get_llm_timeout()),
    )


# ------------------------------------------------------------------
# Token helpers
# ------------------------------------------------------------------

# Globus app constants – configurable via env vars, defaults mirror
# inference_auth_token.py from the ALCF inference-endpoints repo.
_APP_NAME = os.environ.get("GLOBUS_APP_NAME", "inference_app")
_AUTH_CLIENT_ID = os.environ.get("GLOBUS_AUTH_CLIENT_ID")
_GATEWAY_CLIENT_ID = os.environ.get("GLOBUS_GATEWAY_CLIENT_ID")
_GATEWAY_SCOPE = os.environ.get(
    "GLOBUS_GATEWAY_SCOPE",
    f"https://auth.globus.org/scopes/{_GATEWAY_CLIENT_ID}/action_all",
)


def _get_token() -> str:
    """
    Obtain a Globus access token for ALCF inference endpoints.

    Resolution order:
      1. ``ALCF_ACCESS_TOKEN`` environment variable (fastest, good for CI).
      2. ``globus_sdk`` (``pip install aure[alcf]``) — reuses cached tokens.
      3. Run ``inference_auth_token.py get_access_token`` (subprocess fallback).

    Raises:
        RuntimeError: If no token can be obtained.
    """
    # 1. Explicit env-var -------------------------------------------------
    token = os.environ.get("ALCF_ACCESS_TOKEN")
    if token:
        return token

    # 2. globus_sdk (preferred) -------------------------------------------
    try:
        import globus_sdk  # noqa: F811

        app = globus_sdk.UserApp(
            _APP_NAME,
            client_id=_AUTH_CLIENT_ID,
            scope_requirements={
                _GATEWAY_CLIENT_ID: [_GATEWAY_SCOPE],
            },
            config=globus_sdk.GlobusAppConfig(
                request_refresh_tokens=True,
            ),
        )
        auth = app.get_authorizer(_GATEWAY_CLIENT_ID)
        auth.ensure_valid_token()
        logger.debug("[LLM] ALCF token obtained via globus_sdk")
        return auth.access_token
    except ImportError:
        logger.debug(
            "[LLM] globus_sdk not installed — install with: pip install aure[alcf]"
        )
    except Exception as exc:
        logger.warning("[LLM] globus_sdk token retrieval failed: %s", exc)

    # 3. subprocess fallback ----------------------------------------------
    try:
        result = subprocess.run(
            ["python", "inference_auth_token.py", "get_access_token"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        logger.warning(
            "[LLM] inference_auth_token.py exited with code %d: %s",
            result.returncode,
            result.stderr.strip(),
        )
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("[LLM] Failed to run inference_auth_token.py: %s", exc)

    raise RuntimeError(
        "Could not obtain ALCF access token. Set ALCF_ACCESS_TOKEN, "
        "install globus_sdk (pip install aure[alcf]) and authenticate, "
        "or place inference_auth_token.py on PATH."
    )
