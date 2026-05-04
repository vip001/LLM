"""Repository root and default paths for RAG / vectorstore (shared by server and MCP)."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv as _load_dotenv

# llm_common/llm_common/paths.py -> parents[2] = <llm repo root>
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
SERVER_DIR: Path = REPO_ROOT / "server"
DEFAULT_VECTORSTORE_DIR: Path = SERVER_DIR / "vectorstore"
ENV_PATH: Path = REPO_ROOT / ".env"


class PathsUtil:
    """Shared path helpers for server, MCP, and scripts."""

    @staticmethod
    def load_repo_dotenv(*, override: bool = False, env_path: Path | None = None) -> None:
        """Load ``.env`` from repo root if the file exists; does not override existing vars by default.

        In Docker, skip loading (use compose ``env_file`` / ``environment`` instead).
        """
        if Path("/.dockerenv").is_file():
            return
        path = ENV_PATH if env_path is None else env_path
        if path.is_file():
            _load_dotenv(path, override=override)

    @staticmethod
    def get_vectorstore_dir() -> Path:
        if Path("/.dockerenv").is_file():
            return Path("/app/server/vectorstore")
        return DEFAULT_VECTORSTORE_DIR
