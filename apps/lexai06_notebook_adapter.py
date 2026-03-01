import json
import os
import base64
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


NOTEBOOK_FILENAME = "06_High-precision_QA_Legal_Reasoning_Engine.ipynb"
MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
DEFAULT_NOTEBOOK_PATH = REPO_ROOT / "notebooks" / NOTEBOOK_FILENAME
SNAPSHOT_NOTEBOOK_PATH = MODULE_DIR / "notebook_06_snapshot.ipynb"
# Global cell numbers to execute from notebook JSON.
DEFAULT_EXEC_CELL_NUMBERS = [2, 3, 4, 5, 6, 7, 8, 9]


class NotebookEngine:
    def __init__(
        self,
        notebook_path: Path = DEFAULT_NOTEBOOK_PATH,
        exec_cell_numbers: Optional[List[int]] = None,
    ):
        self.notebook_path = Path(notebook_path)
        self.exec_cell_numbers = exec_cell_numbers or list(DEFAULT_EXEC_CELL_NUMBERS)
        self._lock = threading.RLock()
        self._runtime: Dict = {}
        self._ready = False
        self._ready_error = ""
        self._resolved_notebook_path: Optional[Path] = None

    def _ensure_spark(self):
        spark = self._runtime.get("spark")
        if spark is not None:
            return spark

        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            if spark is None:
                spark = SparkSession.builder.getOrCreate()
            self._runtime["spark"] = spark
            return spark
        except Exception as exc:
            raise RuntimeError(
                "Spark session not available. Run this adapter in Databricks or a Spark-enabled runtime."
            ) from exc

    def _get_dbutils(self):
        # Databricks runtime path 1: DBUtils from pyspark
        try:
            from pyspark.dbutils import DBUtils
            spark = self._ensure_spark()
            return DBUtils(spark)
        except Exception:
            pass

        # Databricks runtime path 2: notebook global `dbutils`
        try:
            import builtins
            dbu = getattr(builtins, "dbutils", None)
            if dbu is not None:
                return dbu
        except Exception:
            pass

        return None

    def _workspace_path_variants(self, raw_path: str) -> List[str]:
        p = (raw_path or "").strip()
        if not p:
            return []

        variants = []

        def _add(x: str):
            x = x.strip()
            if x and x not in variants:
                variants.append(x)

        _add(p)

        # Normalize /Workspace/<...> -> /<...> for workspace API calls
        if p.startswith("/Workspace/"):
            _add("/" + p[len("/Workspace/"):])
        elif p.startswith("/"):
            _add("/Workspace" + p)

        # Add path without .ipynb (workspace notebook object can be extensionless)
        for x in list(variants):
            if x.endswith(".ipynb"):
                _add(x[:-6])
            else:
                _add(x + ".ipynb")

        return variants

    def _decode_export_payload(self, payload) -> str:
        if payload is None:
            return ""
        txt = str(payload)
        if txt.lstrip().startswith("{") and '"cells"' in txt:
            return txt

        # Sometimes export payload can be base64-encoded
        try:
            dec = base64.b64decode(txt).decode("utf-8", errors="ignore")
            if dec.lstrip().startswith("{") and '"cells"' in dec:
                return dec
        except Exception:
            pass
        return ""

    def _try_workspace_export_to_temp(self, candidate_path: str) -> Optional[Path]:
        dbutils = self._get_dbutils()
        if dbutils is None:
            return None

        for ws_path in self._workspace_path_variants(candidate_path):
            try:
                payload = dbutils.workspace.export(ws_path, format="JUPYTER")
                txt = self._decode_export_payload(payload)
                if not txt:
                    continue
                tmp_path = Path(tempfile.gettempdir()) / "lexai06_exported_from_workspace.ipynb"
                tmp_path.write_text(txt, encoding="utf-8")
                return tmp_path
            except Exception:
                continue
        return None

    def _resolve_notebook_path(self) -> Path:
        # Highest priority: explicit environment override.
        env_override = os.environ.get("LEXAI06_NOTEBOOK_PATH", "").strip()
        if env_override:
            p = Path(env_override)
            if p.exists():
                return p
            exported = self._try_workspace_export_to_temp(env_override)
            if exported is not None and exported.exists():
                return exported

        candidates = []

        # Given path as-is and cwd-relative.
        given = Path(self.notebook_path)
        candidates.append(given)
        if not given.is_absolute():
            candidates.append(Path.cwd() / given)

        # Repo-relative path (most reliable for Databricks Repos/workspace files).
        candidates.append(REPO_ROOT / "notebooks" / NOTEBOOK_FILENAME)
        if not given.is_absolute():
            candidates.append(REPO_ROOT / given)
        # Adapter-bundled snapshot fallback (works even if notebooks folder is absent in workspace mount).
        candidates.append(SNAPSHOT_NOTEBOOK_PATH)

        for p in candidates:
            try:
                if p.exists():
                    return p.resolve()
            except Exception:
                continue
            exported = self._try_workspace_export_to_temp(str(p))
            if exported is not None and exported.exists():
                return exported.resolve()

        # Last-resort workspace search.
        search_roots = [Path("/Workspace/Repos"), Path("/Workspace/Users"), Path("/Workspace")]
        for root in search_roots:
            if not root.exists():
                continue
            try:
                for hit in root.rglob(NOTEBOOK_FILENAME):
                    if hit.exists():
                        return hit.resolve()
            except Exception:
                continue

        searched = "\n - ".join([str(x) for x in candidates])
        raise FileNotFoundError(
            f"Notebook not found.\nSearched candidates:\n - {searched}\n"
            f"Also searched roots for {NOTEBOOK_FILENAME}: /Workspace/Repos, /Workspace/Users, /Workspace\n"
            f"Snapshot fallback path checked: {SNAPSHOT_NOTEBOOK_PATH}"
        )

    def _selected_code_cells(self) -> List[tuple]:
        nb_path = self._resolve_notebook_path()
        self._resolved_notebook_path = nb_path

        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        out = []
        for global_idx, cell in enumerate(nb.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            if global_idx not in self.exec_cell_numbers:
                continue
            src = "".join(cell.get("source", []))
            out.append((global_idx, src))
        if not out:
            raise RuntimeError(
                f"No executable code cells found at positions {self.exec_cell_numbers} in {nb_path}"
            )
        return out

    def initialize(self) -> Dict:
        with self._lock:
            if self._ready:
                return self.status()

            start = time.perf_counter()
            self._ensure_spark()
            self._runtime.setdefault("__name__", "__lexai06_runtime__")

            # `displayHTML` exists in Databricks notebooks; provide safe fallback in API runtime.
            if "displayHTML" not in self._runtime:
                self._runtime["displayHTML"] = lambda html: html

            global_cell_num = "unknown"
            try:
                for global_cell_num, src in self._selected_code_cells():
                    compiled = compile(
                        src,
                        filename=f"{self._resolved_notebook_path or self.notebook_path}::cell_{global_cell_num}",
                        mode="exec",
                    )
                    exec(compiled, self._runtime, self._runtime)
            except Exception as exc:
                self._ready = False
                self._ready_error = f"Initialization failed at notebook cell {global_cell_num}: {exc}"
                raise RuntimeError(self._ready_error) from exc

            if "high_precision_answer" not in self._runtime:
                self._ready = False
                self._ready_error = "high_precision_answer function was not loaded from notebook."
                raise RuntimeError(self._ready_error)

            self._ready = True
            self._ready_error = ""
            self._runtime["_adapter_init_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
            return self.status()

    def status(self) -> Dict:
        return {
            "ready": self._ready,
            "error": self._ready_error,
            "notebook_path": str(self.notebook_path),
            "resolved_notebook_path": str(self._resolved_notebook_path) if self._resolved_notebook_path else "",
            "exec_cell_numbers": list(self.exec_cell_numbers),
            "init_ms": self._runtime.get("_adapter_init_ms", 0.0),
            "data_signature": self._runtime.get("data_signature"),
            "records_loaded": len(self._runtime.get("records", []) or []),
            "embedder": self._runtime.get("embedder_name", "na"),
            "reranker": self._runtime.get("reranker_name", "na"),
            "llm_backend": (self._runtime.get("llm_backend") or {}).get("type", "na"),
        }

    def ensure_ready(self) -> None:
        if not self._ready:
            self.initialize()

    def answer_query(self, query: str) -> Dict:
        self.ensure_ready()
        if not query or not str(query).strip():
            raise ValueError("Query cannot be empty")

        with self._lock:
            fn = self._runtime["high_precision_answer"]
            return fn(str(query).strip())

    def latency_profile(self) -> Dict:
        self.ensure_ready()
        with self._lock:
            fn = self._runtime.get("latency_profile")
            if callable(fn):
                return fn()
            return {}


_ENGINE_SINGLETON: Optional[NotebookEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> NotebookEngine:
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is None:
        with _ENGINE_LOCK:
            if _ENGINE_SINGLETON is None:
                env_nb_path = os.environ.get("LEXAI06_NOTEBOOK_PATH", "").strip()
                if env_nb_path:
                    _ENGINE_SINGLETON = NotebookEngine(notebook_path=Path(env_nb_path))
                else:
                    _ENGINE_SINGLETON = NotebookEngine()
    return _ENGINE_SINGLETON
