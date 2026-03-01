import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_NOTEBOOK_PATH = Path("notebooks/06_High-precision_QA_Legal_Reasoning_Engine.ipynb")
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

    def _selected_code_cells(self) -> List[tuple]:
        if not self.notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {self.notebook_path}")

        nb = json.loads(self.notebook_path.read_text(encoding="utf-8"))
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
                f"No executable code cells found at positions {self.exec_cell_numbers} in {self.notebook_path}"
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

            try:
                for global_cell_num, src in self._selected_code_cells():
                    compiled = compile(
                        src,
                        filename=f"{self.notebook_path}::cell_{global_cell_num}",
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
                _ENGINE_SINGLETON = NotebookEngine()
    return _ENGINE_SINGLETON
