"""Harness-only import-origin logger (prepended to PYTHONPATH of tool runs).

Every Python process a tool starts (including spawned pool workers) records
where ``ecgfeat`` and ``ecgagent`` were resolved from, so the harness can
prove the tree under test was used and not the editable install of another
checkout.  It never changes import resolution: the finder only observes the
spec that the regular path machinery finds, then declines.  Afterwards the
next ``sitecustomize`` on ``sys.path`` (e.g. the distribution's own) is
executed so this shim is transparent.
"""

import importlib.machinery
import importlib.util
import json
import os
import sys

_LOG = os.environ.get("ECGFEAT_HARNESS_IMPORT_LOG")
_WATCHED = ("ecgfeat", "ecgagent")
_HERE = os.path.dirname(os.path.abspath(__file__))


class _OriginLogger:
    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if name not in _WATCHED or not _LOG:
            return None
        try:
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            origin = getattr(spec, "origin", None) if spec is not None else None
            with open(_LOG, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "module": name,
                                         "origin": origin}) + "\n")
        except Exception:
            pass
        return None


if _LOG:
    sys.meta_path.insert(0, _OriginLogger)


def _chain_next_sitecustomize():
    for entry in sys.path:
        try:
            if not entry or os.path.abspath(entry) == _HERE:
                continue
        except Exception:
            continue
        candidate = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(candidate):
            try:
                spec = importlib.util.spec_from_file_location("_harness_chained_sitecustomize", candidate)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception:
                pass
            return


_chain_next_sitecustomize()
