"""Diffucore UI ⇄ sd-mecha — model merging.

This extension exposes `sd-mecha <https://github.com/ljleb/sd-mecha>`_'s merge
methods as a tab in Diffucore UI. You pick a merge method, the input models, and
the method's hyperparameters; the merge runs on the shared job worker and the
result is written into ``models/`` where the app can load it.

Designed for zero maintenance
-----------------------------
The list of merge methods and *every* method's parameters are read out of
sd-mecha **at runtime** — nothing about a specific method is hardcoded here. So
if you don't touch this extension for a year while sd-mecha keeps shipping new
merge methods, those new methods show up in the UI automatically, with their own
parameter fields, the next time you open the tab.

The only sd-mecha surface this depends on is its stable introspection core — the
same machinery sd-mecha uses internally for every method:

* ``sd_mecha.extensions.merge_methods.get_all()`` / ``get_all_converters()``
* ``MergeMethod.get_param_names() / get_input_types() / get_input_merge_spaces()
  / get_default_args() / get_return_type()``  (each returns a ``FunctionArgs``
  with ``.args`` / ``.vararg`` / ``.kwargs`` / ``.has_varargs()``)
* ``sd_mecha.model(path)`` and ``sd_mecha.merge(recipe, output=...)``

A new *merge method* is just a function registered against that core, so it is
introspectable with no code change here. Every introspection call is wrapped so
that if some future method doesn't fit the assumptions, it's skipped rather than
breaking the whole list (fail-open: when unsure, show the method).

sd-mecha is an opt-in dependency (see ``requirements.txt``). It is imported
lazily — the extension loads fine without it and the UI tells you how to install
it, per Diffucore's RCE-safe "pip is opt-in" install policy.
"""

from __future__ import annotations

import re
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

NAME = "diffucore-mecha"

# Source/target folders the UI offers, mapped to subdirs of models/. The model
# list itself comes from the app's own /api/models endpoint (frontend side), so
# we never duplicate the app's scan logic — we only need the dir to resolve a
# chosen filename to a path and to place the output.
_FOLDERS = {"checkpoints": "checkpoints", "diffusion-models": "diffusion-models"}
_DTYPES = {"fp16": "float16", "fp32": "float32", "bf16": "bfloat16"}


# ── sd-mecha introspection (the forward-compatible part) ─────────────────────
# These helpers turn a sd_mecha MergeMethod into a plain JSON description the
# frontend can render. They assume nothing about which methods exist.

def _is_param_space(ms) -> bool:
    """True if a parameter is a *hyperparameter* rather than a model slot.

    sd-mecha marks scalars you tweak (alpha, beta, …) with the merge space
    ``"param"`` — note this is independent of the Python type: ``weighted_sum``'s
    ``alpha`` is typed ``Tensor`` but lives in ``"param"`` space. Model slots
    carry a real space (``weight`` / ``delta`` / …) or the generic space symbol.
    So merge space — not the type — is the correct discriminator.
    """
    try:
        if isinstance(ms, (set, frozenset)):
            return {getattr(s, "identifier", None) for s in ms} == {"param"}
    except TypeError:
        pass
    return False


def _space_label(ms) -> str:
    """A short merge-space hint for a model slot (e.g. ``"delta"``), or ``""`` for
    the generic/weight space (no hint worth showing)."""
    try:
        if isinstance(ms, (set, frozenset)) and len(ms) == 1:
            ident = next(iter(ms)).identifier
            return "" if ident in ("param", "weight") else ident
    except (TypeError, AttributeError):
        pass
    return ""


def _kind(itype, default=None) -> str:
    """Widget kind for a hyperparameter.

    sd-mecha often types a scalar knob as ``Parameter(Tensor)`` (e.g.
    ``weighted_sum``'s ``alpha: Parameter(Tensor) = 0.5``), so the interface type
    isn't always a plain literal. When it isn't, fall back to the default value's
    type — that's what tells us a tensor-typed knob is really a float/int/bool.
    """
    import typing
    origin = typing.get_origin(itype) or itype
    if origin is bool:
        return "bool"
    if origin is int:
        return "int"
    if origin is float:
        return "float"
    if origin is str:
        return "str"
    # interface isn't a plain literal (e.g. a scalar Tensor) — infer from default
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "str"
    return "float"  # scalar knob of unknown type — treat as a number


def _is_model_iface(itype) -> bool:
    import typing
    import torch
    from sd_mecha.extensions.merge_methods import StateDict
    try:
        origin = typing.get_origin(itype) or itype
        return isinstance(origin, type) and issubclass(origin, (torch.Tensor, StateDict))
    except TypeError:
        return False


def _json_default(v):
    return v if isinstance(v, (int, float, bool, str)) else None


def _param_specs(m):
    """Split a method's inputs into (model slots, accepts-varargs, hyperparams).

    Model slots and hyperparameters are told apart by merge space (see
    :func:`_is_param_space`); defaults are pulled from the signature so the UI
    can prefill them.
    """
    names = m.get_param_names()
    types = m.get_input_types()
    spaces = m.get_input_merge_spaces()
    defaults = m.get_default_args()

    # get_default_args().args holds defaults for the *trailing* positional params
    # that have one, so align by offset.
    n_args = len(names.args)
    def_offset = n_args - len(defaults.args)

    models, params = [], []
    for i, pname in enumerate(names.args):
        ms = spaces.args[i]
        if _is_param_space(ms):
            default = defaults.args[i - def_offset] if i >= def_offset else None
            params.append({
                "name": pname, "kind": _kind(types.args[i], default),
                "default": _json_default(default), "required": i < def_offset,
            })
        else:
            models.append({"name": pname, "space": _space_label(ms)})

    varargs = bool(names.has_varargs()) and not _is_param_space(spaces.vararg)

    for pname in names.kwargs:
        ms = spaces.kwargs[pname]
        if _is_param_space(ms):
            has_def = pname in defaults.kwargs
            kw_default = defaults.kwargs.get(pname)
            params.append({
                "name": pname, "kind": _kind(types.kwargs[pname], kw_default),
                "default": _json_default(kw_default),
                "required": not has_def,
            })
        # A keyword-only *model* input can't be positioned in this flat UI; skip
        # it. The method still appears, driven by its positional model slots.

    return models, varargs, params


def _describe(m):
    """Full JSON description of a method, or ``None`` if it isn't a user-facing
    merge (e.g. a dtype/device getter that returns a non-model)."""
    import inspect
    try:
        ret_iface = m.get_return_type().data.interface
        if not _is_model_iface(ret_iface):
            return None
    except Exception:
        pass  # fail-open: if the return type can't be read, keep the method

    models, varargs, params = _param_specs(m)
    if not models and not varargs:
        return None

    doc = ""
    try:
        doc = (inspect.getdoc(m.__wrapped__) or "").strip()
    except Exception:
        pass
    return {
        "id": m.identifier, "models": models, "varargs": varargs,
        "params": params, "doc": doc[:800],
    }


def _coerce_params(method, raw: dict) -> dict:
    """Cast the form's raw param values to the types sd-mecha expects, dropping
    blanks (so the method's own default is used) and anything the method doesn't
    accept (so a stale UI can't pass an unexpected keyword)."""
    try:
        _models, _varargs, specs = _param_specs(method)
    except Exception:
        return {}
    out = {}
    for p in specs:
        name = p["name"]
        if name not in raw:
            continue
        v = raw[name]
        if v is None or v == "":
            continue
        kind = p["kind"]
        try:
            if kind == "bool":
                v = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
            elif kind == "int":
                v = int(float(v))
            elif kind == "float":
                v = float(v)
            else:
                v = str(v)
        except (ValueError, TypeError):
            continue
        out[name] = v
    return out


# ── merge execution (runs on the shared job worker) ──────────────────────────

def _job_tqdm(job, api):
    """A ``tqdm`` subclass that reports merge progress to the shared queue/SSE UI
    and honours cancellation. Subclassing the real tqdm means sd-mecha's exact
    tqdm usage keeps working — we only add the reporting."""
    from tqdm import tqdm as _tqdm
    last = [0.0]

    class _JobTqdm(_tqdm):
        def update(self, n=1):
            if job.cancel.is_set():
                raise RuntimeError("merge cancelled")
            r = super().update(n)
            now = time.monotonic()
            if self.total and now - last[0] >= 0.2:  # throttle the broadcast
                last[0] = now
                job.step, job.total = int(self.n), int(self.total)
                api.broadcast({"type": "progress", "job": job.id,
                               "step": int(self.n), "total": int(self.total)})
            return r

    return _JobTqdm


def _run_merge(job, *, api, method_id, input_paths, raw_params,
               out_path, out_name, device, dtype):
    import sd_mecha
    import torch
    from sd_mecha.extensions import merge_methods

    method = merge_methods.resolve(method_id)
    kwargs = _coerce_params(method, raw_params)
    nodes = [sd_mecha.model(str(p)) for p in input_paths]
    recipe = method(*nodes, **kwargs)

    api.broadcast({"type": f"ext:{NAME}", "status": "running", "output": out_name})
    sd_mecha.merge(
        recipe,
        output=str(out_path),
        merge_device=device,
        output_dtype=getattr(torch, _DTYPES[dtype]),
        tqdm=_job_tqdm(job, api),
    )
    api.broadcast({"type": f"ext:{NAME}", "status": "done", "output": out_name})
    return {"info": f"merged → {out_name}", "output": out_name, "ext": NAME}


# ── request models ───────────────────────────────────────────────────────────

class MergeRequest(BaseModel):
    method: str
    folder: str = "checkpoints"
    models: list[str] = []
    params: dict = {}
    output: str = ""
    device: str = "cpu"
    dtype: str = "fp16"


# ── entry point ──────────────────────────────────────────────────────────────

def setup(api):
    router = APIRouter()

    def _models_dir(folder: str):
        sub = _FOLDERS.get(folder)
        if not sub:
            raise HTTPException(400, f"unknown folder {folder!r}")
        return (api.root_dir / "models" / sub).resolve()

    def _resolve_input(d, name: str):
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            raise HTTPException(400, f"invalid model name {name!r}")
        p = (d / name).resolve()
        if p.parent != d or not p.is_file():
            raise HTTPException(400, f"model not found: {name}")
        return p

    def _resolve_output(d, name: str):
        base = re.sub(r"[^A-Za-z0-9._-]", "_", (name or "").strip())
        if not base or base in (".", ".."):
            raise HTTPException(400, "invalid output name")
        if not base.lower().endswith(".safetensors"):
            base += ".safetensors"
        p = (d / base).resolve()
        if p.parent != d:
            raise HTTPException(400, "invalid output path")
        if p.exists():
            raise HTTPException(400, f"output already exists: {base}")
        return p, base

    @router.get("")
    def status():
        info = {"installed": False}
        try:
            import sd_mecha  # noqa: F401
            info["installed"] = True
            try:
                from importlib.metadata import version
                info["version"] = version("sd-mecha")
            except Exception:
                info["version"] = getattr(sd_mecha, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            info["error"] = str(e)
        return info

    @router.get("/methods")
    def methods():
        try:
            import sd_mecha  # importing registers every builtin merge method
            from sd_mecha.extensions import merge_methods
        except Exception as e:  # noqa: BLE001
            return {"installed": False, "error": str(e), "methods": []}

        try:
            converters = {c.identifier for c in merge_methods.get_all_converters()}
        except Exception:
            converters = set()  # older/newer sd-mecha without converters API

        out = []
        for m in merge_methods.get_all():
            try:
                if m.identifier in converters:
                    continue  # config converters aren't user-facing merges
                spec = _describe(m)
                if spec:
                    out.append(spec)
            except Exception:
                continue  # one odd method never breaks the whole catalog
        out.sort(key=lambda s: s["id"])
        return {"installed": True, "methods": out}

    @router.post("/merge")
    def merge_models(req: MergeRequest):
        try:
            import sd_mecha  # noqa: F401
            from sd_mecha.extensions import merge_methods
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"sd-mecha is not installed ({e}). "
                                     "pip install sd-mecha, then Reload the extension.")
        try:
            merge_methods.resolve(req.method)
        except Exception:
            raise HTTPException(400, f"unknown merge method: {req.method!r}")

        d = _models_dir(req.folder)
        names = [n for n in req.models if n]
        if not names:
            raise HTTPException(400, "select at least one model to merge")
        input_paths = [_resolve_input(d, n) for n in names]
        out_path, out_name = _resolve_output(d, req.output)
        device = req.device if req.device in ("cpu", "cuda") else "cpu"
        dtype = req.dtype if req.dtype in _DTYPES else "fp16"

        def run(job):
            return _run_merge(
                job, api=api, method_id=req.method, input_paths=input_paths,
                raw_params=req.params, out_path=out_path, out_name=out_name,
                device=device, dtype=dtype,
            )

        job_id = api.enqueue_job(f"mecha merge → {out_name}", run, kind="merge")
        return {"job": job_id, "output": out_name}

    api.add_api_router(router)
