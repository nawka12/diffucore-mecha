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


def _accepts_weight(ms) -> bool:
    """True if a model-slot input can bind a ``weight``-space model — i.e. its
    declared space is a generic symbol that includes ``weight``, a fixed set
    containing ``weight``, or a single fixed ``weight``."""
    sub = getattr(ms, "merge_spaces", None)              # a generic merge-space symbol
    if sub is not None:
        return any(getattr(s, "identifier", None) == "weight" for s in sub)
    if isinstance(ms, (set, frozenset)):                 # a fixed set of spaces
        return any(getattr(s, "identifier", None) == "weight" for s in ms)
    return getattr(ms, "identifier", None) == "weight"   # a single fixed space


def _inputs_accept_weight(m) -> bool:
    """True if every *model-slot* input of ``m`` can bind a ``weight``-space model.
    Hyperparameters (``param`` space) are skipped — they're scalar knobs, not slots."""
    spaces = m.get_input_merge_spaces()
    slots = [*spaces.args, *spaces.kwargs.values()]
    if m.get_param_names().has_varargs():
        slots.append(spaces.vararg)
    return all(_accepts_weight(ms) for ms in slots if not _is_param_space(ms))


def _output_space(m) -> str | None:
    """The merge space a *standalone* single-method merge of ``m`` would produce,
    or ``None`` if it can't run standalone from this flat UI's weight model files.

    ``sd_mecha.merge`` pins the final output to one space (``strict_merge_space``),
    so the return space decides the artifact:

    * ``"weight"`` — a generic return *symbol* (propagates from the inputs) or a
      fixed ``weight`` return: an ordinary merge producing a model.
    * ``"delta"`` — a method that consumes weights and fixes its return to ``delta``
      (e.g. ``subtract``): *delta extraction*, producing a difference model.

    Returns ``None`` for mask builders (fixed ``param`` return) and for methods that
    fix their return to ``delta`` but need ``delta`` *inputs* (e.g. ``ties_sum``) —
    neither can run from weight model files alone. The weight/symbol case is left
    ungated on inputs so it keeps the exact set of ordinary merges shown before.
    """
    ident = getattr(m.get_return_type().data.merge_space, "identifier", None)
    if ident is None or ident == "weight":
        return "weight"
    if ident == "delta" and _inputs_accept_weight(m):
        return "delta"
    return None


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
    merge — e.g. a getter that returns a non-model, or a building block that can't
    run standalone in this flat UI (a mask, or a delta op needing delta inputs; see
    :func:`_output_space`)."""
    import inspect
    out_space = "weight"
    try:
        ret = m.get_return_type().data
        if not _is_model_iface(ret.interface):
            return None
        out_space = _output_space(m)
        if out_space is None:
            return None
    except Exception:
        pass  # fail-open: if the return type can't be read, keep it as a plain merge

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
        "params": params, "doc": doc[:800], "output_space": out_space,
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

    # sd_mecha.merge pins the output to "weight" by default; a delta-extraction
    # method (e.g. subtract) produces a "delta" instead, so match the merge space
    # to what the method actually returns or finalization rejects it.
    try:
        strict_space = _output_space(method) or "weight"
    except Exception:
        strict_space = "weight"

    api.broadcast({"type": f"ext:{NAME}", "status": "running", "output": out_name})
    sd_mecha.merge(
        recipe,
        output=str(out_path),
        merge_device=device,
        output_dtype=getattr(torch, _DTYPES[dtype]),
        strict_merge_space=strict_space,
        tqdm=_job_tqdm(job, api),
    )
    api.broadcast({"type": f"ext:{NAME}", "status": "done", "output": out_name})
    return {"info": f"merged → {out_name}", "output": out_name, "ext": NAME}


# ── LoRA / LoHa / LoKr baking (self-contained — no sd-mecha needed) ───────────
# Bakes one or more low-rank adapters into a base model:
#     merged = base + Σ strengthᵢ · deltaᵢ
# the same delta-reconstruction idea as the reference AnimaPulse LoKr baker,
# generalised to LoRA / LoHa / LoKr and to both key conventions in the wild:
#   • kohya / LyCORIS — underscore-flattened, e.g. lora_unet_blocks_0_attn_q_proj
#   • PEFT / diffusers — dot-path,            e.g. diffusion_model.blocks.0.…lora_A
# sd-mecha's own LoRA converters only know SD1/SDXL/SD3/Flux, so they can't bake
# into other architectures (Cosmos/Anima, …); this path is architecture-agnostic
# because it matches adapter→base layers by their flattened module path.

# Network-name prefixes an adapter key may carry, longest first so the most
# specific wins (lora_te1_ before lora_).
_ADAPTER_PREFIXES = (
    "lora_unet_", "lora_te1_", "lora_te2_", "lora_te_", "lora_transformer_",
    "lycoris_unet_", "lycoris_", "lora_",
    "model.diffusion_model.", "diffusion_model.", "transformer.", "unet.",
)
# Architecture prefixes a base weight key may carry, longest first.
_BASE_PREFIXES = (
    "model.diffusion_model.", "diffusion_model.", "cond_stage_model.",
    "conditioner.", "first_stage_model.", "transformer.", "net.", "model.",
)
# Trailing factor names that mark an adapter layer; stripping one yields the
# layer's module prefix.
_ALGO_SUFFIXES = (
    ".lokr_w1_a", ".lokr_w1", ".lokr_w2_a", ".lokr_w2", ".lokr_t2",
    ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b", ".hada_t1", ".hada_t2",
    ".lora_down.weight", ".lora_up.weight", ".lora_mid.weight",
    ".lora_A.weight", ".lora_B.weight", ".diff", ".diff_b",
    ".alpha", ".dora_scale",
)


def _strip_prefix(s: str, prefixes) -> str:
    for p in prefixes:
        if s.startswith(p):
            return s[len(p):]
    return s


def _adapter_tail(prefix: str) -> str:
    """Normalised module path of an adapter layer: drop the network-name prefix,
    flatten separators so kohya ``_`` and PEFT ``.`` paths land on one form."""
    return _strip_prefix(prefix, _ADAPTER_PREFIXES).replace(".", "_")


def _base_tail(key: str) -> str | None:
    """Same normalised module path for a base *weight* key, or ``None`` if the key
    isn't a weight we can bake into."""
    if not key.endswith(".weight"):
        return None
    return _strip_prefix(key[:-len(".weight")], _BASE_PREFIXES).replace(".", "_")


def _module_prefix(key: str) -> str | None:
    for suf in _ALGO_SUFFIXES:
        if key.endswith(suf):
            return key[:-len(suf)]
    return None


def _scale(alpha, dim) -> float:
    """LoRA-style scale α/dim, with α stored as a 0-dim tensor. Falls back to 1.0
    when there's no alpha or no rank (e.g. a raw ``diff`` weight)."""
    if alpha is None or dim is None:
        return 1.0
    a = float(alpha.float().item())
    if a != a or a in (float("inf"), float("-inf")):  # nan/inf guard
        return 1.0
    return a / dim


def _lokr_delta(get, target_shape):
    import torch
    w1, w1_a, w1_b = get("lokr_w1"), get("lokr_w1_a"), get("lokr_w1_b")
    dim = None
    if w1 is not None:
        W1 = w1.float()
    elif w1_a is not None and w1_b is not None:
        W1 = w1_a.float() @ w1_b.float()
        dim = w1_b.shape[0]
    else:
        raise KeyError("no lokr w1 factors")

    w2, w2_a, w2_b, t2 = get("lokr_w2"), get("lokr_w2_a"), get("lokr_w2_b"), get("lokr_t2")
    if t2 is not None and w2_a is not None and w2_b is not None:
        W2 = torch.einsum("i j ..., i p, j r -> p r ...", t2.float(), w2_a.float(), w2_b.float())
        dim = dim if dim is not None else w2_b.shape[0]
    elif w2_a is not None and w2_b is not None:
        W2 = w2_a.float() @ w2_b.float().flatten(1)
        dim = dim if dim is not None else w2_b.shape[0]
    elif w2 is not None:
        W2 = w2.float()
    else:
        raise KeyError("no lokr w2 factors")

    scale = _scale(get("alpha"), dim)
    while W1.dim() < W2.dim():
        W1 = W1.unsqueeze(-1)
    return (torch.kron(W1, W2) * scale).reshape(target_shape)


def _loha_delta(get, target_shape):
    import torch
    w1a, w1b = get("hada_w1_a"), get("hada_w1_b")
    w2a, w2b = get("hada_w2_a"), get("hada_w2_b")
    if any(x is None for x in (w1a, w1b, w2a, w2b)):
        raise KeyError("incomplete loha factors")
    t1, t2 = get("hada_t1"), get("hada_t2")
    if t1 is not None and t2 is not None:
        m1 = torch.einsum("i j ..., i p, j r -> p r ...", t1.float(), w1a.float(), w1b.float())
        m2 = torch.einsum("i j ..., i p, j r -> p r ...", t2.float(), w2a.float(), w2b.float())
    else:
        m1 = w1a.float() @ w1b.float().flatten(1)
        m2 = w2a.float() @ w2b.float().flatten(1)
    scale = _scale(get("alpha"), w1b.shape[0])
    return (m1 * m2 * scale).reshape(target_shape)


def _lora_delta(get, target_shape):
    up, down = get("lora_up.weight"), get("lora_down.weight")
    if up is None and down is None:  # PEFT/diffusers naming
        up, down = get("lora_B.weight"), get("lora_A.weight")
    if up is None or down is None:
        raise KeyError("no lora up/down factors")
    if get("lora_mid.weight") is not None:
        raise KeyError("CP-decomposed LoRA (lora_mid) not supported")
    up, down = up.float(), down.float()
    delta = up.reshape(up.shape[0], -1) @ down.reshape(down.shape[0], -1)
    return (delta * _scale(get("alpha"), down.shape[0])).reshape(target_shape)


def _adapter_delta(tensors: dict, prefix: str, target_shape):
    """Reconstruct one layer's float32 weight delta from its adapter factors,
    auto-detecting the algorithm. Raises ``KeyError`` for layers we can't (or
    won't) reconstruct — the caller skips those rather than aborting the bake."""
    def get(suffix):
        return tensors.get(f"{prefix}.{suffix}")

    if get("lokr_w1") is not None or get("lokr_w1_a") is not None:
        return _lokr_delta(get, target_shape)
    if get("hada_w1_a") is not None:
        return _loha_delta(get, target_shape)
    if get("diff") is not None:
        return get("diff").float().reshape(target_shape)
    if any(get(s) is not None for s in
           ("lora_up.weight", "lora_down.weight", "lora_A.weight", "lora_B.weight")):
        return _lora_delta(get, target_shape)
    raise KeyError("unrecognised adapter layer")


def _load_adapter(path: str, device: str):
    """Read an adapter file and index its layers by normalised module path
    (tail → module prefix) so base weights can be matched in one pass."""
    from safetensors import safe_open
    tensors: dict = {}
    with safe_open(path, framework="pt", device=device) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    index: dict[str, str] = {}
    for k in tensors:
        mp = _module_prefix(k)
        if mp is not None:
            index[_adapter_tail(mp)] = mp
    return tensors, index


def _run_lora_merge(job, *, api, base_path, adapters, out_path, out_name, device, dtype):
    """Bake the adapters into the base, streaming the base one weight at a time so
    a multi-GB model never has to be held twice. Runs on the shared job worker."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    out_dtype = getattr(torch, _DTYPES[dtype])

    loaded = []
    for ad in adapters:
        tensors, index = _load_adapter(str(ad["path"]), device)
        loaded.append({"name": ad["name"], "strength": float(ad["strength"]),
                       "tensors": tensors, "index": index, "layers": len(index),
                       "applied": 0})

    with safe_open(str(base_path), framework="pt", device=device) as f:
        base_keys = list(f.keys())
        base_meta = f.metadata() or {}

    api.broadcast({"type": f"ext:{NAME}", "status": "running", "output": out_name})

    tq = _job_tqdm(job, api)(total=len(base_keys), desc="bake")
    merged: dict = {}
    skipped: list[str] = []

    with safe_open(str(base_path), framework="pt", device=device) as f:
        for key in base_keys:
            tensor = f.get_tensor(key)
            tail = _base_tail(key)
            acc = None
            if tail is not None:
                for L in loaded:
                    prefix = L["index"].get(tail)
                    if prefix is None:
                        continue
                    try:
                        delta = _adapter_delta(L["tensors"], prefix, tensor.shape)
                    except (KeyError, RuntimeError, ValueError) as e:
                        if len(skipped) < 20:
                            skipped.append(f"{L['name']}:{prefix} ({e})")
                        continue
                    if acc is None:
                        acc = tensor.float()
                    acc = acc + L["strength"] * delta
                    L["applied"] += 1
            merged[key] = (acc if acc is not None else tensor.float()).to(out_dtype).cpu()
            tq.update(1)
    tq.close()

    save_file(merged, str(out_path), metadata=base_meta)

    parts = [f"{L['name']} {L['applied']}/{L['layers']}" for L in loaded]
    info = f"baked → {out_name}  [{'; '.join(parts)}]"
    if skipped:
        info += f"  ({len(skipped)} layer(s) skipped)"
    api.broadcast({"type": f"ext:{NAME}", "status": "done", "output": out_name})
    return {"info": info, "output": out_name, "ext": NAME, "skipped": skipped}


# ── request models ───────────────────────────────────────────────────────────

class MergeRequest(BaseModel):
    method: str
    folder: str = "checkpoints"
    models: list[str] = []
    params: dict = {}
    output: str = ""
    device: str = "cpu"
    dtype: str = "fp16"


class LoraSpec(BaseModel):
    name: str
    strength: float = 1.0


class LoraMergeRequest(BaseModel):
    folder: str = "checkpoints"
    base: str = ""
    loras: list[LoraSpec] = []
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

    def _loras_dir():
        return (api.root_dir / "models" / "loras").resolve()

    @router.post("/lora/merge")
    def lora_merge(req: LoraMergeRequest):
        d = _models_dir(req.folder)
        base_path = _resolve_input(d, req.base)

        ld = _loras_dir()
        specs = [s for s in req.loras if s.name]
        if not specs:
            raise HTTPException(400, "select at least one LoRA")
        adapters = [
            {"name": s.name, "path": _resolve_input(ld, s.name), "strength": s.strength}
            for s in specs
        ]

        out_path, out_name = _resolve_output(d, req.output)
        device = req.device if req.device in ("cpu", "cuda") else "cpu"
        dtype = req.dtype if req.dtype in _DTYPES else "fp16"

        def run(job):
            return _run_lora_merge(
                job, api=api, base_path=base_path, adapters=adapters,
                out_path=out_path, out_name=out_name, device=device, dtype=dtype,
            )

        job_id = api.enqueue_job(f"lora bake → {out_name}", run, kind="merge")
        return {"job": job_id, "output": out_name}

    api.add_api_router(router)
