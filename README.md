# diffucore-mecha

A [Diffucore UI](https://github.com/nawka12/diffucore-ui) extension that merges
model checkpoints with [sd-mecha](https://github.com/ljleb/sd-mecha).

It adds a **Merge** tab: pick a merge method, choose the input models, set the
method's hyperparameters, and the merge runs on the app's shared job worker
(serialized with generation, visible in the queue, cancellable). The result is
written into `models/` where Diffucore can load it.

![tab](https://img.shields.io/badge/Diffucore%20UI-extension-blue)

## Built to not need updates

The list of merge methods — and every method's parameters — is read out of
sd-mecha **at runtime**. Nothing about any specific method is hardcoded.

So if you install this once and never touch it again, but sd-mecha keeps shipping
new merge methods, those new methods show up in the tab automatically, with their
own parameter fields. There is no per-method code to maintain.

How: the extension only depends on sd-mecha's stable introspection core (the same
machinery sd-mecha uses for *every* method):

- `sd_mecha.extensions.merge_methods.get_all()` to list methods,
- `MergeMethod.get_param_names() / get_input_types() / get_input_merge_spaces() /
  get_default_args() / get_return_type()` to read each method's signature,
- `sd_mecha.model(path)` + `sd_mecha.merge(recipe, output=…)` to run it.

A new merge method is just a function registered against that core, so it is
introspected with zero changes here. Each method is introspected defensively — if
a future method doesn't fit the assumptions, it's skipped, not fatal.

Model slots and hyperparameters are told apart by **merge space**, not by type:
sd-mecha marks tweakable scalars (`alpha`, `beta`, …) with the `"param"` merge
space even when they're typed `Tensor`. That's the one non-obvious detail the
introspection relies on, and it's a deliberate, stable part of sd-mecha's API.

## Install

In Diffucore UI: **Settings → Extensions → Install**, paste:

```
https://github.com/nawka12/diffucore-mecha.git
```

Then install sd-mecha into Diffucore's Python environment. Per Diffucore's
"pip is opt-in" install policy, dependencies are **not** auto-installed:

- check **"Install Python dependencies"** in the install panel (only do this for
  sources you trust), **or**
- install it yourself and **Reload** the extension:

  ```
  pip install sd-mecha
  ```

The extension loads fine without sd-mecha — the Merge tab just shows install
instructions until it's present.

## Usage

1. Open the **Merge** tab.
2. Pick the **folder** (`checkpoints` or `diffusion-models`) — this drives both
   the input list and where the output lands.
3. Pick a **method** (use the filter box to search).
4. Fill the **model** slots and any **parameters** (defaults are prefilled).
5. Name the **output**, pick **device** (`cpu` is safest; `cuda` is faster but
   uses VRAM and may contend with a loaded model) and **dtype**.
6. **Merge.** Progress shows in the tab and the shared queue panel.

## Limitations

- **Flat, single-step merges.** Some methods expect a *delta*-space input (e.g.
  `add_difference`'s second model is the difference of two models, normally
  produced by `subtract`). This extension's one-shot UI can't compose those
  intermediate recipes, so delta-requiring methods may error. The common merges
  (`weighted_sum`, `slerp`, `n_average`, `geometric_sum`, …) work directly.
  Recipe chaining is out of scope by design — it's what keeps this extension
  small and maintenance-free.
- Merge correctness, supported architectures, and method semantics are all
  sd-mecha's; this extension is a thin, generic bridge.

## License

MIT — see [LICENSE](LICENSE). sd-mecha is a separate project under its own
license.
