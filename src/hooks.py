"""Activation caching and patching for the XLS-R+AASIST countermeasure.

Written against ``src/ssl_aasist.build_model``: the front-end is an HF
``Wav2Vec2Model`` at ``net.ssl_model.model``, so the hookable sites are

    net.ssl_model.model.feature_extractor.conv_layers[i]   (7 conv blocks)
    net.ssl_model.model.encoder.layers[i]                  (24 transformer layers, XLS-R 300M)

plus whatever AASIST graph modules ``repo.Model`` exposes (enumerate with
``named_modules()`` — kept generic here since that module tree is upstream
code loaded at runtime).

DESIGN. Two context managers, composable:

    with ActivationCache(net, sites) as cache:
        net(wave_clean)                  # cache filled per site
    with PatchActivations(net, {site: cache[site]}):
        out = net(wave_corrupted)        # corrupted run with clean acts at site

The patching unit is a *site output* (a transformer layer's hidden state, a
conv block's feature map). Head-level patching is NOT a separate hook target
in HF wav2vec2 — heads are folded into the attention out_proj input — see
``patch_heads`` below for the slicing approach.

VALIDITY TESTS (tests/test_hooks.py, must pass before any experiment):
    1. identity: patching a site with its own activations reproduces the
       unpatched logits bit-exactly (same dtype/device).
    2. locality: patching site L leaves activations strictly upstream of L
       unchanged.

COMPUTE NOTE. Patching at layer L only requires recomputing layers >= L.
The repo's cache-once philosophy extends: ``run_from_layer`` replays the
encoder from a cached hidden state, so a full layer sweep over the eval set
costs one full forward pass plus per-layer partial replays, not n_layers
full passes.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


def encoder_layer_sites(n_layers: int = 24) -> list[str]:
    """Canonical site names for the XLS-R transformer stack."""
    return [f"ssl_model.model.encoder.layers.{i}" for i in range(n_layers)]


def _resolve(net, dotted: str):
    mod = net
    for part in dotted.split("."):
        mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
    return mod


class ActivationCache:
    """Forward-hook cache keyed by dotted module path.

    Stores each site's output tensor (detached, CPU by default to keep GPU
    memory flat across the eval set — pass ``to_cpu=False`` for speed when
    patching immediately after).
    """

    def __init__(self, net, sites: list[str], *, to_cpu: bool = True):
        self.net, self.sites, self.to_cpu = net, sites, to_cpu
        self.store: dict[str, object] = {}
        self._handles: list = []

    def __enter__(self) -> "ActivationCache":
        for name in self.sites:
            mod = _resolve(self.net, name)

            def hook(_m, _inp, out, name=name):
                t = out[0] if isinstance(out, tuple) else out
                self.store[name] = t.detach().cpu() if self.to_cpu else t.detach()

            self._handles.append(mod.register_forward_hook(hook))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __getitem__(self, name: str):
        return self.store[name]


@contextmanager
def PatchActivations(net, patches: dict[str, object]) -> Iterator[None]:
    """Replace each site's output with the supplied tensor for the duration.

    The hook returns the patched tensor, which torch propagates downstream;
    upstream computation is untouched (locality test 2). Tuple-returning
    modules (HF encoder layers return ``(hidden, attn?)``) have only their
    hidden state replaced.
    """
    handles = []
    try:
        for name, tensor in patches.items():
            mod = _resolve(net, name)

            def hook(_m, _inp, out, tensor=tensor):
                t = tensor.to(out[0].device if isinstance(out, tuple) else out.device)
                if isinstance(out, tuple):
                    return (t, *out[1:])
                return t

            handles.append(mod.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def patch_heads(net, layer: int, head_idx: list[int], donor_hidden):
    """Head-granular patching — the fine pass. SKELETON / TODO.

    HF wav2vec2 attention computes all heads jointly and concatenates before
    ``out_proj``; there is no per-head module to hook. The defensible
    approach: forward-pre-hook on
    ``ssl_model.model.encoder.layers.{layer}.attention.out_proj`` — its input
    is (B, T, n_heads*head_dim) with heads in contiguous head_dim=64 slices
    for XLS-R 300M (16 heads x 64). Run the donor input through the *same*
    pre-hook capture to get donor slices, replace only ``head_idx`` slices,
    let out_proj proceed. This patches the heads' value-path contribution,
    which is the standard head-patching target (cf. attention-head patching
    in the LM interp literature); patching attention *patterns* instead is a
    different intervention — pick one and say which in the paper.
    """
    raise NotImplementedError("fine pass — implement after layer-level results")


def identity_check(net, wave, sites: list[str], *, atol: float = 0.0) -> bool:
    """Validity test 1: self-patching must reproduce the clean logits."""
    import torch

    with torch.no_grad():
        clean = net(wave)
        with ActivationCache(net, sites, to_cpu=False) as cache:
            net(wave)
        with PatchActivations(net, {s: cache[s] for s in sites}):
            patched = net(wave)
    return bool(torch.allclose(clean, patched, atol=atol))
