"""Fairseq-free loader for the SSL_Anti-spoofing baseline (wav2vec2 XLS-R + AASIST).

The upstream repo (TakHemlata/SSL_Anti-spoofing, Interspeech 2022) builds its
XLS-R front-end through ``fairseq``, which targets torch 1.8 / py3.7 and does not
build on modern Python. We avoid fairseq entirely:

  * the AASIST back-end is imported verbatim from the repo's ``model.py``
    (``fairseq`` is stubbed only so the module import succeeds);
  * the ``SSLModel`` front-end is swapped for a HuggingFace ``Wav2Vec2Model``
    built from the ``facebook/wav2vec2-xls-r-300m`` config (same architecture);
  * the pretrained checkpoint's wav2vec2 weights (fairseq key layout) are remapped
    onto HF naming and loaded. The remap is exact: ``load_state_dict`` reports
    0 missing / 0 unexpected, and the model reproduces a believable LA EER, which
    validates it end to end.

Output: 2 logits where **index 1 == bona fide, index 0 == spoof** (note this is
the opposite of AASIST3). Best with raw waveform (pad/crop to 64600), no
per-utterance normalisation.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SSL_REPO_DIR = _ROOT / "third_party" / "SSL_Anti-spoofing"
DEFAULT_CKPT = _ROOT / "third_party" / "weights" / "Pre_trained_SSL_anti-spoofing_models" / "LA_model.pth"
XLSR_HF_ID = "facebook/wav2vec2-xls-r-300m"


def _remap_w2v(fs: dict) -> dict:
    """fairseq wav2vec2 state-dict keys -> HuggingFace Wav2Vec2Model keys.

    ``fs`` has the ``ssl_model.model.`` prefix already stripped. Pretraining-only
    tensors (quantizer / project_q / final_proj) are dropped; they are absent from
    HF's ``Wav2Vec2Model``.
    """
    out = {}
    for k, v in fs.items():
        if k.startswith(("quantizer.", "project_q.", "final_proj.")):
            continue
        if k == "mask_emb":
            nk = "masked_spec_embed"
        elif k.startswith("post_extract_proj"):
            nk = k.replace("post_extract_proj", "feature_projection.projection")
        elif k.startswith("layer_norm."):
            nk = "feature_projection." + k
        elif re.match(r"feature_extractor\.conv_layers\.\d+\.0\.", k):
            nk = re.sub(r"(conv_layers\.\d+)\.0\.", r"\1.conv.", k)
        elif re.match(r"feature_extractor\.conv_layers\.\d+\.2\.1\.", k):
            nk = re.sub(r"(conv_layers\.\d+)\.2\.1\.", r"\1.layer_norm.", k)
        elif k == "encoder.pos_conv.0.bias":
            nk = "encoder.pos_conv_embed.conv.bias"
        elif k == "encoder.pos_conv.0.weight_g":
            nk = "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
        elif k == "encoder.pos_conv.0.weight_v":
            nk = "encoder.pos_conv_embed.conv.parametrizations.weight.original1"
        elif ".self_attn." in k:
            nk = k.replace(".self_attn.", ".attention.")
        elif ".self_attn_layer_norm." in k:
            nk = k.replace(".self_attn_layer_norm.", ".layer_norm.")
        elif ".fc1." in k:
            nk = k.replace(".fc1.", ".feed_forward.intermediate_dense.")
        elif ".fc2." in k:
            nk = k.replace(".fc2.", ".feed_forward.output_dense.")
        else:
            nk = k  # encoder.layer_norm.* and *.final_layer_norm.* are identical
        out[nk] = v
    return out


def _import_repo_model():
    """Import the upstream model.py with fairseq stubbed (only SSLModel uses it)."""
    if "fairseq" not in sys.modules:
        fake = types.ModuleType("fairseq")
        fake.checkpoint_utils = types.SimpleNamespace(
            load_model_ensemble_and_task=lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("fairseq stubbed; SSLModel is replaced by an HF front-end")))
        sys.modules["fairseq"] = fake
    mpath = SSL_REPO_DIR / "model.py"
    if not mpath.exists():
        raise FileNotFoundError(
            f"SSL_Anti-spoofing repo not found at {SSL_REPO_DIR}. Clone it: "
            "git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git "
            f"{SSL_REPO_DIR}")
    spec = importlib.util.spec_from_file_location("ssl_aasist_repo_model", mpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ssl_aasist_repo_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(ckpt_path: str | Path = DEFAULT_CKPT, device: str = "cpu"):
    """Assemble the XLS-R+AASIST countermeasure and load the remapped checkpoint.

    Returns an ``nn.Module`` in eval mode whose forward takes ``(B, 64600)`` raw
    waveform and returns ``(B, 2)`` logits (index 1 = bona fide).
    """
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    repo = _import_repo_model()

    class _HFSSLModel(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.model = Wav2Vec2Model(Wav2Vec2Config.from_pretrained(XLSR_HF_ID))
            self.out_dim = 1024

        def extract_feat(self, x):
            if x.ndim == 3:
                x = x[:, :, 0]
            return self.model(x).last_hidden_state

    repo.SSLModel = _HFSSLModel  # swap fairseq front-end for the HF one

    net = repo.Model(args=None, device=device)

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained CM checkpoint not found at {ckpt_path}. Download from the "
            "SSL_Anti-spoofing Google Drive (Pre_trained_SSL_anti-spoofing_models).")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = {k: v for k, v in raw.items() if not k.startswith("ssl_model.model.")}
    fs = {k[len("ssl_model.model."):]: v for k, v in raw.items() if k.startswith("ssl_model.model.")}
    state = {**head, **{f"ssl_model.model.{k}": v for k, v in _remap_w2v(fs).items()}}

    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected checkpoint mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected keys. The fairseq->HF remap may be stale.")
    return net.to(device).eval()
