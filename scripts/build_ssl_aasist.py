"""Validate a fairseq-free SSL_Anti-spoofing (XLS-R + AASIST) baseline.

Loads the pretrained CM checkpoint (fairseq layout), remaps the wav2vec2 keys to
HuggingFace Wav2Vec2Model naming, assembles the AASIST head from the repo's own
model.py (with fairseq stubbed out + an HF SSL front-end), and checks EER on a
stratified sample. If EER is single-digit, the remap is correct end-to-end.
"""
import sys, types, re, warnings, importlib.util
import numpy as np, soundfile as sf, torch
warnings.filterwarnings("ignore")

REPO = "third_party/SSL_Anti-spoofing"
CKPT = "third_party/weights/Pre_trained_SSL_anti-spoofing_models/LA_model.pth"

# --- 1. stub fairseq so we can import their model.py (fairseq only used in SSLModel) ---
fake = types.ModuleType("fairseq")
fake.checkpoint_utils = types.SimpleNamespace(
    load_model_ensemble_and_task=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stubbed")))
sys.modules["fairseq"] = fake

spec = importlib.util.spec_from_file_location("ssl_aasist_model", f"{REPO}/model.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["ssl_aasist_model"] = mod
spec.loader.exec_module(mod)

# --- 2. HF-based SSL front-end with the same interface their Model expects ---
from transformers import Wav2Vec2Model, Wav2Vec2Config

class HFSSLModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        cfg = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.model = Wav2Vec2Model(cfg)
        self.out_dim = 1024
    def extract_feat(self, x):
        if x.ndim == 3:
            x = x[:, :, 0]
        return self.model(x).last_hidden_state

mod.SSLModel = HFSSLModel  # monkeypatch

# --- 3. fairseq -> HF wav2vec2 key remap ---
def remap_w2v(fs):  # fs: dict with fairseq names (prefix already stripped)
    out = {}
    for k, v in fs.items():
        if k.startswith(("quantizer.", "project_q.", "final_proj.")):
            continue  # pretraining-only, not in HF Wav2Vec2Model
        nk = k
        if k == "mask_emb": nk = "masked_spec_embed"
        elif k.startswith("post_extract_proj"): nk = k.replace("post_extract_proj", "feature_projection.projection")
        elif k.startswith("layer_norm."): nk = "feature_projection." + k
        elif re.match(r"feature_extractor\.conv_layers\.\d+\.0\.", k):
            nk = re.sub(r"(conv_layers\.\d+)\.0\.", r"\1.conv.", k)
        elif re.match(r"feature_extractor\.conv_layers\.\d+\.2\.1\.", k):
            nk = re.sub(r"(conv_layers\.\d+)\.2\.1\.", r"\1.layer_norm.", k)
        elif k == "encoder.pos_conv.0.bias": nk = "encoder.pos_conv_embed.conv.bias"
        elif k == "encoder.pos_conv.0.weight_g": nk = "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
        elif k == "encoder.pos_conv.0.weight_v": nk = "encoder.pos_conv_embed.conv.parametrizations.weight.original1"
        elif ".self_attn." in k: nk = k.replace(".self_attn.", ".attention.")
        elif ".self_attn_layer_norm." in k: nk = k.replace(".self_attn_layer_norm.", ".layer_norm.")
        elif ".fc1." in k: nk = k.replace(".fc1.", ".feed_forward.intermediate_dense.")
        elif ".fc2." in k: nk = k.replace(".fc2.", ".feed_forward.output_dense.")
        out[nk] = v
    return out

device = "cuda" if torch.cuda.is_available() else "cpu"
net = mod.Model(args=None, device=device).to(device).eval()

raw = torch.load(CKPT, map_location="cpu", weights_only=False)
head = {k: v for k, v in raw.items() if not k.startswith("ssl_model.model.")}
fs = {k[len("ssl_model.model."):]: v for k, v in raw.items() if k.startswith("ssl_model.model.")}
hf = {f"ssl_model.model.{k}": v for k, v in remap_w2v(fs).items()}
combined = {**head, **hf}

missing, unexpected = net.load_state_dict(combined, strict=False)
missing = [m for m in missing]
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
if missing[:8]: print("  missing sample:", missing[:8])
if unexpected[:8]: print("  unexpected sample:", list(unexpected)[:8])

# --- 4. EER check on stratified sample, both score conventions, raw vs normalized ---
META = "data/asvspoof2021_LA/keys/CM/trial_metadata.txt"
FLAC = "data/asvspoof2021_LA/flac"
rows = [ln.split() for ln in open(META) if ln.strip()]
recs = [(r[1], 1 if r[5] == "bonafide" else 0) for r in rows]
bona = [u for u, l in recs if l == 1]; spoof = [u for u, l in recs if l == 0]
rng = np.random.default_rng(3)
pick = [(u, 1) for u in rng.choice(bona, 100, replace=False)] + \
       [(u, 0) for u in rng.choice(spoof, 100, replace=False)]

def feats(utt, normalize):
    a, _ = sf.read(f"{FLAC}/{utt}.flac", dtype="float32")
    x = torch.from_numpy(a).float()
    x = torch.nn.functional.pad(x, (0, 64600 - len(x))) if len(x) < 64600 else x[:64600]
    if normalize:
        x = (x - x.mean()) / (x.std() + 1e-7)
    return x

def eer(labels, scores):
    thr = np.sort(np.unique(scores)); best = 1.0
    for t in thr:
        frr = np.mean(scores[labels == 1] <= t); far = np.mean(scores[labels == 0] > t)
        best = min(best, max(frr, far))
    return best

labels = np.array([l for _, l in pick])
for normalize in (False, True):
    outs = []
    with torch.inference_mode():
        for u, _ in pick:
            o = net(feats(u, normalize).unsqueeze(0).to(device)).float().cpu().numpy()[0]
            outs.append(o)
    outs = np.array(outs)
    for name, sc in [("out[:,1]", outs[:, 1]), ("out[:,0]-out[:,1]", outs[:, 0] - outs[:, 1])]:
        e = eer(labels, sc)
        print(f"normalize={normalize!s:5} score={name:18} EER={e*100:5.2f}%")
