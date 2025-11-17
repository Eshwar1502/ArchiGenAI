import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
from itertools import product
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import math
import numpy as np
import torch
import torch.nn.functional as F
from itertools import product

import torch, pickle, json, os

SAVE_PATH = r"C:\Users\dgaur03\OneDrive - dentsu\Desktop\Archi\architecture_predictor.pt"
ENC_PATH  = r"C:\Users\dgaur03\OneDrive - dentsu\Desktop\Archi\encoders.pkl"
META_PATH = r"C:\Users\dgaur03\OneDrive - dentsu\Desktop\Archi\meta.json"

# 1) load encoders
if not os.path.exists(ENC_PATH):
    raise FileNotFoundError(f"Encoders file not found: {ENC_PATH}. Save encoders in training and upload.")
with open(ENC_PATH, 'rb') as f:
    encoders = pickle.load(f)

# 2) load metadata if available to compute num_classes
if os.path.exists(META_PATH):
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    columns = meta['columns']
    num_classes = meta['num_classes']
else:
    # fallback: compute from encoders
    columns = ['cloud','sourcetype','mode','datalake','dataingestion','tools','workflow','involvesml','welldefined']
    num_classes = [len(encoders[c].classes_) for c in columns]

# recompute mask/unk ids exactly like training
mask_token_ids = [n for n in num_classes]
unk_token_ids  = [n+1 for n in num_classes]

# create emb_dims same as training
import math
emb_dims = [min(50, max(8, int(1.5 * math.sqrt(n)))) for n in num_classes]

# ===========================================
# STEP 4: Model Definition
# ===========================================

class MaskedAutoencoder(nn.Module):
    def __init__(self, num_classes_per_col, emb_dim_per_col):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(numc + 2, emb)   # +2 for MASK and UNK
            for numc, emb in zip(num_classes_per_col, emb_dim_per_col)
        ])
        total_emb = sum(emb_dim_per_col)
        self.encoder = nn.Sequential(
            nn.Linear(total_emb, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        self.heads = nn.ModuleList([
            nn.Linear(128, numc + 2)
            for numc in num_classes_per_col
        ])
    def forward(self, x):
        emb_list = []
        for i, emb in enumerate(self.embs):
            emb_list.append(emb(x[:, i]))
        h = torch.cat(emb_list, dim=1)
        z = self.encoder(h)
        logits = [head(z) for head in self.heads]
        return logits

# create embedding dimensions per column
emb_dims = [min(50, max(8, int(1.5 * math.sqrt(n)))) for n in num_classes]

model = MaskedAutoencoder(num_classes, emb_dims)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# 3) load state dict robustly with map_location and inspect
ckpt = torch.load(SAVE_PATH, map_location='cpu')
# if you accidentally saved a dict with a named key:

if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
elif isinstance(ckpt, dict) and any(k.startswith('embs') or k.startswith('encoder') for k in ckpt.keys()):
    # probably saved model.state_dict() directly
    state_dict = ckpt
else:
    state_dict = ckpt


# optional: compare shapes
# for k, v in state_dict.items():
#     print("saved:", k, v.shape)


# try loading

msg = model.load_state_dict(state_dict, strict=False)

# print("load result:", msg)   # missing/unexpected keys will show here

# move model to device afterwards
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_topk(model,
                 encoders,
                 partial_input_dict,
                 mask_token_ids,
                 unk_token_ids,
                 columns,
                 num_classes,
                 topk_per_field=5,
                 top_output=10):
    """
    model: trained model (already on device)
    encoders: dict of LabelEncoders per column
    partial_input_dict: mapping column->string or None
    mask_token_ids: list of per-column mask ids (integers)
    unk_token_ids: list of per-column unk ids (integers)
    columns: list of column names (same order as training)
    num_classes: list of int (# real categories per column)
    """
    device = next(model.parameters()).device
    model.eval()

    # build partial_row (list of ints)
    partial_row = []
    for i, col in enumerate(columns):
        val = partial_input_dict.get(col, None)
        if val is None:
            partial_row.append(int(mask_token_ids[i]))
        else:
            le = encoders[col]
            # treat string values safely (they might already be encoded int)
            try:
                # if user passed the original label string
                if isinstance(val, str):
                    if val in le.classes_:
                        encoded = int(le.transform([val])[0])
                    else:
                        encoded = int(unk_token_ids[i])
                else:
                    # user gave integer already
                    encoded = int(val)
            except Exception:
                encoded = int(unk_token_ids[i])
            partial_row.append(encoded)

    x = torch.tensor([partial_row], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(x)   # list of tensors per field, each shape [1, num_classes_i + 2]

    # convert logits -> probabilities on CPU numpy
    probs = [F.softmax(logit[0].cpu(), dim=0).numpy() for logit in logits]

    top_lists = []
    for i, p in enumerate(probs):
        # if field was provided (not masked), keep that single known value
        if partial_row[i] != int(mask_token_ids[i]):
            top_lists.append([(partial_row[i], 1.0)])
        else:
            p_filtered = p.copy()
            # zero out MASK and UNK indices so they are not selected
            p_filtered[int(mask_token_ids[i])] = 0.0
            p_filtered[int(unk_token_ids[i])]  = 0.0
            # fallback: if everything zero (rare), use original p but still zero mask
            if p_filtered.sum() == 0:
                p_filtered = p.copy()
                p_filtered[int(mask_token_ids[i])] = 0.0
            # get top-k indices
            idxs = p_filtered.argsort()[-topk_per_field:][::-1]
            top_lists.append([(int(idx), float(p[idx])) for idx in idxs])

    # enumerate combinations and rank by joint log-prob
    combos = []
    for comb in product(*top_lists):
        cats, ps = zip(*comb)
        joint_logprob = sum(math.log(max(1e-12, p)) for p in ps)
        combos.append((joint_logprob, cats))
    combos.sort(reverse=True)
    top = combos[:top_output]

    # decode indices back to labels, handling MASK/UNK gracefully
    decoded = []
    for logp, cats in top:
        row_dict = {}
        for i, col in enumerate(columns):
            idx = cats[i]
            if idx >= num_classes[i]:
                # idx is MASK or UNK (shouldn't happen because we filtered earlier, but safe-guard)
                if idx == int(mask_token_ids[i]):
                    row_dict[col] = "[MASK]"
                elif idx == int(unk_token_ids[i]):
                    row_dict[col] = "[UNK]"
                else:
                    row_dict[col] = "[SPECIAL]"
            else:
                # normal decode
                row_dict[col] = encoders[col].inverse_transform([int(idx)])[0]
        decoded.append({"solution": row_dict, "score": float(math.exp(logp))})
    return decoded

partial_example = {
    "cloud": "AWS",
    "sourcetype": "Structured",
    "mode": "Batch",
    "datalake": None,
    "dataingestion": None,
    "tools": "Databricks",
    "workflow": None,
    "involvesml": "Yes",
    "welldefined": "Yes", # Assuming "Unstructured Data Source" and "Source Type" are combined/re-mapped to 'sourcetype

}

solutions = predict_topk(
    model=model,
    encoders=encoders,
    partial_input_dict=partial_example,
    mask_token_ids=mask_token_ids,   # list you created earlier
    unk_token_ids=unk_token_ids,     # list you created earlier
    columns=columns,
    num_classes=num_classes,
    topk_per_field=5,
    top_output=10
)

for i, s in enumerate(solutions, 1):
    print(f"\nSolution {i}: score={s['score']}")
    for k,v in s['solution'].items():
        print(f"  {k}: {v}")

