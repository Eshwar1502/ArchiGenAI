from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from typing import Optional
from dotenv import load_dotenv
from google import genai
import json
from fastapi.middleware.cors import CORSMiddleware
import time
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

SAVE_PATH = r"C:\Users\epalla01\OneDrive - dentsu\Documents\Data Generative Solutions\Archi 1\Archi\architecture_predictor.pt"
ENC_PATH  = r"C:\Users\epalla01\OneDrive - dentsu\Documents\Data Generative Solutions\Archi 1\Archi\encoders.pkl"
META_PATH = r"C:\Users\epalla01\OneDrive - dentsu\Documents\Data Generative Solutions\Archi 1\Archi\meta.json"

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Or set your frontend origin(s) for better security
    allow_credentials=True,
    allow_methods=["*"],            # Allows all HTTP methods, inc. OPTIONS
    allow_headers=["*"],            # Allows all headers
)

# Initialize Gemini client - automatically uses GEMINI_API_KEY from .env
client = genai.Client()

# Static prompt in backend
STATIC_PROMPT = """You are a Senior technical cloud data architect. I will give you a detailed technical architecture or system description.

IMPORTANT:
- Do NOT use ASCII diagrams, box-and-arrow illustrations, or visuals of any kind.
- Instead, SUMMARIZE the pipeline/architecture in clearly structured, step-by-step sections—one for each main stage (data ingestion, discovery/catalog, ETL/processing, curated storage, analytics/reporting, security, monitoring, etc.).
- For each stage:
    - List the main cloud components involved
    - Explain what happens at that stage in clear, plain English
    - Describe how the data flows from one stage to the next
    - Use bold for stage names, bullets and numbering for clarity, and avoid dense/unbroken text blocks
    - For each stage, add brief explanations of service roles
    - DO NOT include diagrams or visuals; present everything in written, step-by-step, ordered list or section format.
If helpful, provide an example end-to-end workflow.

NOTE: Dont include anything, just directly start with answer, no greeting, nothing. Straight away start with answers
"""

class InputData(BaseModel):
    country: Optional[str] = Field(...)
    cloud: Optional[str] = Field(...)
    well_defined_use_case: Optional[str] = Field(...)
    involves_ml: Optional[str] = Field(...)
    source_type: Optional[str] = Field(...)
    mode: Optional[str] = Field(...)
    confirmed_tools: Optional[str] = Field(...)
    workflowOrchestration: Optional[str] = None
    dataLakeWarehouse: Optional[str] = None
    dataIngestion: Optional[str] = None

# Add this model near your other models
class SelectedArchitecture(BaseModel):
    rank: int
    cloud: str
    source_type: str
    mode: str
    data_ingestion: str
    workflow_orchestration: str
    data_transformation: str
    datalake_warehouse: str
    score: float
    involves_ml: str
    well_defined: str

@app.post("/intake")
async def intake(data: InputData):
    print("Received data:", data)
    data_dict = data.dict()
    for key in ['country', 'cloud', 'well_defined_use_case', 'involves_ml', 'source_type', 'mode', 'confirmed_tools']:
        value = data_dict.get(key)
        if value == "None":
            data_dict[key] = None
    
    app.state.received_data = data_dict
    print("Normalized data:", data_dict)
    # -- NEW CODE STARTS HERE --
    # Map FastAPI InputData to your model's expected input columns


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
    print('--CHECKPOINT--')
    print(type(data_dict["cloud"]))
    # print(data.cloud)
    print('--CHECKPOINT--')


    # partial_example = {
    #     "cloud": data.cloud,
    #     "sourcetype": data.source_type,   # If you use source_type for sourcetype
    #     "mode": data.mode,
    #     "datalake": None,                 # Or data.dataLakeWarehouse if present
    #     "dataingestion": None,            # Or data.dataIngestion if present
    #     "tools": data.confirmed_tools,
    #     "workflow": None,                 # Or data.workflowOrchestration if present
    #     "involvesml": data.involves_ml,
    #     "welldefined": data.well_defined_use_case,
    # } 
    partial_example = {
    "cloud": data_dict["cloud"],
    "sourcetype": data_dict["source_type"],   # If you use "source_type" for sourcetype
    "mode": data_dict["mode"],
    "datalake": None,                 # You would need to check if data_dict has a "dataLakeWarehouse" key if you want to use it
    "dataingestion": None,            # You would need to check if data_dict has a "dataIngestion" key if you want to use it
    "tools": data_dict["confirmed_tools"],
    "workflow": None,                 # You would need to check if data_dict has a "workflowOrchestration" key if you want to use it
    "involvesml": data_dict["involves_ml"],
    "welldefined": data_dict["well_defined_use_case"],
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

    # Prepare a matrix (list of lists) format for frontend display
    matrix_data = []  # Each inner list corresponds to a matrix row
    for idx, s in enumerate(solutions, 1):
        row = [
            idx,  # Rank/serial
            s['score'],
            s['solution']['cloud'],
            s['solution']['sourcetype'],
            s['solution']['mode'],
            s['solution']['datalake'],
            s['solution']['dataingestion'],
            s['solution']['tools'],
            s['solution']['workflow'],
            s['solution']['involvesml'],
            s['solution']['welldefined'],
        ]
        matrix_data.append(row)

    # Optionally, send solution details as well
    frontend_solutions = []
    for idx, s in enumerate(solutions, 1):
        frontend_solutions.append({
            "rank": idx,
            "score": s["score"],
            "fields": s["solution"]
        })
        

    return {
        "status": "Data received",
        "data": data,
        "matrix_data": matrix_data,
        "solutions": frontend_solutions  # frontend picks by rank/id
    }


# --- RETRY LOGIC ---
MAX_ATTEMPTS = 5
INITIAL_BACKOFF = 2  # seconds

# @app.post("/gemini-conversation")
# async def gemini_conversation():
#     try:
#         if not hasattr(app.state, 'received_data') or app.state.received_data is None:
#             raise HTTPException(
#                 status_code=400,
#                 detail="No data available. Please call /intake endpoint first to provide data."
#             )
#         data = app.state.received_data
#         print("Data sent to Gemini:", json.dumps(data, indent=2))
#         full_prompt = f"{STATIC_PROMPT}\n\nData Pipeline Requirements:\n{json.dumps(data, indent=2)}"

#         attempt = 0
#         while True:
#             try:
#                 response = client.models.generate_content(
#                     model="gemini-2.5-flash",
#                     contents=full_prompt
#                 )
#                 print("Gemini response received")
#                 return {
#                     "status": "success",
#                     "response": response.text,
#                     "model": "gemini-2.5-flash",
#                     "used_data": data
#                 }
#             except Exception as e:
#                 attempt += 1
#                 err_msg = str(e)
#                 is_503 = '503' in err_msg or 'UNAVAILABLE' in err_msg.upper()
#                 if attempt < MAX_ATTEMPTS and is_503:
#                     wait = INITIAL_BACKOFF * (2 ** (attempt - 1))
#                     print(f"[Retry] Gemini API 503 error encountered (attempt {attempt}/{MAX_ATTEMPTS}), retrying in {wait} seconds...")
#                     time.sleep(wait)
#                     continue
#                 print(f"Error on attempt {attempt}: {err_msg}")
#                 raise HTTPException(status_code=503 if is_503 else 500, detail=err_msg)
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/gemini-conversation")
async def gemini_conversation(selected_arch: SelectedArchitecture):
    """
    Generate detailed architecture explanation based on user's selected configuration.
    Receives the selected row data from frontend.
    """
    try:
        # Convert the selected architecture to a dictionary
        data = selected_arch.dict()
        
        print("Selected architecture sent to Gemini:", json.dumps(data, indent=2))
        
        # Create a detailed prompt with the selected architecture
        full_prompt = f"""{STATIC_PROMPT}

Selected Data Architecture Configuration:
===========================================

Cloud Platform: {data['cloud']}
Source Type: {data['source_type']}
Data Processing Mode: {data['mode']}
Data Ingestion Method: {data['data_ingestion']}
Workflow Orchestration: {data['workflow_orchestration']}
Data Transformation Tool: {data['data_transformation']}
Data Lake/Warehouse: {data['datalake_warehouse']}
Involves Machine Learning: {data['involves_ml']}
Well-Defined Use Case: {data['well_defined']}
Architecture Confidence Score: {data['score']:.2f}
Recommended Rank: #{data['rank']}

Please provide a comprehensive, step-by-step architecture explanation for this specific configuration. Include:
1. Data ingestion layer details
2. Storage architecture
3. Processing and transformation workflows
4. Analytics and consumption layer
5. Security and governance considerations
6. Monitoring and observability setup"""

        attempt = 0
        while True:
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt
                )
                print("Gemini response received successfully")
                return {
                    "status": "success",
                    "response": response.text.strip(), 
                    "model": "gemini-2.5-flash",
                    "selected_architecture": data
                }
            except Exception as e:
                attempt += 1
                err_msg = str(e)
                is_503 = '503' in err_msg or 'UNAVAILABLE' in err_msg.upper()
                
                if attempt < MAX_ATTEMPTS and is_503:
                    wait = INITIAL_BACKOFF * (2 ** (attempt - 1))
                    print(f"[Retry] Gemini API 503 error encountered (attempt {attempt}/{MAX_ATTEMPTS}), retrying in {wait} seconds...")
                    time.sleep(wait)
                    continue
                
                print(f"Error on attempt {attempt}: {err_msg}")
                raise HTTPException(
                    status_code=503 if is_503 else 500, 
                    detail=f"Gemini API error: {err_msg}"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in gemini_conversation: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )



