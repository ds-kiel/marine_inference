"""
Inference memory and latency footprint for the Bocknis Eck 2 SageFormer model.
Run from marine-inference/:
    python measure_footprint.py
"""

import os, sys, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import yaml

from src.preprocessing import load_scaler_stats
from sageformer.model import Model

# ── config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = "config/model_config.yaml"
CKPT_PATH   = "checkpoints/bocknis_eck_3day.pth"
N_WARMUP    = 3
N_RUNS      = 20

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
m = cfg["model"]

SEQ_LEN   = m["seq_len"]      # 1008
LABEL_LEN = m["label_len"]    # 504
PRED_LEN  = m["pred_len"]     # 432
N_FEAT    = m["enc_in"]       # 7
D_MODEL   = m["d_model"]

import types
args = types.SimpleNamespace(**{k: v for k, v in m.items()})

CKPT_MB = os.path.getsize(CKPT_PATH) / 1024**2

# ── helpers ───────────────────────────────────────────────────────────────────
def fmt(n): return f"{n:,}"
def mb(b):  return f"{b / 1024**2:.1f} MB"

def build_dummy(device, bs=1):
    x_enc      = torch.zeros(bs, SEQ_LEN,             N_FEAT,  device=device)
    x_mark_enc = torch.zeros(bs, SEQ_LEN,             4,       device=device)
    x_dec      = torch.zeros(bs, LABEL_LEN + PRED_LEN, N_FEAT, device=device)
    x_mark_dec = torch.zeros(bs, LABEL_LEN + PRED_LEN, 4,      device=device)
    return x_enc, x_mark_enc, x_dec, x_mark_dec

def param_counts(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ── 1. load model (CPU first, to measure CPU RAM) ────────────────────────────
print("Loading model on CPU …")
model_cpu = Model(args).float()
state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model_cpu.load_state_dict(state)
model_cpu.eval()

total_params, trainable_params = param_counts(model_cpu)

# ── 2. GPU path ───────────────────────────────────────────────────────────────
gpu_available = torch.cuda.is_available()
gpu_model_mb = gpu_peak_alloc_mb = gpu_peak_reserved_mb = None
gpu_name = None

if gpu_available:
    gpu_name = torch.cuda.get_device_name(0)
    device   = torch.device("cuda:0")

    # -- model-only VRAM
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    model_gpu = Model(args).float().to(device)
    model_gpu.load_state_dict(state)
    model_gpu.eval()

    gpu_model_mb = torch.cuda.memory_allocated(device) / 1024**2

    # -- peak inference VRAM (single forward)
    torch.cuda.reset_peak_memory_stats(device)
    x_enc, x_mark_enc, x_dec, x_mark_dec = build_dummy(device)

    # warmup
    for _ in range(N_WARMUP):
        torch.manual_seed(0)
        with torch.no_grad():
            _ = model_gpu(x_enc, x_mark_enc, x_dec, x_mark_dec)
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model_gpu(x_enc, x_mark_enc, x_dec, x_mark_dec)
    torch.cuda.synchronize(device)

    gpu_peak_alloc_mb    = torch.cuda.max_memory_allocated(device)    / 1024**2
    gpu_peak_reserved_mb = torch.cuda.max_memory_reserved(device)     / 1024**2

    del model_gpu, x_enc, x_mark_enc, x_dec, x_mark_dec, out
    torch.cuda.empty_cache()

# ── 3. CPU latency + RSS ──────────────────────────────────────────────────────
try:
    import psutil
    proc = psutil.Process(os.getpid())
    def rss_mb(): return proc.memory_info().rss / 1024**2
except ImportError:
    import resource
    def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

baseline_rss = rss_mb()

x_enc_cpu, x_mark_enc_cpu, x_dec_cpu, x_mark_dec_cpu = build_dummy("cpu")

# warmup
for _ in range(N_WARMUP):
    torch.manual_seed(0)
    with torch.no_grad():
        _ = model_cpu(x_enc_cpu, x_mark_enc_cpu, x_dec_cpu, x_mark_dec_cpu)

timings = []
for _ in range(N_RUNS):
    torch.manual_seed(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model_cpu(x_enc_cpu, x_mark_enc_cpu, x_dec_cpu, x_mark_dec_cpu)
    timings.append(time.perf_counter() - t0)

peak_rss = rss_mb()
cpu_mean_ms = np.mean(timings) * 1000
cpu_std_ms  = np.std(timings)  * 1000
cpu_ram_mb  = peak_rss - baseline_rss

# ── 4. summary ────────────────────────────────────────────────────────────────
SEP = "─" * 56
print(f"\n{SEP}")
print("  SageFormer Bocknis Eck 2 — Inference Footprint")
print(f"  Checkpoint : {CKPT_PATH}")
print(f"  Config     : seq={SEQ_LEN}, pred={PRED_LEN}, enc_in={N_FEAT}, d={D_MODEL}")
print(SEP)
print(f"  Parameters (total)     : {fmt(total_params)}")
print(f"  Parameters (trainable) : {fmt(trainable_params)}")
print(f"  Checkpoint file        : {CKPT_MB:.1f} MB")
print(SEP)
if gpu_available:
    print(f"  GPU                    : {gpu_name}")
    print(f"  Model-only VRAM        : {gpu_model_mb:.1f} MB")
    print(f"  Peak allocated (fwd)   : {gpu_peak_alloc_mb:.1f} MB")
    print(f"  Peak reserved  (fwd)   : {gpu_peak_reserved_mb:.1f} MB")
else:
    print("  GPU                    : not available")
print(SEP)
print(f"  CPU latency ({N_RUNS} runs)  : {cpu_mean_ms:.1f} ± {cpu_std_ms:.1f} ms")
print(f"  CPU peak RSS delta     : {cpu_ram_mb:.1f} MB  (process baseline: {baseline_rss:.0f} MB)")
print(SEP)
print()
print("Provisioning guidance:")
print(f"  GPU (inference only)  : {gpu_peak_reserved_mb:.0f} MB reserved  "
      f"({gpu_peak_alloc_mb:.0f} MB allocated)" if gpu_available else "  GPU: N/A")
print(f"  CPU RAM               : ~{cpu_ram_mb + baseline_rss:.0f} MB total process RSS at inference")
print(f"  Latency (CPU, bs=1)   : ~{cpu_mean_ms:.0f} ms per 3-day forecast")
print()
