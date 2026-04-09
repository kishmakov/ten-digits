FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 || true

# Copy source code
COPY src/ /app/src/
COPY evaluate_checkpoints.py /app/evaluate_checkpoints.py

# Create output directories
RUN mkdir -p /app/results/runs /app/results/data /app/checkpoints

# Default: train the 456-parameter model (seed 43), matching the reference paper exactly.
# Reference: https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition
#   pos_rank=3, qkv_rank=3, attn_out_rank=2, ffn_rank=3
#   tie_qkv=shareA_tieKV, use_rmsnorm=True
#   Total: 98 (token_emb) + 120 (pos_emb) + 7 (ln1) + 63 (qkv)
#          + 28 (attn_out) + 7 (ln2) + 126 (mlp) + 7 (ln_f) = 456 params
CMD ["python", "-m", "src.train", \
     "--run-name", "456p_s43", \
     "--pos-rank", "3", \
     "--qkv-rank", "3", \
     "--attn-out-rank", "2", \
     "--ffn-rank", "3", \
     "--tie-qkv", "shareA_tieKV", \
     "--use-rmsnorm", \
     "--seed", "43", \
     "--train-steps", "54000", \
     "--warmup-steps", "1350", \
     "--device", "cuda", \
     "--dtype", "bf16", \
     "--run-dir", "/app/results/runs/456p_s43"]
