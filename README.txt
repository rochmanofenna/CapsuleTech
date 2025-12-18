Flash Drive Layout – Temporal Accumulators
==========================================

1. paper/
   - temporal_accumulators.pdf        ← read this first (Nov 4/6/11 framing)
   - appendix_metrics.pdf             ← kernel details, timing tables

2. docs/
   - README.txt                       ← quick guide (this summary is mirrored there)
   - colab_instructions.txt           ← step-by-step for running the GPU demo in Colab

3. code/
   - BEF_main/                        ← full repo (BICEP, ENN-C++, FusionAlpha)
   - gpu_accumulator/                 ← cleaned CUDA/PyTorch module + Colab zip + HTML export

Supporting materials:
---------------------
• `code/gpu_accumulator/batch_streaming_accum.html` – readable version of the notebook
• `code/gpu_accumulator/gpu_accumulator.zip` – upload into Colab, see instructions
• `code/gpu_accumulator/benchmarks.csv` – raw CPU/GPU metrics (GTX 1650 + A100)

Paper structure (temporal_accumulators.pdf):
--------------------------------------------
1. Motivation – revisiting Nov 4/6/11: hash-first, hash-only signatures, Merkle tries at scale
2. Hybrid accumulator – Merkle roots + algebraic sketch, security intuition
3. Complexity comparison – O(log N) Merkle proofs vs O(1) sketch; GPU vs CPU table
4. Classification / open question – vector commitment vs streaming PCS
5. Appendix – CUDA kernel pseudocode, tiling strategy, tables (GTX1650, A100)

Colab usage (docs/colab_instructions.txt):
------------------------------------------
• Upload `gpu_accumulator.zip`
• !unzip -o gpu_accumulator.zip
• !pip install -q ninja
• from gpu_accumulator import demo_cuda; demo_cuda(...)

