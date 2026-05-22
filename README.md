# As X, Do Y: How Persona and Task Combine in Instruction-Tuned LLMs

This repository contains the paper and reproducibility artifacts.

## What the paper claims

Role prompts of the form *"As X, do Y"* induce a residual-stream computation that can be decomposed into a pure persona effect Δ_X, a pure task effect Δ_Y, and an interaction term. The central finding is that this composition is **approximately additive in a localized region near answer formation** — spanning the last prompt position `p_last` and the first generated-token positions `g_1` and `g_2` — in an early/mid-depth layer band on Gemma-2-2B-IT, Qwen-2.5-1.5B-Instruct, and Qwen-2.5-3B-Instruct.

The paper also reports an explicit limit: substituting cached residual vectors into a baseline host prompt (without the persona text) does **not** reproduce clean long-persona behavior. The local additive structure does not license full prompt replacement; the wider persona-conditioning mechanism is distributed across prompt positions and the KV cache.

A behavioral-marker test complements the distributional KL evidence: the additive substitution recovers persona-specific output markers in 14 of 24 cells (58%), close to the clean ceiling of 16 of 24 (67%) and far above the persona-text-absent floor of 1 of 24 (4%).

## Repository layout

```
paper/                              Paper source and built PDF
├── paper.tex                       Main LaTeX manuscript
├── paper.pdf                       Built PDF (16 pages)
├── refs.bib                        Bibliography
└── figures/
    ├── localized_kl_by_layer.pdf   Figure 1 (per-position KL vs layer)
    └── localized_kl_by_layer.png

scripts/                            Five experiment scripts
├── v14b_v2_inject.py               Host-prompt single-site substitution (§7)
├── v14e_multilayer_inject.py       Multi-layer substitution (§7)
├── v15_localized_positions.py      Localized-position causal sweep (§4)
├── v16_diverse_grid.py             Broadened 8×6 long-persona grid (§5)
└── v17_behavioral_markers.py       Behavioral-marker recovery test (§6)

results/                            Cached experiment outputs the paper cites
├── v14b_v2.json
├── v14e_multilayer.json
├── v15_gemma2b_localized_positions.json
├── v15_qwen15_localized_positions.json
├── v15_qwen3b_localized_positions.json
├── v16_gemma2b_diverse_grid.json
├── v16_qwen15_diverse_grid.json
└── v17_behavioral_markers.json
```

## Reproducibility

### Environment

All experiments use a single Apple Silicon device with the PyTorch `mps` backend. Equivalent setups on CUDA should work with minor adjustments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Model dtypes

The choice of dtype is load-bearing for the smaller Qwen models — `float16` produces NaN residuals at the layers used in the paper. Use `bfloat16` instead.

| Model | dtype |
|---|---|
| `google/gemma-2-2b-it` | `float32` |
| `Qwen/Qwen2.5-1.5B-Instruct` | `bfloat16` |
| `Qwen/Qwen2.5-3B-Instruct` | `bfloat16` |

### Generation settings

All generation is fully greedy (`do_sample=False`, `num_beams=1`) so that reported KLs and behavioral scores are deterministic functions of the model, prompt, and intervention.

### Running the experiments

Each script writes its output to `results/` next to the repo root, with a `v##` filename prefix matching the paper's Artifact Map.

```bash
# §4: localized-position sweep across the 12-cell short grid
python scripts/v15_localized_positions.py --model gemma2b
python scripts/v15_localized_positions.py --model qwen15
python scripts/v15_localized_positions.py --model qwen3b

# §5: broadened 48-cell long-persona grid
python scripts/v16_diverse_grid.py --model gemma2b
python scripts/v16_diverse_grid.py --model qwen15

# §6: behavioral-marker verification
python scripts/v17_behavioral_markers.py

# §7: negative result on prompt-to-vector replacement
python scripts/v14b_v2_inject.py
python scripts/v14e_multilayer_inject.py
```

The included `results/*.json` files are the exact outputs the paper cites. Rerunning a script will overwrite the corresponding file with a fresh (greedy, deterministic) run.

### Building the paper

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

The pre-built PDF is checked in at `paper/paper.pdf`.

## License

This repository is released under the MIT License. See `LICENSE` for details.

## Citation

```bibtex
@misc{as_x_do_y_2026,
  title         = {As X, Do Y: How Persona and Task Combine in Instruction-Tuned LLMs},
  author        = {Anonymous},
  year          = {2026},
  howpublished  = {arXiv preprint}
}
```
