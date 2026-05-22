# arXiv submission metadata

Copy-paste source for every field the arXiv submission form will ask about.

---

## Step 1 — License

**Select:** `Creative Commons Attribution 4.0 International (CC-BY 4.0)`

(Reasoning: matches the repo's MIT spirit, allows downstream reuse with citation.)

---

## Step 2 — Categories

**Primary archive/subject:** `cs` (Computing Research Repository)

**Primary category:** `cs.CL` — Computation and Language

**Cross-list (secondary) categories:** `cs.LG` — Machine Learning

---

## Step 3 — Upload

Upload the single tarball:

```
/Users/x/src/localized-additive-composition/arxiv_submission.tar.gz
```

Contents (flat root + `figures/` subdir):

```
./paper.tex
./refs.bib
./paper.bbl
./figures/localized_kl_by_layer.pdf
```

arXiv will auto-detect `paper.tex` as the main file and use `paper.bbl` for the bibliography. After upload, click "Process" and wait for the proof PDF to compile.

---

## Step 4 — Metadata fields

### Title

```
As X, Do Y: How Persona and Task Combine in Instruction-Tuned LLMs
```

### Authors

```
Eric Xu
```

### Abstract (plain text, ~1440 chars; arXiv limit is 1920)

```
Role prompts of the form "As X, do Y" admit a clean linear decomposition at one specific site in the residual stream: the prompt-to-answer transition -- the last prompt token together with the first two generated tokens -- in an early/mid layer band. There, persona and task contribute through partially orthogonal additive directions. Forming a pure persona effect Delta_X, a pure task effect Delta_Y, and substituting h_BB + Delta_X + Delta_Y for the clean residual yields downstream output within a small KL of clean on Gemma-2-2B-IT and Qwen-2.5-{1.5B, 3B}-Instruct, across a 12-cell short grid and a 48-cell long-persona grid, with persona-specific behavioral markers preserved.

The natural inference from this additive structure is that the role prompt can be compressed into a single cached residual vector. We show it cannot. Injecting the cached additive prediction -- or even the oracle clean residual h_XY -- into a baseline host prompt with the persona text removed does not approach the clean long-persona target, at one site or at many layers. Persona-conditioned multi-token generation flows through attention back to the persona-text positions throughout the prompt, which no residual at one site reproduces.

Local additivity in the residual stream does not imply prompt compressibility. The additive structure at the prompt-to-answer transition supports interpretability and fine-grained steering of persona or task contributions; persona-conditioned behavior across the full continuation depends on a distributed prompt/KV mechanism that local activation arithmetic does not displace.
```

### Comments

```
12 pages, 1 figure, 4 tables. Code and cached experiment outputs: https://github.com/xuy/localized-additive-composition
```

### Report number

leave blank

### Journal reference

leave blank

### External DOI

leave blank

### ACM class

leave blank (or `I.2.7; I.2.6` if you want one)

### MSC class

leave blank

---

## Step 5 — Preview and submit

1. arXiv will render a proof PDF — verify it looks like your local `paper.pdf` (12 pages, title page with author block, figure on page 5, etc.).
2. Click **Submit**.
3. Moderation queue is typically 1–3 business days for a new author. Once accepted, you'll get an arXiv ID like `2606.XXXXX` and the paper goes live on the next announcement (00:00 ET, Sun–Thu).

---

## After acceptance — repo follow-ups

Once arXiv assigns an ID, update the README's BibTeX entry:

```bibtex
@misc{xu2026asxdoy,
  title         = {As X, Do Y: How Persona and Task Combine in Instruction-Tuned LLMs},
  author        = {Xu, Eric},
  year          = {2026},
  eprint        = {2606.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

And add an arXiv badge to the top of the README:

```markdown
[![arXiv](https://img.shields.io/badge/arXiv-2606.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2606.XXXXX)
```
