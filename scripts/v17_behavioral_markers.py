#!/usr/bin/env python3
"""V17 — ROME-style behavioral verification of additive substitution.

For each (long persona, task) cell, generate 80 tokens under four conditions
and score whether persona-specific target markers appear:

  clean   : "As [long persona], [task]" with no intervention
  additive: same prompt, substitute h_BB + Δ_X + Δ_Y at p_last, L=14
  remove_X: same prompt, substitute h_XY - Δ_X at p_last, L=14
  bare    : "[task]" with no persona prefix, no intervention

Predicted pattern:
  - clean:    high marker presence (ceiling)
  - additive: matches clean if the additive substitution is behaviorally faithful
  - remove_X: substantially below clean
  - bare:     substantially below clean

Headline metric: fraction of cells where each condition contains at least one
persona-target marker. Secondary: distinct-marker count.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "google/gemma-2-2b-it"
INJECT_L = 14
N_GEN = 80
CACHE = Path(__file__).parent.parent / "results"
CACHE.mkdir(exist_ok=True)

LONG_PERSONAS = {
    "engineer": "a senior software engineer with 10 years of experience who pays close "
                "attention to architecture, reliability, and avoiding single points of failure",
    "counselor": "an empathetic counselor with deep training in active listening, cognitive "
                 "behavioral therapy, and trauma-informed care, who helps clients feel heard "
                 "without imposing solutions",
    "founder":   "a pragmatic startup founder who has bootstrapped three companies, makes "
                 "capital-efficient decisions, iterates fast based on user feedback, and "
                 "avoids vanity metrics",
    "teacher":   "a middle school science teacher who has taught for 15 years, explains "
                 "concepts with relatable analogies, gently checks for understanding, and "
                 "meets students at their level",
    "journalist": "an investigative journalist who has covered city government for two decades, "
                  "asks pointed questions, follows the money, and verifies every claim against "
                  "primary sources",
    "doctor":    "a primary-care physician who has practiced for 25 years, listens carefully "
                 "to symptoms, considers differential diagnoses without alarming the patient, "
                 "and explains options clearly",
    "lawyer":    "a corporate litigator who has tried cases at the appellate level for 20 years, "
                 "anticipates opposing arguments, builds case theory from the record, and "
                 "communicates dense law in plain English",
    "chef":      "a head chef trained in classical French technique who has run three "
                 "Michelin-starred kitchens, builds menus around seasonal ingredients, "
                 "and teaches young cooks by demonstration",
}

# Persona-specific behavioral markers. A cell scores "persona-present" if any
# marker appears (case-insensitive, word-boundary-aware).
PERSONA_MARKERS = {
    "engineer": [
        "SPOF", "single point of failure", "single-point-of-failure",
        "scalability", "scalable", "reliability", "fault tolerance",
        "fault-tolerant", "redundancy", "resilience", "throughput",
        "latency", "consistency", "availability",
    ],
    "counselor": [
        "feel heard", "validate", "validated", "acknowledge",
        "acknowledged", "your experience", "your feelings",
        "compassion", "compassionate", "without judgment",
        "trauma-informed", "active listening", "feelings", "emotion",
    ],
    "founder": [
        "iterate", "iterating", "iteration", "MVP", "minimum viable",
        "user feedback", "customer feedback", "capital-efficient",
        "lean", "runway", "validation", "ship", "shipping",
        "product-market fit", "vanity metric", "traction", "burn",
        "bootstrapped",
    ],
    "teacher": [
        "imagine", "think of", "like a", "analogy", "analogies",
        "for example", "step by step", "step-by-step", "students",
        "understand", "let's say", "picture this", "as if",
    ],
    "journalist": [
        "sources", "source", "primary source", "primary sources",
        "follow the money", "accountability", "transparency",
        "investigate", "verify", "verified", "on the record",
        "off the record", "evidence", "public interest",
        "pointed question", "track record",
    ],
    "doctor": [
        "symptom", "symptoms", "diagnosis", "diagnose", "differential",
        "patient", "treatment", "ruling out", "rule out", "clinical",
        "examination", "condition", "medication", "evaluate",
        "underlying", "comorbid",
    ],
    "lawyer": [
        "liability", "liabilities", "jurisdiction", "statute",
        "precedent", "evidence", "evidentiary", "opposing",
        "counterparty", "due diligence", "parties", "indemnif",
        "contractual", "compliance", "compliant", "jurisprudence",
        "case theory", "on the record",
    ],
    "chef": [
        "seasonal", "season", "ingredient", "ingredients", "flavor",
        "flavour", "palate", "fresh", "simmer", "sauté", "saute",
        "balance", "garnish", "mise en place", "technique",
        "classical", "French",
    ],
}

BASELINE_PERSONA = "a thoughtful person"
BASELINE_TASK = "Give advice to someone facing a difficult decision."
TASKS = {
    "arch": "Review this design: a microservice architecture where eight services share a "
            "single PostgreSQL database for both transactional state and event log.",
    "plan": "Review this plan: a three-person team building a B2B SaaS product, planning to "
            "launch in three months, with no usage analytics in v1.",
    "proposal": "Review this proposal: an internal tool that automates calendar scheduling "
                "using an LLM, sending tentative meetings to all parties before confirmation.",
}


def apply_chat(tok, user_text):
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True
    )


def make_sub_hook(target_value, position):
    target_tensor = torch.from_numpy(target_value).to(DEVICE).float()
    def hook(module, args, kwargs):
        hs = args[0]
        if hs.shape[1] <= position:
            return None
        new_hs = hs.clone()
        new_hs[:, position, :] = target_tensor
        return (new_hs,) + args[1:], kwargs
    return hook


def get_residuals(model, tok, prompt):
    ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    p_last = ids.shape[1] - 1
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, return_dict=True)
    arr = np.stack([h[0, p_last].float().cpu().numpy() for h in out.hidden_states])
    return arr, p_last


def gen(model, tok, prompt, hook_fn=None, layer=None, n=N_GEN):
    ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    handle = None
    if hook_fn is not None:
        handle = model.model.layers[layer].register_forward_pre_hook(hook_fn, with_kwargs=True)
    try:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=n, do_sample=False, num_beams=1,
                                 pad_token_id=tok.eos_token_id)
    finally:
        if handle is not None:
            handle.remove()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def score_markers(text, markers):
    """Return (any_present: bool, distinct_count: int, hits: list[str])."""
    txt = text.lower()
    hits = []
    for m in markers:
        # Multi-word: use substring; single-word: use word boundary
        m_lower = m.lower()
        if " " in m_lower or "-" in m_lower:
            if m_lower in txt:
                hits.append(m)
        else:
            # word-boundary regex, allow internal hyphens
            if re.search(rf"\b{re.escape(m_lower)}\b", txt):
                hits.append(m)
    return (len(hits) > 0, len(hits), hits)


def main():
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32,
                                                  attn_implementation="eager").to(DEVICE)
    model.eval()
    print(f"  n_layers={model.config.num_hidden_layers}\n")

    # Capture all needed residuals
    print("Capturing residuals...")
    residuals = {}
    positions = {}
    prompts = {}
    def cap(key, prompt):
        r, pl = get_residuals(model, tok, prompt)
        residuals[key] = r
        positions[key] = pl
        prompts[key] = prompt

    cap(("BP", "BT"), apply_chat(tok, f"As {BASELINE_PERSONA}, {BASELINE_TASK}"))
    for tkey, ttext in TASKS.items():
        cap(("BP", tkey), apply_chat(tok, f"As {BASELINE_PERSONA}, {ttext}"))
    for pkey, pphrase in LONG_PERSONAS.items():
        cap((pkey, "BT"), apply_chat(tok, f"As {pphrase}, {BASELINE_TASK}"))
        for tkey, ttext in TASKS.items():
            cap((pkey, tkey), apply_chat(tok, f"As {pphrase}, {ttext}"))
    for tkey, ttext in TASKS.items():
        cap(("BARE", tkey), apply_chat(tok, ttext))
    print(f"  captured {len(residuals)} residuals\n")

    print("=" * 80)
    print(f"V17 — Behavioral marker recovery at L={INJECT_L}, 24 cells × 4 conditions")
    print("=" * 80)
    results = {}
    for pkey in LONG_PERSONAS:
        markers = PERSONA_MARKERS[pkey]
        for tkey in TASKS:
            cell = f"{pkey}__{tkey}"
            xy_prompt = prompts[(pkey, tkey)]
            xy_p_last = positions[(pkey, tkey)]
            bare_prompt = prompts[("BARE", tkey)]

            # Build interventions
            h_BB = residuals[("BP", "BT")][INJECT_L]
            dx = residuals[(pkey, "BT")][INJECT_L] - h_BB
            dy = residuals[("BP", tkey)][INJECT_L] - h_BB
            h_XY = residuals[(pkey, tkey)][INJECT_L]
            additive = h_BB + dx + dy
            removeX = h_XY - dx

            text_clean    = gen(model, tok, xy_prompt)
            text_additive = gen(model, tok, xy_prompt, make_sub_hook(additive, xy_p_last), INJECT_L)
            text_removeX  = gen(model, tok, xy_prompt, make_sub_hook(removeX,  xy_p_last), INJECT_L)
            text_bare     = gen(model, tok, bare_prompt)

            sc_clean    = score_markers(text_clean, markers)
            sc_additive = score_markers(text_additive, markers)
            sc_removeX  = score_markers(text_removeX, markers)
            sc_bare     = score_markers(text_bare, markers)

            results[cell] = {
                "clean":    {"any": sc_clean[0], "count": sc_clean[1], "hits": sc_clean[2], "text": text_clean},
                "additive": {"any": sc_additive[0], "count": sc_additive[1], "hits": sc_additive[2], "text": text_additive},
                "remove_X": {"any": sc_removeX[0], "count": sc_removeX[1], "hits": sc_removeX[2], "text": text_removeX},
                "bare":     {"any": sc_bare[0], "count": sc_bare[1], "hits": sc_bare[2], "text": text_bare},
            }
            print(f"  {cell:>22s}: clean={int(sc_clean[0])}({sc_clean[1]:>2d})  "
                  f"add={int(sc_additive[0])}({sc_additive[1]:>2d})  "
                  f"rmX={int(sc_removeX[0])}({sc_removeX[1]:>2d})  "
                  f"bare={int(sc_bare[0])}({sc_bare[1]:>2d})")

    # Summary
    print("\n" + "=" * 80)
    print("V17 — Summary across 24 cells")
    print("=" * 80)
    conds = ["clean", "additive", "remove_X", "bare"]
    print(f"  {'condition':>10s}  {'any-marker':>11s}  {'distinct-marker mean':>22s}")
    summary = {}
    for c in conds:
        any_count = sum(1 for r in results.values() if r[c]["any"])
        avg_distinct = float(np.mean([r[c]["count"] for r in results.values()]))
        summary[c] = {"any_present_rate": any_count / len(results),
                      "any_present_count": any_count,
                      "total_cells": len(results),
                      "distinct_mean": avg_distinct}
        print(f"  {c:>10s}  {any_count}/{len(results)} = {any_count/len(results)*100:>5.1f}%  {avg_distinct:>22.2f}")

    # Per-persona breakdown for any-marker
    print("\nPer-persona any-marker recovery (across 3 tasks):")
    print(f"  {'persona':>11s}  {'clean':>6s}  {'add':>6s}  {'rmX':>6s}  {'bare':>6s}")
    per_persona = {}
    for pkey in LONG_PERSONAS:
        row = {}
        for c in conds:
            v = sum(1 for tkey in TASKS if results[f"{pkey}__{tkey}"][c]["any"])
            row[c] = v
        per_persona[pkey] = row
        print(f"  {pkey:>11s}  {row['clean']}/3    {row['additive']}/3    {row['remove_X']}/3    {row['bare']}/3")

    out = {
        "model_name": MODEL_NAME,
        "inject_layer": INJECT_L,
        "n_gen_tokens": N_GEN,
        "personas": list(LONG_PERSONAS.keys()),
        "tasks": list(TASKS.keys()),
        "persona_markers": PERSONA_MARKERS,
        "results": results,
        "summary": summary,
        "per_persona": per_persona,
    }
    out_path = CACHE / "v17_behavioral_markers.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
