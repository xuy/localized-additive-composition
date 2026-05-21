#!/usr/bin/env python3
"""V16 — Broader X/Y coverage for localized additive composition.

Purpose: stress-test the localized additive regime on a broader and more
realistic prompt grid after V15 established where the regime lives.

Grid:
  - 8 long personas
  - 6 tasks spanning review / planning / policy / recommendation / constrained creative

Method:
  - Probe positions: p_last, g1, g2
  - Probe layers: a fixed early/mid/late set in the localized regime
  - Measure directional additivity and causal additive substitution KL

This is a robustness experiment, not a new search over positions.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_PREFIX = 2
N_EVAL = 10
CACHE = Path(__file__).parent.parent / "results"
CACHE.mkdir(exist_ok=True)

MODELS = {
    "gemma2b": ("google/gemma-2-2b-it", torch.float32),
    "qwen15": ("Qwen/Qwen2.5-1.5B-Instruct", torch.bfloat16),
}

LONG_PERSONAS = {
    "engineer": "a senior software engineer with 10 years of experience who pays close "
                "attention to architecture, reliability, and avoiding single points of failure",
    "counselor": "an empathetic counselor with deep training in active listening, cognitive "
                 "behavioral therapy, and trauma-informed care, who helps clients feel heard "
                 "without imposing solutions",
    "founder": "a pragmatic startup founder who has bootstrapped three companies, makes "
               "capital-efficient decisions, iterates fast based on user feedback, and "
               "avoids vanity metrics",
    "teacher": "a middle school science teacher who has taught for 15 years, explains "
               "concepts with relatable analogies, gently checks for understanding, and "
               "meets students at their level",
    "journalist": "an investigative journalist who has covered city government for two decades, "
                  "asks pointed questions, follows the money, and verifies every claim against "
                  "primary sources",
    "doctor": "a primary-care physician who has practiced for 25 years, listens carefully "
              "to symptoms, considers differential diagnoses without alarming the patient, "
              "and explains options clearly",
    "lawyer": "a corporate litigator who has tried cases at the appellate level for 20 years, "
              "anticipates opposing arguments, builds case theory from the record, and "
              "communicates dense law in plain English",
    "chef": "a head chef trained in classical French technique who has run three Michelin-starred "
            "kitchens, builds menus around seasonal ingredients, and teaches young cooks by demonstration",
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
    "policy": "Comment on whether universal basic income is a good policy.",
    "haiku": "Write a haiku about Monday mornings.",
    "recommend": "Recommend a book worth reading and explain why.",
}
PROBES = ["p_last", "g1", "g2"]
LAYERS_BY_MODEL = {
    "gemma2b": [10, 14, 18],
    "qwen15": [7, 15, 20],
}


def apply_chat(tok, user_text, add_generation_prompt=True):
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def find_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
    raise AttributeError(f"Cannot find decoder layers in {type(model)}")


def get_model_sizes(model):
    cfg = model.config
    if not hasattr(cfg, "hidden_size") and hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    return cfg.hidden_size, cfg.vocab_size


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


def make_prompt(prompt_persona, task_text):
    return f"As {prompt_persona}, {task_text}"


def make_prompt_structures(tok):
    prompts = {}
    personas = {"baseline": BASELINE_PERSONA, **LONG_PERSONAS}
    tasks = {"baseline": BASELINE_TASK, **TASKS}
    for pkey, ptext in personas.items():
        for tkey, ttext in tasks.items():
            prompt_no_gen = apply_chat(tok, make_prompt(ptext, ttext), add_generation_prompt=False)
            prompt_full = apply_chat(tok, make_prompt(ptext, ttext), add_generation_prompt=True)
            ids_no_gen = encode(tok, prompt_no_gen)
            ids_full = encode(tok, prompt_full)
            prompts[(pkey, tkey)] = {
                "prompt_full": prompt_full,
                "ids_full": ids_full,
                "p_last": len(ids_full) - 1,
                "p_user_last": len(ids_no_gen) - 1,
            }
    return prompts


def capture_hidden_at_position(model, input_ids, position):
    ids = torch.tensor([input_ids], device=DEVICE)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, return_dict=True)
    return np.stack([h[0, position].float().cpu().numpy() for h in out.hidden_states])


def greedy_ref_tokens(model, input_ids, n_total, pad_token_id):
    ids = torch.tensor([input_ids], device=DEVICE)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=n_total,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_token_id,
        )
    return out[0, ids.shape[1]:].cpu().tolist()


def make_sub_hook(target_value, position, dtype):
    target_tensor = torch.from_numpy(target_value).to(DEVICE).to(dtype)

    def hook(module, args, kwargs):
        hs = args[0]
        if hs.shape[1] <= position:
            return None
        new_hs = hs.clone()
        new_hs[:, position, :] = target_tensor
        return (new_hs,) + args[1:], kwargs

    return hook


def get_logp_window(model, input_ids, ref_tokens, vocab_size, hook=None, layer_module=None):
    handles = []
    if hook is not None and layer_module is not None:
        handles.append(layer_module.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        ids = torch.tensor([input_ids], device=DEVICE)
        ref_t = torch.tensor([ref_tokens], device=DEVICE)
        full = torch.cat([ids, ref_t], dim=1)
        prompt_len = ids.shape[1]
        with torch.no_grad():
            out = model(full, return_dict=True)
        logp = torch.zeros(len(ref_tokens), vocab_size, dtype=torch.float32)
        for t in range(len(ref_tokens)):
            logp[t] = torch.log_softmax(out.logits[0, prompt_len - 1 + t].float(), dim=-1).cpu()
        return logp
    finally:
        for handle in handles:
            handle.remove()


def kl_sum(p, q):
    total = 0.0
    for t in range(p.shape[0]):
        ex = p[t].exp()
        total += float((ex * (p[t] - q[t])).sum().item())
    return total


def median_summary(values):
    return {
        "median": float(np.median(values)),
        "iqr": [float(np.percentile(values, 25)), float(np.percentile(values, 75))],
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_probe_sequence(prompt_info, probe, clean_xy_ref):
    ids_full = prompt_info["ids_full"]
    if probe == "p_last":
        return ids_full, prompt_info["p_last"], clean_xy_ref[:N_EVAL]
    if probe == "g1":
        seq = ids_full + clean_xy_ref[:1]
        return seq, len(seq) - 1, clean_xy_ref[1 : 1 + N_EVAL]
    if probe == "g2":
        seq = ids_full + clean_xy_ref[:2]
        return seq, len(seq) - 1, clean_xy_ref[2 : 2 + N_EVAL]
    raise ValueError(probe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), default="gemma2b")
    args = parser.parse_args()

    model_name, dtype = MODELS[args.model]
    layers_to_test = LAYERS_BY_MODEL[args.model]
    out_path = CACHE / f"v16_{args.model}_diverse_grid.json"

    print(f"Loading {model_name} on {DEVICE}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()
    decoder_layers = find_decoder_layers(model)
    n_layers = len(decoder_layers)
    d_model, vocab_size = get_model_sizes(model)
    print(
        f"Loaded in {time.time()-t0:.1f}s. n_layers={n_layers}, d_model={d_model}, "
        f"layers_to_test={layers_to_test}"
    )

    prompts = make_prompt_structures(tok)
    persona_keys = list(LONG_PERSONAS.keys())
    task_keys = list(TASKS.keys())

    print("Generating clean references for all XY cells...")
    clean_refs = {}
    for xkey in persona_keys:
        for ykey in task_keys:
            clean_refs[(xkey, ykey)] = greedy_ref_tokens(
                model,
                prompts[(xkey, ykey)]["ids_full"],
                N_PREFIX + N_EVAL,
                tok.eos_token_id,
            )

    print("Capturing localized residuals...")
    localized = {}
    for xkey in persona_keys:
        for ykey in task_keys:
            ref = clean_refs[(xkey, ykey)]
            for probe in PROBES:
                for pkey, tkey in [
                    ("baseline", "baseline"),
                    (xkey, "baseline"),
                    ("baseline", ykey),
                    (xkey, ykey),
                ]:
                    seq_ids, position, _ = build_probe_sequence(prompts[(pkey, tkey)], probe, ref)
                    localized[(xkey, ykey, probe, pkey, tkey)] = {
                        "position": position,
                        "residuals": capture_hidden_at_position(model, seq_ids, position),
                    }

    print("Computing decomposition and causal summaries...")
    decomposition = {}
    causal = {}
    for probe in PROBES:
        decomposition[probe] = {}
        causal[probe] = {}
        for layer in layers_to_test:
            coses = []
            ratios = []
            cos_xys = []
            kls = []
            per_cell = {}
            per_persona = {k: [] for k in persona_keys}
            per_task = {k: [] for k in task_keys}
            for xkey in persona_keys:
                for ykey in task_keys:
                    h_BB = localized[(xkey, ykey, probe, "baseline", "baseline")]["residuals"][layer]
                    h_XB = localized[(xkey, ykey, probe, xkey, "baseline")]["residuals"][layer]
                    h_BY = localized[(xkey, ykey, probe, "baseline", ykey)]["residuals"][layer]
                    h_XY = localized[(xkey, ykey, probe, xkey, ykey)]["residuals"][layer]
                    dx = h_XB - h_BB
                    dy = h_BY - h_BB
                    dxy = h_XY - h_BB
                    dsum = dx + dy
                    inter = dxy - dsum
                    eps = 1e-9
                    norm_dx = np.linalg.norm(dx)
                    norm_dy = np.linalg.norm(dy)
                    norm_dxy = np.linalg.norm(dxy)
                    norm_dsum = np.linalg.norm(dsum)
                    norm_inter = np.linalg.norm(inter)
                    coses.append(float(np.dot(dxy, dsum) / (norm_dxy * norm_dsum + eps)))
                    cos_xys.append(float(np.dot(dx, dy) / (norm_dx * norm_dy + eps)))
                    ratios.append(float(norm_inter / (norm_dxy + eps)))

                    ref = clean_refs[(xkey, ykey)]
                    seq_xy, position_xy, eval_ref = build_probe_sequence(prompts[(xkey, ykey)], probe, ref)
                    clean_logp = get_logp_window(model, seq_xy, eval_ref, vocab_size, None, None)
                    additive = h_BB + dx + dy
                    add_logp = get_logp_window(
                        model,
                        seq_xy,
                        eval_ref,
                        vocab_size,
                        make_sub_hook(additive, position_xy, dtype),
                        decoder_layers[layer],
                    )
                    kl = kl_sum(clean_logp, add_logp)
                    cell = f"{xkey}__{ykey}"
                    per_cell[cell] = kl
                    per_persona[xkey].append(kl)
                    per_task[ykey].append(kl)
                    kls.append(kl)

            decomposition[probe][layer] = {
                "cos_xy_sum": median_summary(coses),
                "inter_over_xy": median_summary(ratios),
                "cos_x_y": median_summary(cos_xys),
            }
            causal[probe][layer] = {
                "summary": median_summary(kls),
                "per_cell": per_cell,
                "per_persona": {k: median_summary(v) for k, v in per_persona.items()},
                "per_task": {k: median_summary(v) for k, v in per_task.items()},
            }

    payload = {
        "model_key": args.model,
        "model_name": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "personas": persona_keys,
        "tasks": task_keys,
        "probes": PROBES,
        "layers_to_test": layers_to_test,
        "decomposition": decomposition,
        "causal": causal,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {out_path}")
    print("\nCausal additive KL medians on the diverse grid:")
    for probe in PROBES:
        print(f"\n[{probe}]")
        for layer in layers_to_test:
            s = causal[probe][layer]["summary"]
            print(
                f"  L={layer:>2d}: median KL={s['median']:.3f}  "
                f"IQR=[{s['iqr'][0]:.3f}, {s['iqr'][1]:.3f}]  max={s['max']:.3f}"
            )


if __name__ == "__main__":
    main()
