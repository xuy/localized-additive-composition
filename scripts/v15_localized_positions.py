#!/usr/bin/env python3
"""V15 — Localized multi-position residual composition near answer formation.

Extends the V12 2x2 residual decomposition beyond p_last only. For each
persona-task cell (X, Y), we probe a small local band of positions:

  - p_user_last: last token of the user turn before the generation header
  - p_last:      last prompt token before generation
  - g1:          first generated token position, teacher-forced on clean XY
  - g2:          second generated token position, teacher-forced on clean XY

At each probe position and layer, we compute the usual decomposition:

  Δ_X    = h_XB - h_BB
  Δ_Y    = h_BY - h_BB
  Δ_XY   = h_XY - h_BB
  Inter  = Δ_XY - Δ_X - Δ_Y

We then run additive causal substitution (h_BB + Δ_X + Δ_Y) at a sparse layer
grid for each probe position and measure 10-token KL against the clean XY
continuation from that same probe prefix.
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
    "qwen3b": ("Qwen/Qwen2.5-3B-Instruct", torch.bfloat16),
}

PERSONAS = {
    "baseline": "a thoughtful person",
    "buffett": "Warren Buffett",
    "marx": "Karl Marx",
    "yoda": "Yoda",
    "angelou": "Maya Angelou",
}
TASKS = {
    "baseline": "Give advice to someone facing a difficult decision.",
    "policy": "Comment on whether universal basic income is a good policy.",
    "haiku": "Write a haiku about Monday mornings.",
    "recommend": "Recommend a book worth reading and explain why.",
}
NON_BASELINE_PERSONAS = ["buffett", "marx", "yoda", "angelou"]
NON_BASELINE_TASKS = ["policy", "haiku", "recommend"]
PROBES = ["p_user_last", "p_last", "g1", "g2"]


def apply_chat(tok, user_text, add_generation_prompt=True):
    msgs = [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=add_generation_prompt
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
    if hasattr(model, "language_model"):
        lm = model.language_model
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


def make_prompt_structures(tok):
    prompts = {}
    for pkey, ptext in PERSONAS.items():
        for tkey, ttext in TASKS.items():
            user_text = f"As {ptext}, {ttext}"
            prompt_no_gen = apply_chat(tok, user_text, add_generation_prompt=False)
            prompt_full = apply_chat(tok, user_text, add_generation_prompt=True)
            ids_no_gen = encode(tok, prompt_no_gen)
            ids_full = encode(tok, prompt_full)
            prompts[(pkey, tkey)] = {
                "user_text": user_text,
                "prompt_no_gen": prompt_no_gen,
                "prompt_full": prompt_full,
                "ids_full": ids_full,
                "p_user_last": len(ids_no_gen) - 1,
                "p_last": len(ids_full) - 1,
            }
    return prompts


def capture_hidden_at_position(model, input_ids, position):
    ids = torch.tensor([input_ids], device=DEVICE)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, return_dict=True)
    arr = np.stack([h[0, position].float().cpu().numpy() for h in out.hidden_states])
    return arr


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


def layer_grid(n_layers):
    raw = [int(round(f * n_layers)) for f in [0.25, 0.40, 0.55, 0.70, 0.85]]
    clipped = sorted({min(max(v, 1), n_layers - 1) for v in raw})
    return clipped


def build_probe_sequence(prompt_info, probe, clean_xy_ref):
    ids_full = prompt_info["ids_full"]
    if probe == "p_user_last":
        return ids_full, prompt_info["p_user_last"], clean_xy_ref[:N_EVAL]
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
    out_path = CACHE / f"v15_{args.model}_localized_positions.json"

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
    layers_to_test = layer_grid(n_layers)
    print(
        f"Loaded in {time.time()-t0:.1f}s. n_layers={n_layers}, d_model={d_model}, "
        f"layers_to_test={layers_to_test}"
    )

    prompts = make_prompt_structures(tok)

    print("Generating clean XY references...")
    clean_refs = {}
    for xkey in NON_BASELINE_PERSONAS:
        for ykey in NON_BASELINE_TASKS:
            prompt_info = prompts[(xkey, ykey)]
            clean_refs[(xkey, ykey)] = greedy_ref_tokens(
                model,
                prompt_info["ids_full"],
                N_PREFIX + N_EVAL,
                tok.eos_token_id,
            )

    print("Capturing localized residuals...")
    localized = {}
    for xkey in NON_BASELINE_PERSONAS:
        for ykey in NON_BASELINE_TASKS:
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

    print("Computing decomposition summaries...")
    decomp = {}
    for probe in PROBES:
        decomp[probe] = {}
        for layer in range(n_layers + 1):
            cos_xy_sum = []
            cos_x_y = []
            inter_over_xy = []
            for xkey in NON_BASELINE_PERSONAS:
                for ykey in NON_BASELINE_TASKS:
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
                    cos_xy_sum.append(float(np.dot(dxy, dsum) / (norm_dxy * norm_dsum + eps)))
                    cos_x_y.append(float(np.dot(dx, dy) / (norm_dx * norm_dy + eps)))
                    inter_over_xy.append(float(norm_inter / (norm_dxy + eps)))
            decomp[probe][layer] = {
                "cos_xy_sum": median_summary(cos_xy_sum),
                "cos_x_y": median_summary(cos_x_y),
                "inter_over_xy": median_summary(inter_over_xy),
            }

    print("Running sparse causal substitution sweep...")
    causal = {}
    for probe in PROBES:
        causal[probe] = {}
        for layer in layers_to_test:
            kls = []
            per_cell = {}
            for xkey in NON_BASELINE_PERSONAS:
                for ykey in NON_BASELINE_TASKS:
                    ref = clean_refs[(xkey, ykey)]
                    seq_xy, position_xy, eval_ref = build_probe_sequence(
                        prompts[(xkey, ykey)], probe, ref
                    )
                    if len(eval_ref) < N_EVAL:
                        raise RuntimeError(f"Probe {probe} did not leave {N_EVAL} eval tokens")
                    h_BB = localized[(xkey, ykey, probe, "baseline", "baseline")]["residuals"][layer]
                    h_XB = localized[(xkey, ykey, probe, xkey, "baseline")]["residuals"][layer]
                    h_BY = localized[(xkey, ykey, probe, "baseline", ykey)]["residuals"][layer]
                    dx = h_XB - h_BB
                    dy = h_BY - h_BB
                    additive = h_BB + dx + dy
                    clean_logp = get_logp_window(
                        model, seq_xy, eval_ref, vocab_size, None, None
                    )
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
                    kls.append(kl)
            causal[probe][layer] = {
                "summary": median_summary(kls),
                "per_cell": per_cell,
            }

    payload = {
        "model_key": args.model,
        "model_name": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "layers_to_test": layers_to_test,
        "probes": PROBES,
        "n_eval": N_EVAL,
        "decomposition": decomp,
        "causal": causal,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {out_path}")
    print("\nDecomposition medians at tested probes/layers:")
    for probe in PROBES:
        print(f"\n[{probe}]")
        print(f"  {'L':>4s}  {'cos(XY,sum)':>11s}  {'I/XY':>7s}  {'cos(X,Y)':>9s}")
        for layer in layers_to_test:
            d = decomp[probe][layer]
            print(
                f"  {layer:>4d}  {d['cos_xy_sum']['median']:>11.3f}  "
                f"{d['inter_over_xy']['median']:>7.3f}  {d['cos_x_y']['median']:>9.3f}"
            )

    print("\nCausal additive KL medians:")
    for probe in PROBES:
        print(f"\n[{probe}]")
        for layer in layers_to_test:
            s = causal[probe][layer]["summary"]
            print(
                f"  L={layer:>2d}: median KL={s['median']:.3f}  "
                f"IQR=[{s['iqr'][0]:.3f}, {s['iqr'][1]:.3f}]"
            )


if __name__ == "__main__":
    main()
