#!/usr/bin/env python3
"""V14e — Multi-layer injection: can we improve cache-and-inject by substituting
the additive prediction at multiple layers simultaneously at p_last?

Hypothesis: V14b-v2 showed single-position single-layer injection closes only
7-14% of the host-vs-long gap. The downstream computation attends back to the
persona-text tokens at all layers during multi-token generation. If we clamp
the residual at p_last across multiple layers to the additive prediction at
each layer, the model has fewer "channels" to revert to baseline-host behavior.

Test:
  - Sub at L=14 only (baseline, matches V14b-v2)
  - Sub at L ∈ {10, 12, 14}
  - Sub at L ∈ {10, 12, 14, 16, 18}
  - Sub at L ∈ {10, 12, 14, 16, 18, 20, 22}

For each, measure KL(host + multi-layer-sub → long-prefix-clean) across
8 personas × 3 tasks = 24 cells. Compare to V14b-v2 baseline.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "google/gemma-2-2b-it"
N_REF = 10
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

INJECT_LAYER_SETS = [
    [14],
    [10, 12, 14],
    [10, 12, 14, 16, 18],
    [10, 12, 14, 16, 18, 20, 22],
    [6, 8, 10, 12, 14, 16, 18, 20, 22],
]


def apply_chat(tok, user_text):
    return tok.apply_chat_template([{"role": "user", "content": user_text}],
                                    tokenize=False, add_generation_prompt=True)


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


def get_logp(model, tok, prompt, ref_tokens, hooks_and_layers=None):
    handles = []
    if hooks_and_layers:
        for hook_fn, layer in hooks_and_layers:
            handles.append(
                model.model.layers[layer].register_forward_pre_hook(hook_fn, with_kwargs=True)
            )
    try:
        ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        ref_t = torch.tensor(ref_tokens, device=DEVICE).unsqueeze(0)
        full = torch.cat([ids, ref_t], dim=1)
        prompt_len = ids.shape[1]
        with torch.no_grad():
            out = model(full, return_dict=True)
        logp = torch.zeros(N_REF, model.config.vocab_size, dtype=torch.float32)
        for t in range(N_REF):
            logp[t] = torch.log_softmax(out.logits[0, prompt_len - 1 + t].float(), dim=-1).cpu()
        return logp
    finally:
        for h in handles:
            h.remove()


def kl_sum(p, q):
    total = 0.0
    for t in range(p.shape[0]):
        ex = p[t].exp()
        total += float((ex * (p[t] - q[t])).sum().item())
    return total


def gen_with_hooks(model, tok, prompt, hooks_and_layers=None, n=80):
    handles = []
    if hooks_and_layers:
        for hook_fn, layer in hooks_and_layers:
            handles.append(
                model.model.layers[layer].register_forward_pre_hook(hook_fn, with_kwargs=True)
            )
    try:
        ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=n, do_sample=False, num_beams=1,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()


def main():
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32,
                                                  attn_implementation="eager").to(DEVICE)
    model.eval()

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
    print(f"  captured {len(residuals)}\n")

    # ===================================================================
    # Multi-layer inject test
    # ===================================================================
    print("=" * 80)
    print("V14e — Multi-layer additive injection into baseline-host prompt")
    print("=" * 80)

    all_results = {}
    for inject_set in INJECT_LAYER_SETS:
        set_key = "_".join(str(L) for L in inject_set)
        print(f"\nLayers injected: {inject_set}")
        results = {}
        for pkey in LONG_PERSONAS:
            for tkey in TASKS:
                cell = f"{pkey}__{tkey}"
                long_prompt = prompts[(pkey, tkey)]
                host_prompt = prompts[("BP", tkey)]
                host_p_last = positions[("BP", tkey)]

                # Reference: clean long-prefix
                ids = tok.encode(long_prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    g = model.generate(ids, max_new_tokens=N_REF, do_sample=False, num_beams=1,
                                       pad_token_id=tok.eos_token_id)
                ref_long = g[0, ids.shape[1]:].cpu().tolist()
                logp_long = get_logp(model, tok, long_prompt, ref_long)

                # Build hooks for this inject_set
                hooks = []
                for L in inject_set:
                    h_BB = residuals[("BP", "BT")][L]
                    dx = residuals[(pkey, "BT")][L] - h_BB
                    dy = residuals[("BP", tkey)][L] - h_BB
                    additive = h_BB + dx + dy
                    hooks.append((make_sub_hook(additive, host_p_last), L))

                logp_inject = get_logp(model, tok, host_prompt, ref_long, hooks)
                kl_inject = kl_sum(logp_long, logp_inject)
                results[cell] = kl_inject
        kls = list(results.values())
        all_results[set_key] = {"kls": results, "median": float(np.median(kls)),
                                 "iqr": [float(np.percentile(kls,25)), float(np.percentile(kls,75))],
                                 "max": float(max(kls))}
        print(f"  median KL = {all_results[set_key]['median']:.2f}  "
              f"IQR=[{all_results[set_key]['iqr'][0]:.2f}, {all_results[set_key]['iqr'][1]:.2f}]  "
              f"max={all_results[set_key]['max']:.2f}")

    # Host-clean baseline
    host_kls = []
    for pkey in LONG_PERSONAS:
        for tkey in TASKS:
            long_prompt = prompts[(pkey, tkey)]
            host_prompt = prompts[("BP", tkey)]
            ids = tok.encode(long_prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=N_REF, do_sample=False, num_beams=1,
                                   pad_token_id=tok.eos_token_id)
            ref_long = g[0, ids.shape[1]:].cpu().tolist()
            logp_long = get_logp(model, tok, long_prompt, ref_long)
            logp_host = get_logp(model, tok, host_prompt, ref_long)
            host_kls.append(kl_sum(logp_long, logp_host))
    print(f"\nHost clean baseline: median KL = {np.median(host_kls):.2f}")

    print("\n=== Summary ===")
    print(f"  {'inject layers':>40s}  {'median KL':>9s}  {'closure':>8s}")
    host_med = np.median(host_kls)
    print(f"  {'host (no inject)':>40s}  {host_med:>9.2f}  {'-':>8s}")
    for set_key, res in all_results.items():
        closure = (host_med - res["median"]) / host_med * 100
        print(f"  {set_key:>40s}  {res['median']:>9.2f}  {closure:>7.1f}%")

    # Qualitative: engineer/arch with the largest inject set
    print(f"\n=== Qualitative: engineer/arch with 9-layer inject ===")
    pkey, tkey = "engineer", "arch"
    long_prompt = prompts[(pkey, tkey)]
    host_prompt = prompts[("BP", tkey)]
    host_p_last = positions[("BP", tkey)]
    hooks = []
    for L in INJECT_LAYER_SETS[-1]:
        h_BB = residuals[("BP", "BT")][L]
        dx = residuals[(pkey, "BT")][L] - h_BB
        dy = residuals[("BP", tkey)][L] - h_BB
        additive = h_BB + dx + dy
        hooks.append((make_sub_hook(additive, host_p_last), L))
    text_long = gen_with_hooks(model, tok, long_prompt)
    text_host = gen_with_hooks(model, tok, host_prompt)
    text_multi = gen_with_hooks(model, tok, host_prompt, hooks)
    print(f"\n[LONG (target)]:    {text_long[:280]!r}")
    print(f"\n[HOST clean]:       {text_host[:280]!r}")
    print(f"\n[MULTI-LAYER inj]:  {text_multi[:280]!r}")

    out = {"host_baseline_median": host_med, "results": all_results,
           "qual": {"long": text_long, "host": text_host, "multi_inject": text_multi}}
    with open(CACHE / "v14e_multilayer.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {CACHE}/v14e_multilayer.json")


if __name__ == "__main__":
    main()
