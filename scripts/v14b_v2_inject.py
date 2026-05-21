#!/usr/bin/env python3
"""V14b-v2 — Use-case-relevant cache-and-inject.

Previous V14b tried injecting v_X into a *bare* task prompt with no "As ___,"
frame. That fails because the bare prompt is structurally different from the
long-persona prompt (different token sequence, different KV cache contents).

V14b-v2 keeps the prompt structure consistent: use a baseline-persona host
prompt ("As a thoughtful person, [task]"), substitute the additive prediction
at p_last at L=14, and compare downstream output to the clean long-persona
prompt's output.

Two interventions per cell:
  (1) "Oracle": substitute h_XY's residual at p_last L=14 into the
      baseline-persona prompt's p_last L=14 → tests upper bound for inject.
  (2) "Cached": substitute h(baseline_P, baseline_T) + Δ_X + Δ_Y → tests
      the realistic use case where Δ_X was extracted once and recombined
      with Δ_Y(new task).

Compare each against: clean baseline-persona-prompt generation (no inject)
and clean long-persona-prompt generation (target).
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
INJECT_L = 14

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


def get_logp(model, tok, prompt, ref_tokens):
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


def get_logp_with_hook(model, tok, prompt, ref_tokens, hook_fn, layer):
    handle = model.model.layers[layer].register_forward_pre_hook(hook_fn, with_kwargs=True)
    try:
        return get_logp(model, tok, prompt, ref_tokens)
    finally:
        handle.remove()


def kl_sum(p, q):
    total = 0.0
    for t in range(p.shape[0]):
        ex = p[t].exp()
        total += float((ex * (p[t] - q[t])).sum().item())
    return total


def gen_with_hook(model, tok, prompt, hook_fn=None, layer=None, n=80):
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


def main():
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32,
                                                  attn_implementation="eager").to(DEVICE)
    model.eval()

    # Capture residuals we need
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
    print(f"  captured {len(residuals)}")

    print("\n" + "=" * 80)
    print(f"V14b-v2 — Inject into baseline-persona host prompt at L={INJECT_L}")
    print("=" * 80)
    print(f"Procedure: host prompt = 'As {BASELINE_PERSONA}, [task]'.")
    print(f"  Reference = clean 'As [long X], [task]' generation.")
    print(f"  (1) Oracle: substitute h_XY's residual at L={INJECT_L} into host at p_last.")
    print(f"  (2) Cached: substitute h_BB + Δ_X + Δ_Y at L={INJECT_L} into host at p_last.")
    print(f"  (3) Control: clean host (no inject).\n")

    results = {}
    for pkey in LONG_PERSONAS:
        for tkey in TASKS:
            cell = f"{pkey}__{tkey}"

            long_prompt = prompts[(pkey, tkey)]
            host_prompt = prompts[("BP", tkey)]
            host_p_last = positions[("BP", tkey)]

            # Get reference: clean long-persona generation
            ids = tok.encode(long_prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=N_REF, do_sample=False, num_beams=1,
                                   pad_token_id=tok.eos_token_id)
            ref_long = g[0, ids.shape[1]:].cpu().tolist()

            logp_long = get_logp(model, tok, long_prompt, ref_long)
            logp_host_clean = get_logp(model, tok, host_prompt, ref_long)
            kl_host_clean = kl_sum(logp_long, logp_host_clean)

            # (1) Oracle: substitute h_XY at p_last L=14 into host prompt
            h_XY = residuals[(pkey, tkey)][INJECT_L]
            logp_oracle = get_logp_with_hook(
                model, tok, host_prompt, ref_long,
                make_sub_hook(h_XY, host_p_last), INJECT_L
            )
            kl_oracle = kl_sum(logp_long, logp_oracle)

            # (2) Cached additive: substitute h_BB + Δ_X + Δ_Y at p_last L=14 into host prompt
            h_BB = residuals[("BP", "BT")][INJECT_L]
            dx = residuals[(pkey, "BT")][INJECT_L] - h_BB
            dy = residuals[("BP", tkey)][INJECT_L] - h_BB
            additive = h_BB + dx + dy
            logp_cached = get_logp_with_hook(
                model, tok, host_prompt, ref_long,
                make_sub_hook(additive, host_p_last), INJECT_L
            )
            kl_cached = kl_sum(logp_long, logp_cached)

            results[cell] = {
                "kl_host_clean": kl_host_clean,
                "kl_oracle":     kl_oracle,
                "kl_cached":     kl_cached,
                "oracle_ratio":  kl_oracle / max(kl_host_clean, 1e-9),
                "cached_ratio":  kl_cached / max(kl_host_clean, 1e-9),
            }
            print(f"  {cell:>22s}: KL(host)={kl_host_clean:>5.2f}  "
                  f"KL(oracle)={kl_oracle:>5.2f}  KL(cached)={kl_cached:>5.2f}  "
                  f"oracle_r={results[cell]['oracle_ratio']:.2f}  cached_r={results[cell]['cached_ratio']:.2f}")

    # Summary
    print("\n=== V14b-v2 summary across 24 cells ===")
    kls_host = [r["kl_host_clean"] for r in results.values()]
    kls_oracle = [r["kl_oracle"]    for r in results.values()]
    kls_cached = [r["kl_cached"]    for r in results.values()]
    print(f"  KL(host clean → long)          median={np.median(kls_host):>5.2f}  IQR=[{np.percentile(kls_host,25):.2f}, {np.percentile(kls_host,75):.2f}]")
    print(f"  KL(host + oracle sub → long)   median={np.median(kls_oracle):>5.2f}  IQR=[{np.percentile(kls_oracle,25):.2f}, {np.percentile(kls_oracle,75):.2f}]")
    print(f"  KL(host + cached sub → long)   median={np.median(kls_cached):>5.2f}  IQR=[{np.percentile(kls_cached,25):.2f}, {np.percentile(kls_cached,75):.2f}]")
    print(f"  Cells where ORACLE beats host: {sum(1 for r in results.values() if r['oracle_ratio'] < 1)}/{len(results)}")
    print(f"  Cells where CACHED beats host: {sum(1 for r in results.values() if r['cached_ratio'] < 1)}/{len(results)}")

    # Qualitative: engineer/arch
    print(f"\n=== Qualitative: engineer/arch ===")
    pkey, tkey = "engineer", "arch"
    long_prompt = prompts[(pkey, tkey)]
    host_prompt = prompts[("BP", tkey)]
    host_p_last = positions[("BP", tkey)]
    h_BB = residuals[("BP", "BT")][INJECT_L]
    dx = residuals[(pkey, "BT")][INJECT_L] - h_BB
    dy = residuals[("BP", tkey)][INJECT_L] - h_BB
    additive = h_BB + dx + dy
    h_XY = residuals[(pkey, tkey)][INJECT_L]

    text_long = gen_with_hook(model, tok, long_prompt)
    text_host = gen_with_hook(model, tok, host_prompt)
    text_oracle = gen_with_hook(model, tok, host_prompt, make_sub_hook(h_XY, host_p_last), INJECT_L)
    text_cached = gen_with_hook(model, tok, host_prompt, make_sub_hook(additive, host_p_last), INJECT_L)
    print(f"\n[LONG (target)]:    {text_long[:280]!r}")
    print(f"\n[HOST (no inject)]: {text_host[:280]!r}")
    print(f"\n[ORACLE inject]:    {text_oracle[:280]!r}")
    print(f"\n[CACHED inject]:    {text_cached[:280]!r}")

    out = {"results": results,
           "qual": {"long": text_long, "host": text_host,
                    "oracle": text_oracle, "cached": text_cached}}
    with open(CACHE / "v14b_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {CACHE}/v14b_v2.json")


if __name__ == "__main__":
    main()
