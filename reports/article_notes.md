# Reading notes: "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, Nov 2026)

Read directly from https://rikiyat.github.io/blog/weight-space-programming/ on 2026-11-19 (including the 12 Nov "Response to Comments").

**Claims and evidence, by strength**
- *Reading weights, small networks (Exp 1–3):* identifying the game for four equal-shaped PPO policies; the shape for three DDPMs;
  the training language, plus induction ("copying") heads, from the middle blocks of three 7.2M char-LMs, without forward passes.
  Strong qualitative evidence on tiny, structured networks with few candidates (n = 3–4 per experiment, 1–2 attempts).
- *Writing/fixing small networks (Exp 4–6):* one-shot fix of a stuck Kinetix policy (1,024/1,024 episodes); MinAtar networks
  written from scratch that beat 100M-step PPO; a Go network (1.5M params) improved with 0–50 forward passes (batches of 128 rollouts).
- *Real LMs (Exp 7–8):* the sycophant could **not** be identified from weights alone (4 attempts); it was identified only once
  attention patterns on agent-written inputs were allowed. That is activation evidence, not weight reading. On Qwen3.5-0.8B, an
  edit to 64 output columns in each of two MLP layers (131,072 weights, 0.016%) removes a looping direction. With 50 "forward passes"
  (average batch 64 sequences, up to 32k tokens each) MATH-500 rises 7.9→48.1 and IFEval 21.0→62.3. With 10 or 25 passes the loops mostly remain.
- *Negative control they report:* Gemma 3 270M, which rarely loops, could not be improved (MATH-500 6.8→7.0–7.7; no change was established).
  The authors conclude that editing "could remove an obstacle that hid an ability, but it could not add an ability the model lacked".

**What is and is not demonstrated**
- The strongest LM result is a narrow failure-mode repair (loops), found with substantial behavioural feedback. Each "forward pass"
  is a batch of up to hundreds of long rollouts, so pass counts are not comparable units of feedback or compute.
- "No gradients" ≠ "no optimization": the Qwen and Go edits iterate on task feedback (keep a change if it wins more).
- Raw-weight-only understanding is shown for tiny networks; for LMs the evidence mixes activations with feedback.
- There are no matched random or untargeted edit controls, and no held-out-phenomenon tests. Baselines are explicitly not intended as fair
  method comparisons (authors' FAQ). Generalization of the Qwen gains to IFEval is consistent with removing a shared obstacle (looping).

**Relevance here:** our setting is closest to Exp 8, with three differences. The goal is a capability (Swedish grammatical preference),
not a failure-mode repair. The feedback budget is tiny (26–104 labelled pairs plus small dev sets). A matched random control is required.
The Gemma 270M negative suggests the prior expectation that targeted edits help less when no single obstacle exists.
