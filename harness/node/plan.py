"""Piano delle celle della seconda prova generale (deciso il 26/9/2026).

Unica fonte per seedcheck.py, che sceglie i semi prima del nodo, e per
phase3.py, che esegue le celle. num_prompts e' un multiplo di 168
(PROTOCOL.md:434). D2 sale fino a 160: 160 richieste da 512 token occupano
81920 token della cache KV misurata sul nodo (89968).
"""
ALL8 = [f"a{i}" for i in range(1, 9)]
IN_LEN = 256
OUT_LEN = 256
BLOCK = 16
D2_LEVELS = (160, 128, 96, 64)
D2_PROMPTS = 672
D3_PROMPTS = 336
D3_REPS = 4
WARMUP_PROMPTS = 24
R30_PROMPTS = 24

# (tag, adattatori, prompt). r30 ha un solo seme, usato da r30-a e da r30-b.
CELLS = (
    [("warmup", ALL8, WARMUP_PROMPTS)]
    + [(f"d2-c{c}", ALL8, D2_PROMPTS) for c in D2_LEVELS]
    + [(f"d3-{k}-r{i}", ads, D3_PROMPTS)
       for i in range(1, D3_REPS + 1)
       for k, ads in (("n1", ["a1"]), ("n8", ALL8))]
    + [("r30", ALL8, R30_PROMPTS)]
)
