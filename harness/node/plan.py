"""Cell plan of the second dry run (decided on 2026-09-26).

The single source for seedcheck.py, which chooses the seeds before the node,
and for phase3.py, which runs the cells. num_prompts is a multiple of 168
(PROTOCOL.md:434). D2 goes up to 160: 160 requests of 512 tokens take 81920
tokens of the KV cache measured on the node (89968).
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

# (tag, adapters, prompts). r30 has a single seed, used by r30-a and r30-b.
CELLS = (
    [("warmup", ALL8, WARMUP_PROMPTS)]
    + [(f"d2-c{c}", ALL8, D2_PROMPTS) for c in D2_LEVELS]
    + [(f"d3-{k}-r{i}", ads, D3_PROMPTS)
       for i in range(1, D3_REPS + 1)
       for k, ads in (("n1", ["a1"]), ("n8", ALL8))]
    + [("r30", ALL8, R30_PROMPTS)]
)
