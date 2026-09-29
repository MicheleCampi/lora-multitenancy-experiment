# Faulty node, 2026-09-28

Files from a node rented from Lambda on 2026-09-28 for the second dry run,
one NVIDIA A10, on which vLLM could not start. They were copied from the
node before it was terminated; there is no archive. The second dry run ran
on another node.

- `phase2.out`: printed by `harness/node/phase2.py`: the server, PID 3969,
  exited with code 1.
- `serve.log`: the server's log. Its engine process, PID 4105, loaded the
  model and failed at 17:22:55, during its start, with
  `torch.AcceleratorError: CUDA error: uncorrectable ECC error encountered`
  (line 163).
- `ecc.txt`: the ECC report of `nvidia-smi`, taken at 17:25:15 with driver
  580.105.08: 301 uncorrectable DRAM errors in the volatile count and
  373,452 in the aggregate count.
