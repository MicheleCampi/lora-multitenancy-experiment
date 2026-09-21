# Audit — readers of `schema_version`, before raising it

inferscope at 892cfb6, 2026-09-21. Required by ADR-017, whose Negative
consequences state that every reader of `schema_version` must be checked
before the version rises. Every count below comes from a command.

## What was searched

`schema_version` and `REPORT_SCHEMA_VERSION` across all 202 tracked
files, excluding the ADRs and the CHANGELOG, which name the version in
prose without reading it: 41 lines, 26 in `crates/` and 15 in
`validation-results/`.

## Result

**One reader of the value in production.** `render.rs:60` passes the
version to `render_kvcache`, which gives it to
`HitRateProvenance::resolve` at line 257. That function matches
`(None, Some(_)) => Self::Unknown`, treating every present version
alike, so moving from 1 to 2 does not change what it returns.

**Two writers in production**, `main.rs:394` on the probe path and
`main.rs:640` on the `--sample-only` path. Both write the constant, so
raising it updates both.

**Thirteen writers in tests.** Ten were checked line by line against
the position of their file's test module. The remaining three are
covered another way: `cost_end_to_end.rs:80` sits under
`crates/inferscope/tests/`, and `resource_report.rs:154` and `:199`
follow line 101 in a file with no production code after its test
module.

One file needed a finer check. `trajectory.rs` has two test modules,
at lines 121 and 602, with production code between them (193 to 395).
Its writer at 683 is in the second module. A check that takes the first
`cfg(test)` as the start of test code gets this one right by accident;
the check used here does not rely on that.

**Fifteen archived reports** carry `"schema_version": 1`. They are data
and are not rewritten. After the rise they read, correctly, as
predating ADR-017.

**No assertion compares the version with the literal `1`.**

## Conclusion

Raising `REPORT_SCHEMA_VERSION` changes no behaviour in the code as it
stands. The record belongs with the change it authorises, and is
repeated in the commit that implements ADR-017.
