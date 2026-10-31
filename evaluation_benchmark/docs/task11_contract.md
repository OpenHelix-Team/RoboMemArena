# Task11: cookies in the top drawer, butter in the middle drawer

The Task11 instruction is:

> Put cookies into the top drawer, then put butter into the middle drawer.

This contract applies to the generic adapter prompt, the minimal OpenPI prompt,
both BDDL language descriptions, and both VLM/VLA reference task descriptions.
Task13 instead places both cookies and butter in the middle drawer; Task14 places
cookies in the top drawer and chocolate in the middle drawer.

The existing BDDL goals and stage predicates already require `cookies_1` in
`wooden_cabinet_1_top_region` and `butter_1` in
`wooden_cabinet_1_middle_region`. Their geometry, thresholds, ordering, and six
stages are unchanged. Opening the top drawer, placing cookies, closing the top
drawer, opening the middle drawer, and placing butter are the five counted
stages. The final middle-drawer close remains optional. Putting only chocolate
in the middle drawer cannot satisfy the butter stage: completing the other four
counted stages yields CSR 80% and TSR failure. Completing the butter stage yields
CSR 100% and TSR success even if the final optional close is incomplete.

The reference configurations retain historical `pick_chocolate` and
`place_chocolate` primitive stems and labels solely for compatibility with
existing training data and VLA action labels. Those legacy names do not specify
the Task11 target object. This change does not relabel or retrain that data.

Results produced with the previous chocolate instruction retain their original
instruction and scorer provenance. Evaluate the corrected instruction in a new
run; do not overwrite or silently combine it with historical Task11 results.

Run the CPU contract checks from `evaluation_benchmark`:

```bash
python -m unittest discover -s tests -p 'test_task*contract*.py' -v
```

The scorer regression compiles the actual functions from both scorer entrypoints
and supplies synthetic object positions. It checks target identity and the
optional closing stage without loading the simulator or a model; it is not a
robot rollout.

## Verification on 2026-10-31

The five Task11/prompt contract tests passed. Both scorer copies were tested
with distinct top/middle site coordinates: cookies at the top and butter at the
top while chocolate is in the middle fails; moving butter to the middle passes.
The cookie placement predicate is evaluated as well as the butter predicate.
Both JSON configurations were compared structurally with base commit
`c8994677ddbbd1c74324ba98bcf15f9b9305497f`: all other 25 task blocks,
non-task configuration, and Task11 `primitive_order` remain unchanged. Both BDDL
goal sections also remain byte-for-byte unchanged.

The broader command `python -m unittest discover -s tests -v` ran 34 tests:
31 passed and three existing counting-pour tests failed. Two call
`_body_pour_stage` without its required `target_kind` and `target_name`; the other
expects `goal_success is None` where the current runner returns `True`. Repeating
`test_counting_pour_stage.py` against an isolated copy of the base commit's
unmodified scripts and test reproduced exactly the same two errors and one
failure (six tests total). Those unrelated tests and scoring code were left
unchanged. This is not an all-suite pass or a simulator rollout result.
