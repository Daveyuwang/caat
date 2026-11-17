# Use a saved HELM run directory

Pass `caat helm` one directory from `benchmark_output/runs/SUITE/`.
Its `scenario_state.json` needs one successful text completion with
recorded token log probabilities per instance. The CSV keeps the HELM
instance ID and the SHA-256 of the exact completion. It includes mean,
summed, and lowest-tenth token log probability and the token count.
[HELM run files](https://crfm-helm.readthedocs.io/en/latest/tutorial/#using-helm-run)

## Use a published per-instance metric

Place `per_instance_stats.json` from the same run in the directory.
`--metric` matches the named statistic by instance ID and training trial.
The selected metric must have one observation in [0, 1] for each response.
CAAT records `1` for full credit and `0` for any lower value. Repeat
`--metric` to use several metrics. For IFEval, this checks whether the
response passed every instruction under HELM's strict grader.

```bash
caat helm HELM_IFEVAL_RUN --metric ifeval_strict_accuracy --output replay/ifeval-helm.csv
caat policy replay/ifeval-helm.csv --score-column mean_token_logprob --correct-column ifeval_strict_accuracy --risk-target 0.4 --holdout-confidence 0.95
```

On the HELM Capabilities v1.0.0 Qwen2.5-7B-Instruct-Turbo run, all 541
rows matched the archived confidence and grade columns. With seed 42,
the policy accepted 92 of 217 held-out answers, of which 23 were wrong
(25.00% error, 33.52% one-sided 95% upper bound).
[IFEval replay provenance](../research/benchmarks.md#replay)

## Check a saved judge change

Use `--annotation GROUP.FIELD` for a Boolean outcome in the request
state. Each requested annotation needs a verdict on every exported row.
In this Omni-MATH run, GPT and Llama supplied complete verdicts.
Fit the cutoff under GPT and check the same holdout responses under Llama.
[Omni-MATH run and judge provenance](../research/judge-study.md#archived-inputs)

```bash
caat helm HELM_OMNI_MATH_RUN --annotation omni_math.gpt_equivalence_judgement --annotation omni_math.llama_equivalence_judgement --output replay/omni-helm.csv
caat policy replay/omni-helm.csv --score-column mean_token_logprob --correct-column omni_math.gpt_equivalence_judgement --check-column omni_math.llama_equivalence_judgement --risk-target 0.8 --holdout-confidence 0.95
```

The 80% risk target is for this replay. The cutoff accepted 398 of 400
held-out answers. GPT counted 278 errors and Llama counted 280. The
one-sided 95% upper error bounds were 73.64% and 74.12%, respectively.
With a separate labeled holdout file, pass these two columns to
`caat audit` with `--output-key-column output_sha256` and the frozen
cutoff. [Audit guide](ci-audit.md)
