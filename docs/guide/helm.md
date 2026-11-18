# Import a HELM run

Use a run directory containing `scenario_state.json` with one text completion and token log probabilities per instance. For a published metric, include `per_instance_stats.json` from the same run. Full credit counts as correct. [HELM run format](https://crfm-helm.readthedocs.io/en/latest/tutorial/#using-helm-run)

```bash
caat helm HELM_IFEVAL_RUN --metric ifeval_strict_accuracy --output replay/ifeval-helm.csv
caat policy replay/ifeval-helm.csv --score-column mean_token_logprob --correct-column ifeval_strict_accuracy --risk-target 0.4 --holdout-confidence 0.95
```

## Check a saved judge change

Read Boolean grades with `--annotation GROUP.FIELD`. The Omni-MATH example selects under GPT and checks the same held-out predictions under Llama.

```bash
caat helm HELM_OMNI_MATH_RUN --annotation omni_math.gpt_equivalence_judgement --annotation omni_math.llama_equivalence_judgement --output replay/omni-helm.csv
caat policy replay/omni-helm.csv --score-column mean_token_logprob --correct-column omni_math.gpt_equivalence_judgement --check-column omni_math.llama_equivalence_judgement --risk-target 0.8 --holdout-confidence 0.95
```
