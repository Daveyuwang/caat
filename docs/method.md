# Offline evaluation of confidence-based review policies

CAAT evaluates a review policy on saved model outputs. The evaluator takes
an item ID, a score for the saved output, and a binary correctness outcome.
A task-specific grader can supply the outcome, or CAAT can compare a
predicted label to a reference label. The policy accepts outputs with scores
at or above a fixed threshold and defers the rest for review. This contract
supports replay when the original model is unavailable.

## Definition

Let $s_i \in \mathbb{R}$ be a score for output $\hat y_i$, with larger
scores indicating greater confidence. Let $z_{ig} \in \{0,1\}$ indicate
whether grader $g$ counts that output as correct, and define
$a_i(t) = \mathbf{1}\{s_i \ge t\}$. On a batch of $n$ items, the coverage
and risk at cutoff $t$ are

$$
\widehat C(t) = \frac{\sum_i a_i(t)}{n},
\qquad
\widehat R_g(t) = \frac{\sum_i a_i(t)(1-z_{ig})}{\sum_i a_i(t)}.
$$

Risk is undefined at zero coverage. The same estimates apply to any subset
with a known denominator, including a reference class or a supplied group.
The CLI reports the overall and per-subset counts at the supplied cutoff
and sweeps distinct scores for the overall curve. Tied scores enter the
curve together. Sorting costs $O(n\log n)$, followed by a linear scan.
The coverage–risk trade-off follows classical selective classification.
[Geifman and El-Yaniv, NeurIPS 2017](https://arxiv.org/abs/1705.08500)

AUROC compares the scores of correct and incorrect outputs, with half
credit for a tie. AURC averages accepted-set risk over prefix sizes
1 through $n$, averaging over orderings within each tie group. Normalized
AURC gain is $(A_{\mathrm{random}}-A)/(A_{\mathrm{random}}-A_{\mathrm{oracle}})$.
Random ordering gives expected risk equal to the whole-set error rate at
every prefix. Oracle ordering puts all correct outputs first. AUROC and
normalized gain are undefined when all outputs share one correctness value.

When a score is a probability of correctness $p_i$, CAAT also reports
the Brier loss $n^{-1}\sum_i (p_i-z_{ig})^2$ over the scored batch. A
strictly increasing transformation preserves the risk–coverage curve under
the corresponding cutoffs. Probability calibration depends on the score
scale and on the grader's definition of correctness.

For fixed review capacity, compare policies on the same IDs and the same
grader at the same number of accepted outputs. Under uniform selection
from the boundary tie group, the expected additional errors are the number
of slots left times the error fraction in that group. If the task instead
specifies a risk target, choose the policy with design and calibration data
and apply its frozen threshold to a held-out split. `caat select` implements
empirical selection and two versions of Learn then Test, Bonferroni
and fixed sequence.
[Benchmark protocol and results](benchmarks.md) ·
[Learn then Test, §3.2](https://arxiv.org/pdf/2110.01052)

## Closest work

This review covers papers and released tools inspected on 2026-11-15.
Search terms included *selective classification*, *selective QA*,
*confidence-based annotation*, *offline confidence routing*, and
*grader sensitivity*. The current threshold evaluator and acceptance rule
have direct precedents. The practical question is which workflow the toolkit
can make easier or more reliable.

| Work | Overlap with CAAT and implications |
| --- | --- |
| [CalibRoute 0.4.0](https://pypi.org/project/calibroute-ai/0.4.0/) · [Bring-your-own CSV walkthrough](https://github.com/Garyouki/calibroute/blob/main/examples/your_data/README.md) | Closest software substitute. Its standard-library CLI covers saved CSV/JSONL predictions, tie-aware risk–coverage, domain metrics, threshold fitting, independent holdout validation, and machine-readable routing. Its `validate` command checks a frozen policy against a declared risk limit on holdout data and returns exit codes 0, 1, and 2, so the CI gate in `caat audit` has a direct precedent. Its `fit` command selects a threshold without a multiplicity correction, so its valid statement is the holdout check of one frozen threshold. Its input expects a correctness bit and a confidence in `[0, 1]`. A converter script added to its `main` branch after 0.4.0 maps columns and matches labels. CAAT's remaining input difference is that the installed CLI takes finite scores on any scale. CalibRoute has no multi-grader comparison. The installed release reproduced all 1,671 tested CAAT threshold points after a rank-preserving conversion ([local record](benchmarks.md#threshold-selection)). |
| [LM-Polygraph, TACL 2025](https://aclanthology.org/2025.tacl-1.11/) · [Saved benchmark manager](https://lm-polygraph.readthedocs.io/en/latest/components/ue_manager.html) | Closest research toolkit for interchangeable uncertainty estimators and selective generation evaluation. It benchmarks multiple estimators, quality metrics, selective QA and generation tasks, and saves and reloads runs. Its benchmark scores the same estimators under two quality metrics on the same outputs and finds different winners, which is prior evidence that selective-evaluation conclusions depend on the quality metric. Two of its recognized ranking baselines run directly from the token probabilities in CAAT's HELM replay ([ranking results](../results/baselines/ranking.csv)). |
| [Potato Labeling Engine](https://potatoannotator.readthedocs.io/en/latest/labeling-engine/overview/) · [Static sources](https://potatoannotator.readthedocs.io/en/latest/labeling-engine/configuration/#sources) · [Contracts](https://potatoannotator.readthedocs.io/en/stable/labeling-engine/contracts/#selective-risk) | Closest end-to-end framework for annotation. It supports existing labels imported from static sources, confidence-based acceptance, human review, class-level risk contracts via Learn then Test, and CLI, API, and MCP surfaces. The original course dashboard and routing idea overlap substantially with this workflow. |
| [Tavakoli and Zamani, 2025](https://arxiv.org/pdf/2507.00543) · [Jung, Brahman, and Choi, ICLR 2025](https://arxiv.org/abs/2407.18370) | Direct method precedents. The former combines confidence and inter-model disagreement to allocate human review for search clarification annotation. The latter calibrates when to trust or escalate LLM pairwise judgments using fixed-sequence tests and human agreement as the outcome. Confidence thresholding, disagreement, and escalation are established mechanisms. |
| [Wang et al., CHI 2024](https://megagon.ai/human-llm-collab-annote-thru-llm-labels/) · [Bachar et al., 2026](https://arxiv.org/abs/2601.07006) · [Kamath, Jia, and Liang, ACL 2020](https://aclanthology.org/2020.acl-main.503/) | Direct precedents for the course trust scorer. Each trains a small model on confidence and input features to predict whether a model output is correct, then abstains or sends low-scored outputs to people. Bachar et al. add probability calibration and a cost-based escalation threshold. A fine-tuned encoder's probability as a feature is a variant of this mechanism. |
| [Gupta et al., 2024](https://arxiv.org/abs/2404.10136) · [Pathak et al., 2026](https://arxiv.org/abs/2604.11996) | Precedents for one-response token features. Gupta et al. use sum, mean, and quantile token log probabilities with a learned rule and analyze length bias. Pathak et al. score a reasoning trace by its lowest-probability tenth of tokens. CAAT's low-token and fused scores follow these designs. |
| [Same Outputs, Different Scores, 2026](https://arxiv.org/html/2611.03377v1) · [HELM](https://crfm-helm.readthedocs.io/en/latest/tutorial/) | Hasan and Lwowski vary the scoring rule (answer parser and parse-failure fallback) inside the official scorers of eight multiple-choice benchmarks on identical multimodal outputs, adjudicate discrepancies, and publish artifacts for reproduction. They report rule disagreements and ranking effects, with no thresholds or selective prediction. HELM already preserves model outputs and grader results. The narrower opportunity for CAAT is to test how grader changes propagate to a frozen review policy's accepted risk and release decision. |
| [Valid Per-Field Selective Risk Control, 2026](https://arxiv.org/abs/2608.14639) · [Selective Classification Can Magnify Disparities, ICLR 2021](https://arxiv.org/abs/2010.14134) | Group breakdowns and risk-control protocols also have precedents. The document-extraction study uses grouped Learn then Test and measures the effects of clustering, ties, and split leakage. It also re-grades a sample of accepted fields with human annotators, compares two matching protocols on the same predictions, and ships a CLI and MCP server. It is the closest precedent for checking accepted-set risk under a second grader, though it does not report both graders at one frozen threshold. Group reporting alone offers little research distinction. |
| [Santilli et al., ACL 2025](https://arxiv.org/abs/2504.13677) · [Ielanskyi et al., 2025](https://arxiv.org/abs/2510.02279) | Closest evidence that uncertainty evaluation depends on the correctness function. Both show that AUROC values and method rankings change with the grader, and Santilli et al. trace part of the effect to length bias shared by the score and the grader. Both work at the ranking level, with no fixed policy or risk limit. |
| [Learn then Test](https://arxiv.org/abs/2110.01052) · [Feng et al., 2026](https://arxiv.org/abs/2601.20913) · [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | The two-grader threshold rule in the module study is Learn then Test's multiple-risk construction with graders in place of losses, so it adds no new statistical method. Feng et al. test a failure rate from judge labels using judge error rates estimated on a human-labeled set. MAPIE implements Learn then Test with Bonferroni, Holm, and fixed-sequence corrections as a library over arrays. |

## Paired grader audit

Freeze the output, score, accept/review rule, and split. Join each grader's
binary outcomes on the same output identities. The accepted-set shift
from grader $g$ to grader $h$ is

$$
D_{gh}(t)=\widehat R_h(t)-\widehat R_g(t)
=\frac{\sum_i a_i(t)(z_{ig}-z_{ih})}{\sum_i a_i(t)}.
$$

The numerator counts newly failed outputs minus newly passed outputs.
Compute $D_{gh}(\mathrm{all})$ with every output accepted. Then
$E_{gh}(t)=D_{gh}(t)-D_{gh}(\mathrm{all})$ measures the excess net shift
associated with the accepted set. A zero excess can coexist with
disagreements in both directions. CAAT reports those counts, the affected
IDs, and whether the grader change moves the accepted-set risk across a
declared limit. Preserve the output identity along with the prompt ID
when the same prompt can produce several saved responses.

On the WildBench Qwen probe, the Llama-to-GPT shift added 6.41 percentage
points beyond the whole-set offset at 50% coverage (95% CI 3.61 to 8.72).
Across the 16 eligible WildBench models, 30 of 48 grader-pair intervals
included zero, meeting the plan's stopping rule for the broader study. At
the same coverage, three of four deterministic grader controls included
zero. Qwen IFEval had excess −2.23 points (95% CI −4.63 to −0.015).
[Judge replay, controls, and provenance](judge-study.md)

An empirical target crossing is useful even when the excess is small.
A cutoff chosen with the Google IFEval regrade accepted 144 held-out Qwen
responses. The risk was 22.92% under Google and 31.25% under HELM, crossing
a 30% limit on the same responses. For selecting a policy under several
graders, the maximum of their one-sided binomial p-values tests the null
that at least one grader's risk is at or above the limit at a fixed
threshold. Bonferroni corrects for a search over the fixed grid.
Fixed-sequence testing evaluates from the strictest cutoff and stops at
the first failure. The resulting statement concerns the specified grades
under the calibration sampling assumptions. [Held-out policy results](module-study.md) ·
[Learn then Test](https://arxiv.org/abs/2110.01052)

## Open-source use

`caat audit` accepts one aligned CSV or separate grader exports. The latter
checks that every grader file matches the prediction IDs and output
identities. It reports what changed at a frozen cutoff and identifies the
affected rows. Versioned JSON and exit codes let CI consume the output.
The evaluator and audit run on the standard library. Dataset preparation
and scorer fitting live in standalone benchmark scripts.
[Runnable examples](../README.md#compare-graders-on-the-same-outputs)

The release criterion for this workflow is that an outside user can replay
their own artifacts, interpret the report, and diagnose an incorrect join
without writing a custom adapter for the documented CSV format. Evaluate
additional integrations by the effort they save in that workflow. Benchmark
replays document the tested behavior and provide evidence for a technical
report.

## Confidence score design

The scorer estimates which saved predictions are likely to be correct.
Calibration maps a score to a correctness probability for a specified
grader. The policy converts scores to accept/review decisions at a budget
or risk target. Changing the scorer can improve ranking, while probability
calibration and risk control answer different questions.

The course trust scorer combines Qwen self-report, DeBERTa output and
agreement, and text features in an isotonic-calibrated logistic model. In
the retrospective replay at 75% coverage on the same saved Qwen labels,
DeBERTa's probability of the Qwen label had 1.69% accepted error on Jigsaw
and 35.91% on FEVER. The full scorer had 1.78% and 37.60%. IMDb favored the
full scorer by less than 0.1 percentage point at this coverage. These
comparisons motivate using the single-feature result as a baseline for
each classification task. The course sample boundaries limit
generalization, especially for Jigsaw.
[Retrospective comparisons](../results/baselines/course.csv) ·
[Maximum softmax probability baseline](https://arxiv.org/abs/1610.02136)

Start a scorer comparison with signals already available from one response.
For classification, compare the probability assigned to the frozen label
and the margin against the strongest alternative. For generation, compare
mean and sum token log probability, answer-span log probability when a
span is defined, and a length-only control. Add format validity or
truncation as features if available before acceptance. Freeze the grader
and keep gold labels out of feature computation. Compare individual
features against a regularized logistic model trained on the design
split, then evaluate each model on the same held-out IDs and budgets.
Calibrate probabilities on a separate split and score calibration with
Brier loss. If calibration and risk certification both use labels, allocate
independent data for them or predefine the full selection procedure.
[MaxProb and supervised correctness calibration](https://aclanthology.org/2020.acl-main.503/)

The initial comparison found a useful candidate for multiple-choice tasks.
The log probability of the explicit answer token improved the held-out
ranking over mean token log probability on both MMLU-Pro models at 75%
coverage. The gain on Llama was clearer than the gain on Qwen. The
benchmarks keep scores configurable by task and evaluate each score on the
same saved outputs. [Score ablations](module-study.md#confidence-scores)

Use P(True) or semantic entropy as additional comparators when saved
verifier outputs or repeated generations are available. They require more
inference than the one-response scores on the current archives. Compare
matched-coverage risk and coverage at a declared risk target, alongside
model calls, tokens, and runtime. Keep a new feature group if it lowers
held-out risk at the same review budget or achieves similar risk with
less inference. Repeat the comparison using a second grader where a
version change can affect the policy.
[P(True)](https://arxiv.org/abs/2207.05221) ·
[Semantic entropy](https://www.nature.com/articles/s41586-024-07421-0)
