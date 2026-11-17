# Terms in CAAT reports

Each entry gives the meaning CAAT uses, where the word appears, and how to
read the number next to it.

## The decision

**Score.** One number per prediction, where a larger value means more
confidence. A model's top class probability, a mean token log probability,
and a self-reported confidence are all scores. Use `--direction lower` or
`--lower-is-better` when a smaller value means more confidence.

**Grader.** The source of the 0/1 outcome for each prediction. A column of
human labels is a grader, and so is an automatic check. With
`--prediction-column` and `--reference-column`, the grader is exact match
and takes the reference column's name.

**Cutoff, or threshold.** Predictions with a score at or above the cutoff
are accepted. The rest go to review.

**Coverage.** The share of rows a cutoff accepts.

**Accepted error, or risk.** The share of accepted rows that the grader
marks wrong. This is the quantity a policy controls. Rows sent to review do
not count toward it.

**Risk target.** The largest accepted error you are willing to allow, set
with `--risk-target`. A 5% target reads "at most 5 in 100 accepted
predictions may be wrong".

## The split

`caat policy` and `caat scores` divide the rows by ID with a seed.

| Part | Share | Used for |
| --- | ---: | --- |
| Design | 20% | Fixing the candidate cutoffs and choosing among scores |
| Calibration | 40% | Testing the candidates and selecting one cutoff |
| Held-out | 40% | Checking the selected cutoff on rows it has not seen |

Only the held-out rows give an error estimate, because the design and
calibration rows chose the cutoff. The line "Applied to all N rows" counts
how many rows the cutoff accepts in the whole file. It is a count for
planning work, and it carries no error estimate.

## Selection rules

A rule decides which candidate cutoff to keep. Set it with `--rule`.

**`empirical`.** Keeps the most permissive cutoff whose calibration error
is at or below the target. It gives no statistical guarantee. In the
[rule study](../research/benchmarks.md#default-selection-rule), its
held-out error exceeded the target in 22.6% of splits.

**`ltt-bonferroni`** (default), **`ltt-holm`**, **`ltt-fixed-sequence`.**
Learn then Test rules. Each candidate gets an exact binomial test of
"the accepted error is above the target". The rule keeps a cutoff only if
the data reject that statement, after a correction for testing several
candidates. Bonferroni divides the failure rate evenly across candidates.
Holm does the same in steps and can keep slightly more. Fixed sequence
tests from the strictest cutoff downward and stops at the first failure.
[Learn then Test](https://arxiv.org/abs/2110.01052)

**Failure rate.** `--failure-rate`, 0.05 by default. A Learn then Test rule
keeps the accepted error at or below the target with probability at least
one minus the failure rate, over the draw of calibration rows. The summary
states this as "with 95% confidence on new rows like the calibration rows".
The statement assumes new rows come from the same population as the
calibration rows and are graded the same way.

**Why a candidate under the target can still fail.** A Learn then Test rule
needs enough accepted rows to rule out chance. Four accepted rows with no
errors have a 0% observed rate, but four correct rows in a row happen 13% of
the time when the true error is 40%, so a 40% target is not shown. The
summary reports this case and names the count. More calibration rows or
fewer candidates raise the chance of selecting a cutoff.

**p-value and test level.** Each candidate's p-value measures how
surprising its calibration counts would be if its true error were at the
target. A candidate passes when the p-value is at or below its test level.
With `ltt-bonferroni` and ten candidates, the level is the 5% failure rate
divided by ten, or 0.005. The summary names the next looser candidate and
why it was not selected, which explains a result well under the target.

**No cutoff selected.** The command exits with status 1 and reports the
lowest calibration error any candidate reached. A stronger score, a larger
target, or more calibration rows are the ways forward.

## Settings that move the result

The rule, the number of candidates, and the split seed all change which
cutoff is selected, sometimes by a wide margin on a file of a few hundred
or a thousand rows. Decide them before looking at results.

| Setting | Default | What changing it does |
| --- | --- | --- |
| `--rule` | `ltt-bonferroni` | `ltt-holm` accepts slightly more with the same guarantee. `empirical` accepts the most and guarantees nothing. |
| `--design-grid-size` | 10 | More candidates allow a cutoff closer to the target, but each must pass a stricter test level. |
| `--seed` | 42 | A different seed puts different rows in calibration and moves the cutoff. |

**Split check.** `caat policy` refits on the next 20 seeds and reports how
many selected a cutoff and the range of rows they would accept in the whole
file. A wide range means the file is small for the target. The check is a
description of spread and takes no part in the result. Use
`--split-checks 0` to skip it.

**Trying several settings.** The guarantee covers one run whose settings
were fixed first. Running several rules or seeds against the same held-out
rows and reporting the best one is no longer a held-out check. If you
compare settings, pick one for a stated reason and confirm it on new
labeled rows with `caat plan` and `caat evaluate`.

**Which number to report.** "Accepted N of all rows" is the workload: how
many rows the cutoff passes in the whole file. The held-out counts are the
evidence: errors among accepted rows the cutoff never saw. Report both and
say which is which.

## The held-out check

**Observed rate.** Errors divided by accepted rows on the held-out part.

**95% upper bound.** A one-sided exact (Clopper–Pearson) bound on the
accepted error, computed from the held-out counts. Read "22.9%, 95% upper
bound 29.4%" as: the observed rate is 22.9%, and rates above 29.4% are
unlikely given these rows.

**Holdout confidence.** By default the verdict compares the observed rate
with the target. With `--holdout-confidence 0.95`, the verdict compares the
upper bound with the target, which is stricter. The summary names the
number that decided the verdict. When the observed rate meets the target
and the bound is above it, the summary says so: the held-out rows are too
few to rule out a higher rate.

**Two statements at 95%.** The rule's 95% confidence comes from the
calibration rows and covers the cutoff's error on new rows. The 95% upper
bound comes from the held-out rows and describes how high the error could
plausibly be given what was observed there. They are separate checks on
separate rows.

**Certify.** To show, with the stated confidence, that the accepted error
is at or below the target using fresh labels on accepted rows. `caat plan`
gives the number of labels to collect and the largest error count that
still certifies the target.

**Power.** In `caat plan`, the chance that the check certifies the target
when the true error equals the assumed rate. The label count grows quickly
as the assumed rate approaches the target.

**Confirmed.** A cutoff was selected and every grader met the held-out
check. `caat policy` exits with status 0 when the policy is confirmed and
1 when it is not.

## Comparing scores

**AUROC.** The chance that a randomly chosen correct prediction has a
higher score than a randomly chosen wrong one. 0.5 is random ordering and
1.0 is perfect ordering. AUROC measures ranking, so it does not change
under calibration. In the bundled examples, a score with AUROC 0.60
supports a 40% target, and scores at 0.95 and above support 2% to 5%
targets at about 70% coverage. A value well below 0.5 means the score
points the wrong way.

**Risk–coverage curve.** Accepted error at each coverage, from the most
confident rows down to all rows.

**AURC.** The area under the risk–coverage curve. Lower is better.

**Normalized AURC gain.** The score's improvement over random ordering,
divided by the improvement a perfect ordering would give. 0 matches random
ordering and 1 matches a perfect ordering. `caat scores` recommends the
candidate with the largest gain on the design rows.

**Tied.** `caat scores` calls two scores tied when every paired 95%
bootstrap interval for their difference includes zero. Prefer the cheaper
score in that case.

**Control.** A baseline that uses no confidence information: random
ordering, or shortest output first with `--length-column`.

## Probabilities

These apply when the score is meant to be a probability of being correct
(`--probability`, or after `caat calibrate`).

**Calibration.** A score is calibrated when, among predictions scored 0.8,
about 80% are correct. Calibration changes the values of the scores so they
can be read as probabilities. Temperature, Platt, isotonic, and histogram
are the available methods.

**Overconfident, underconfident.** The mean probability is above, or below,
the share that is correct.

**Brier score.** Mean squared difference between the probability and the
0/1 outcome. Lower is better. 0.25 matches always answering 0.5.

**ECE.** Expected calibration error. Rows are grouped into equal-width
probability bins, and ECE is the average gap between mean probability and
accuracy across bins, weighted by bin size. 0 is perfectly calibrated.

**ACE.** Adaptive calibration error. The same gap computed over bins that
hold equal numbers of rows, which is steadier when most scores sit near 0
or 1.

**Reliability diagram.** A plot of accuracy against mean probability per
bin. Points on the diagonal are calibrated.

[Report fields and exit codes](reports.md) · [Score definitions](features.md)
