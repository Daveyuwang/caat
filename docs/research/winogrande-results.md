# A blockwise audit on WinoGrande

The audit requested **90 of the planned 100 labels** on a new WinoGrande batch. The ninth block brought the error count to 28. At least 27 errors were enough to establish that the eligible population exceeded its 20% target. The remaining ten grades could not change that verdict. The original ANLI policy stayed withheld from routing. The draw was recorded at `fcf70d5` and the stopping decision at `34c8aea`. [Frozen sample](../../results/winogrande/sample.csv) · [Stopping record](../../results/winogrande/stop.json)

| Check | Labels inspected | Errors | Result |
| --- | ---: | ---: | --- |
| Blockwise handoff | 90 | 28 | Exceeds at the fixed cap |
| Full seeded draw | 100 | 31 | Exceeds, one-sided 95% lower limit 23.99% |
| Eligible census | 667 | 215 | 32.23% error |

The 90-label decision reports the range of possible final sample counts, 28 to 38 errors. Completing the draw supplied its usual one-sided risk limits, 23.99% lower and 38.83% upper. Of the 821 scored rows, the candidate cutoff qualified 667 and would send 154 to review. [Full draw](../../results/winogrande/full-result.json) · [Eligible census](../../results/winogrande/census.csv)

## Label requests and elapsed time

The automated grader received nine nonoverlapping CSV requests of ten outputs each. It returned 90 labels in nine calls. The runner made eleven audit calls including initialization and one repeated receipt. The handoff took 0.28 seconds after grader setup and includes local file I/O and the retry. The full-draw check subsequently accessed ten additional unique labels and the census covered the other 567 eligible rows. The RTX 5080 scored 821 prompts in 52 batches and 22.21 seconds excluding model loading, using 60,713 input tokens. [Issued grades](../../results/winogrande/receipts.csv) · [Handoff timing](../../results/winogrande/stop.json) · [GPU measurements](../../results/winogrande/collection.json)

At the observed census error count, ten-at-a-time grading requires **86.84 labels on average** over uniform random draws. The replay computes the exact hypergeometric probability that the full verdict is still open at each block boundary. It multiplies that probability by the size of the next block and sums over the cap. This expectation describes random sampling on the same 667 outputs.

| Replay at seed 42 | Labels requested in blocks of ten | Full-draw verdict |
| --- | ---: | --- |
| WinoGrande | 90 | Exceeds |
| ANLI R2 | 60 | Exceeds |
| ANLI R1 | 90 | Exceeds |
| BoolQ | 100 | Meets |

Each blockwise verdict matched its 100-label comparator. BoolQ used the whole budget. A team can use bulk export when another round of dispatch costs more than the remaining labels. [0.7 study](fixed-batch-audit.md) · [Block handoff guide](../guide/queue.md#issue-audit-labels-in-blocks)

## Reproduce the check

With the base package installed, run the first command from the checkout root. The second rebuilds the archived source policies and uses the workflow and scorer extras.

```bash
python scripts/study_winogrande.py replay
python scripts/study_winogrande.py archives
```

To repeat GPU collection, use the model environment and software versions in the collection record, with PyArrow and huggingface-hub installed. The validation split had 1,267 rows. Keeping one representative per distinct unordered option pair yielded the 821-row study population. The frozen protocol fixes the original option order, prompt, model revision and transferred 4.75 cutoff. The stages below read the dataset answer column only for the IDs needed at each grading stage. [Protocol](winogrande-protocol.md) · [AI2 dataset](https://huggingface.co/datasets/allenai/winogrande/tree/01e74176c63542e6b0bcb004dcdea22d94fb67b5)

```bash
python scripts/collect_winogrande.py freeze
python scripts/collect_winogrande.py collect
python scripts/study_winogrande.py draw
python scripts/study_winogrande.py grade-blocks
python scripts/study_winogrande.py verify-full
python scripts/study_winogrande.py census
```
