# Scores from saved responses

`caat features INPUT.jsonl --output scores.csv` reads one JSON object per
line. Each object needs a unique string `id` and the saved response `text`.
For token scores, supply a nonempty `tokens` array. Each token has `text` and a finite,
nonpositive `logprob`. Log probabilities use the natural logarithm.

```json
{"id":"response-1","text":"Answer: A","tokens":[{"text":"Answer","logprob":-0.4},{"text":":","logprob":-0.2},{"text":" A","logprob":-0.7}],"answer_span":[8,9]}
```

The optional `answer_span` is a half-open range of Python character offsets
in the saved response. CAAT sums the log probabilities of the tokens that
overlap that range. The field is empty when no span is supplied or the
token strings cannot reconstruct the saved text.

| Output column | Definition |
| --- | --- |
| `mean_token_logprob` | Average log probability of the saved tokens. |
| `sequence_logprob` | Sum of the log probabilities. |
| `low_token_logprob` | Mean of the lowest tenth, rounding the token count up. |
| `token_count` | Number of saved tokens. |
| `shorter_response` | Negative token count. |
| `negative_mean_token_logprob` | Negative average log probability. |
| `answer_span_logprob` | Sum over tokens that overlap the supplied answer span. |

Repeat `--feature NAME` to choose columns. With no `--feature` flag CAAT
outputs the seven token and length scores above. A score with missing
inputs leaves a blank CSV cell. The review commands accept higher
scores first. Use `shorter_response` when shorter answers should rank ahead
of longer ones. Choose a ranking direction on design data, then freeze it
before selecting a threshold on calibration data.

Use `--include-output-key` to export a stable key for each saved response. [Separate grader exports](csv-format.md#separate-grader-exports)

## Saved black-box signals

```json
{"id":"response-1","text":"Paris","p_true":0.92,"verbalized_confidence":"85%","samples":["Paris","paris.","Lyon"],"clusters":["paris","paris","lyon"]}
```

| Feature | Inputs and interpretation | Collection cost |
| --- | --- | --- |
| `p_true` | A saved probability that the model assigns to its answer being correct. | One extra call |
| `verbalized_confidence` | A number in [0, 1], a percentage, or a documented word. | One extra call |
| `sample_agreement` | Fraction of samples matching the most frequent exact answer. | N samples |
| `normalized_sample_agreement` | The same fraction after Unicode NFKC normalization, case folding, whitespace normalization, and trimming outer spaces, periods, question marks, and exclamation marks. | N samples |
| `semantic_entropy` | Shannon entropy in nats over supplied meaning clusters. Lower values indicate more confidence. | N samples, plus any external clustering cost |

The word scale maps very low, low, medium, high, and very high to 0.1,
0.25, 0.5, 0.75, and 0.9. The parser accepts an optional
`Confidence:` prefix. Each value is one expression.

Semantic entropy uses counts by default. To weight cluster mass by
sequence likelihood, supply `sample_logprobs` with one natural log
likelihood per cluster entry. Cluster labels are supplied by the caller.
They match the order of `samples` when both are provided.
[Semantic entropy method](https://doi.org/10.1038/s41586-024-07421-0) ·
[Local collection](collect.md)
