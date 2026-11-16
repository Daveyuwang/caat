# Scores from saved responses

`caat features INPUT.jsonl --output scores.csv` reads one JSON object per
line. Each object needs a unique string `id`, the saved response `text`,
and a nonempty `tokens` array. Each token has `text` and a finite,
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

Repeat `--feature NAME` to choose columns. The review commands accept higher
scores first. Use `shorter_response` when shorter answers should rank ahead
of longer ones. Choose a ranking direction on design data, then freeze it
before selecting a threshold on calibration data.

The CSV supplies the scores and IDs. Join these to grades from the same
saved responses before evaluating or selecting a policy.
