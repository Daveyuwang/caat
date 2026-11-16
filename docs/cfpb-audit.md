# Audit a grader change on a routing export

This walkthrough replays a frozen review policy from the public CFPB ticket
routing project. It uses 750 ModernBERT predictions. The upstream policy
keeps the encoder's output when confidence is at least
`0.7980151891708375` and escalates the rest. The source project selected
that cutoff on a separate split for a 10% escalation budget.
[Upstream project](https://github.com/ryansloan/cfpb-routing/tree/a8b059421bb7eb670f29ce93a77fcd3d14b5c29f) ·
[Predictions](https://github.com/ryansloan/cfpb-routing/blob/a8b059421bb7eb670f29ce93a77fcd3d14b5c29f/results/followup/encoder_eval.csv) ·
[Frozen policy report](https://github.com/ryansloan/cfpb-routing/blob/a8b059421bb7eb670f29ce93a77fcd3d14b5c29f/results/followup/stats.json)

## Reproduce the audit

Run from the CAAT checkout in a Python virtual environment.

```bash
python -m pip install .
python scripts/prepare_cfpb_audit.py
caat audit replay/cfpb/sources/encoder_eval.csv --prediction-filter tier=encoder --id-column complaint_id --score-column confidence --group-column product --output-key-column output_text --grader-file archived=replay/cfpb/archived.csv --grader-file regraded=replay/cfpb/regraded.csv --threshold 0.7980151891708375 --risk-target 0.10
```

Preparation downloads the pinned prediction export and policy report and
writes two grader CSVs. The archived grader uses the published exact-match
outcome. For this walkthrough, the candidate grader parses the same stored
score map and requires a unique highest-scoring label matching the same
reference. The candidate is a local rubric change for the audit. The 10%
accepted-error limit is an example CI target. The upstream 10% escalation
budget determines the threshold.

The prediction export has four tiers with overlapping complaint IDs. The
filter scopes the audit to the final encoder tier before CAAT checks ID
uniqueness. The two grader exports use the raw `output_text` value as the
output key. Separate grader files must cover exactly the selected IDs and
carry the same key for each. Their row order can differ. CAAT reads the
original prediction CSV directly.

## Results

| Measure | Archived grader | Candidate grader |
| --- | ---: | ---: |
| Accepted | 671 / 750 | 671 / 750 |
| Errors among accepted | 62 | 62 |
| Accepted error rate | 9.24% | 9.24% |
| Meets the example 10% target | Yes | Yes |

The two changed outcomes are complaints `22203333` and `23779143`. Each
has a tie at the top of its saved score map. Both fall below the frozen
cutoff, so the audit shows them as reviewed. Preparation writes
`replay/cfpb/reference.json` with direct counts and IDs. An independent
replay matched the report, including per-product counts, and reproduced
the upstream encoder-to-cheap cascade's 650 correct outcomes out of 750
after fallback. [Upstream fallback predictions](https://github.com/ryansloan/cfpb-routing/blob/a8b059421bb7eb670f29ce93a77fcd3d14b5c29f/results/full_run.csv) ·
[Upstream cutoff semantics](https://github.com/ryansloan/cfpb-routing/blob/a8b059421bb7eb670f29ce93a77fcd3d14b5c29f/src/followup/report.py#L176-L191)

On one Windows machine with Python 3.14, the median over nine runs with
cached inputs was 0.053 s for a purpose-built direct counting script,
0.089 s to produce the two grader exports and reference JSON, and 0.081 s
for the CAAT command. These timings cover execution and exclude writing
the integration. The walkthrough has one preparation command and one
audit command. The prediction input needs no filtered copy, renamed
columns, or manual grader join.

## Run as a CI check

Add these GitHub Actions steps after checkout and installation.

```yaml
- name: Prepare grader exports
  run: python scripts/prepare_cfpb_audit.py
- name: Check the frozen review policy
  id: audit
  shell: bash
  run: |
    status=0
    caat audit replay/cfpb/sources/encoder_eval.csv \
      --prediction-filter tier=encoder --id-column complaint_id \
      --score-column confidence --group-column product --output-key-column output_text \
      --grader-file archived=replay/cfpb/archived.csv --grader-file regraded=replay/cfpb/regraded.csv \
      --threshold 0.7980151891708375 --risk-target 0.10 --output replay/cfpb/audit.json || status=$?
    printf 'status=%s\n' "$status" >> "$GITHUB_OUTPUT"
    exit "$status"
- name: Keep the audit report
  if: ${{ always() && (steps.audit.outputs.status == '0' || steps.audit.outputs.status == '1') }}
  uses: actions/upload-artifact@v7
  with:
    name: caat-audit
    path: replay/cfpb/audit.json
```

The installed CLI passed with a 10% target (exit 0) and reported a target
miss at 9% (exit 1). Replacing one candidate output key with a different
tier's output for the same complaint rejected the input (exit 2) and
identified the grader and complaint. CI keeps the report for the first
two outcomes. [GitHub artifact action](https://github.com/actions/upload-artifact)

For a first run on your own exports, use your existing cutoff and error
target. Record any field conversions you needed and check one discordant
ID against its source output. The audit report and those notes are enough
to identify the next integration change.
