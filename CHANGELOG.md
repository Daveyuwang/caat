# Changelog

## Unreleased

- Compare an audit with two planned checkpoints to the best fixed-cap block size, using the same batch, scenarios and dispatch cost. [Budget planning](docs/guide/reports.md#plan)

## 1.1.0

- Preview the chances of meeting or exceeding a saved policy's risk target before issuing labels with `caat plan --forecast`. [Budget forecast](docs/guide/reports.md#plan)
- Add `--block-size` to the forecast to compare expected labels and requests for one grader with a bulk export.
- The [UCI Letter Recognition audit](docs/research/letter-audit.md) replays budget selection, block delivery and a full-population check on a new classifier batch.

## 1.0.0

CAAT takes saved classifier or LLM predictions from score comparison and calibration to a review policy. The same frozen policy works with `route`, `evaluate`, `audit` and `monitor`. A withheld policy can be audited on a new batch while routing remains disabled. [Policy workflow](docs/guide/workflow.md) · [Fixed-budget audit](docs/guide/queue.md#draw-an-audit-sample-after-a-policy-misses-its-target)

- Block requests now carry the question, model answer or other fields selected when the plan was created. Retries use the copy in the journal. [Block handoff](docs/guide/queue.md#issue-audit-labels-in-blocks)
- Upgrade in place with `python -m pip install --upgrade caat-annotation`. Existing policy and plan files keep their recorded settings. Resume an existing block audit with the same handoff directory and original inputs. Its already issued requests retain their IDs and four-column format.
- For a quick check, install the base package and evaluate the five ticket predictions. Three meet the cutoff and one of those three is wrong. [Quickstart](README.md#try-caat-on-five-predictions)

The public Git history starts at the 1.0.0 root. To move an existing clone, save local work and clone into a new directory. Update GitHub Actions references to `Daveyuwang/caat@v1.0.0` or pin the commit identified by that tag. Earlier PyPI distributions remain available. [1.x public interface](CONTRIBUTING.md#public-interface) · [Issues](https://github.com/Daveyuwang/caat/issues)
