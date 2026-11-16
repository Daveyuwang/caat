# CAAT roadmap

CAAT runs confidence experiments from saved model outputs. The toolkit
covers score comparison, calibration, evaluation, and review decisions.
The local collection extra can produce those inputs.

| Stage | Open work |
| --- | --- |
| Features | Replay black-box confidence signals and collection cost on short-answer and multiple-choice datasets with Qwen2.5-7B. |
| Calibrators and metrics | Add calibration references and fixtures for score types proposed by users. |
| Rules | Recheck proportional grader shifts on held-out WildBench archives. |
| Adapters | Read a pinned HELM run directory directly. |
| Renderers | Extend the offline report when a replay needs additional fields. |

Start with a saved input, an independent check or hand-calculated
example, and an acceptance case. [Contributor guide](../CONTRIBUTING.md) ·
[Plugin contracts](guide/plugins.md)
