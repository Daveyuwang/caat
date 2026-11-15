# Confidence-Aware Annotation with LLMs  
**CAAT** (Confidence-Aware Annotation Toolkit) is a framework for reliable, auditable, semi-automated dataset labeling.

Use agreement, disagreement, and calibrated trust scoring to decide when LLM annotations are safe to keep and when humans should intervene.  

**Interactive Dashboard**: [https://one684proj.onrender.com/](https://one684proj.onrender.com/)  
*Available for research purposes. Text content is not displayed. Not intended for public distribution.*

[Replay the saved predictions on CPU](#offline-evaluation)

---

# Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Trust Scorer](#trust-scorer)
- [Dashboard](#dashboard)
- [Results](#results)
- [Offline Evaluation](#offline-evaluation)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
---

# Overview

Modern LLMs are powerful annotators, yet they remain:

- overconfident  
- inconsistent  
- mismatched with supervised baselines  
- unreliable for sensitive tasks such as fact verification or toxicity detection

This project develops a **trust-aware annotation pipeline** that quantifies when an LLM prediction is likely correct and when it should be reviewed by a human.  
The system turns raw LLM outputs into **calibrated, actionable trust scores** and visualizes them through an interactive dashboard.

---

# System Architecture

![System Architecture](pics/systemarch.png)

A high-level view of the flow from LLM outputs → feature extraction → trust scoring → dashboard visualization.

---

# Key Features

### Deterministic LLM annotation  
- JSON schema enforcement  
- Low-temperature decoding  
- Automatic validation and re-querying  

### Supervised baseline for cross-model validation  
- DeBERTa-v3 fine-tuned for each dataset  
- Provides entropy, margins, and confidence features  

### Trust Scorer  
- Logistic regression + isotonic calibration  
- Uses confidence, disagreement, and text features  
- Outputs actionable trust scores for deferral policies  

### Selective annotation policies  
- High Precision  
- High Coverage  
- Balanced F1  

### Interactive dashboard  
- Confidence regions  
- Hard-case explorer  
- Threshold tuning  
- Dataset-level summaries  

---

## Trust Scorer

![Trust Scorer Diagram](pics/trustscore.png)

Feature groups include:

- LLM confidence  
- baseline probabilities  
- cross-model disagreement  
- text-level complexity indicators  

The scorer outputs a calibrated trust score  
`P(correct annotation | features)`  
used to determine whether to **automate** or **defer**.

---

# Dashboard

![Annotation Explorer](pics/annotation.png)  
![Confidence Regions](pics/confidence.png)  
![Policy Thresholds](pics/policythresh.png)

The dashboard provides:

### **1. Overview Metrics**  
Summaries of accuracy, κ, F1, ECE, and class-level behavior.

### **2. Annotation Explorer**  
Interactive filtering of examples by confidence, agreement, difficulty, and correctness.

### **3. Confidence Regions + Policy Analysis**  
Scatterplots, threshold sliders, and coverage/accuracy tradeoff curves.

---

# Results

![Selective Annotation Performance](pics/coverage_accuracy_combined.png)

Selective annotation highlights:

- IMDb: nearly full automation with minimal accuracy loss  
- Jigsaw: 97.3 percent accepted accuracy at 77 percent coverage  
- FEVER: trust scoring substantially improves acceptance precision  

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>ROC-AUC</th>
      <th>ECE</th>
      <th>Accepted Accuracy</th>
      <th>Coverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>IMDb</td>
      <td>0.894</td>
      <td>0.006</td>
      <td>95.7%</td>
      <td>99.4%</td>
    </tr>
    <tr>
      <td>Jigsaw</td>
      <td>0.974</td>
      <td>0.003</td>
      <td>97.3%</td>
      <td>77.0%</td>
    </tr>
    <tr>
      <td>FEVER</td>
      <td>0.736</td>
      <td>0.040</td>
      <td>71.5%</td>
      <td>55.9%</td>
    </tr>
  </tbody>
</table>

For detailed metrics, predictions, plots, and ablation studies, explore the `results/` directory.

---

# Offline Evaluation

Run the CPU evaluation from the repository root with Python 3.10 or later.

```bash
python -m pip install matplotlib
python scripts/eval_saved_predictions.py
```

The script compares the policies on the same saved predictions for IMDb,
Jigsaw, and FEVER. Python and Matplotlib are enough to run it.

- **Trust scorer** keeps Qwen's label and ranks examples by the saved calibrated
  trust score. The marked point uses the saved balanced policy.
- **Two-model agreement** accepts an example only when Qwen and DeBERTa predict
  the same label. This rule gives a single operating point.
- **DeBERTa-only** uses DeBERTa's own label and its saved probability. The marked
  point gives the highest attainable coverage at or below the trust scorer's
  balanced coverage. The curve also includes accepting every example.

The script writes two files to [`results/offline_eval/`](results/offline_eval/).
Use `--output-dir PATH` to write them elsewhere.

- [`risk_coverage.png`](results/offline_eval/risk_coverage.png) plots the share
  of accepted examples against the error rate among accepted labels. Each panel
  has its own error scale. The curves connect distinct thresholds and keep tied
  scores together.
- [`class_errors.csv`](results/offline_eval/class_errors.csv) reports totals,
  accepted counts, errors, coverage, and accepted error rates overall and by
  gold class. An empty error rate means the policy accepted no examples in that
  group. DeBERTa-only has rows at matched and full coverage.

On Jigsaw, the original trust scorer and DeBERTa-only each accepted 2,311 of
3,000 examples. DeBERTa-only made 14 errors overall, compared with 62 for the
trust scorer. Among the 230 gold-toxic examples, DeBERTa-only accepted 14 and
misclassified all 14. The trust scorer accepted 128 and misclassified 29.
The [Jigsaw sampling record](results/llm_annotations_7b/jigsaw/jigsaw_sample/jigsaw_20k_sampling_report.json)
shows that the course sample included rows from the original train,
development, and test splits.

The [answer-token pilot](results/answer_token_pilot/README.md) uses a local GPU
to rescore 400 saved Jigsaw labels with Qwen2.5-7B-Instruct. It compares the raw
answer-token probability with Qwen's saved self-report and the original trust
scorer using Brier loss and risk–coverage.

---

# Datasets

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Train</th>
      <th>Dev</th>
      <th>Test</th>
      <th>Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>IMDb</td>
      <td>25000</td>
      <td>12500</td>
      <td>12500</td>
      <td>2</td>
    </tr>
    <tr>
      <td>Jigsaw</td>
      <td>1.8M</td>
      <td>97320</td>
      <td>97320</td>
      <td>2</td>
    </tr>
    <tr>
      <td>FEVER</td>
      <td>145449</td>
      <td>9999</td>
      <td>9999</td>
      <td>3</td>
    </tr>
  </tbody>
</table>

For dataset structure, statistics, label mappings, and download information, see `data/README.md`.

---

# Installation

```bash
git clone https://github.com/Daveyuwang/caat.git
cd caat
python -m pip install -r requirements.txt
```

---

# Usage

### Launch dashboard
```
python dashboard/app.py
```

### Generate LLM annotations
```
python scripts/run_llm_annotations.py
```

### Train the baseline model
```
python scripts/train_baseline_models.py
```

### Train trust scorer
```
python scripts/train_trust_scorer.py
```
