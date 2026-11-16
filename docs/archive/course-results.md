# Course results

The training and dashboard commands below refer to the archived checkout.
[Course project archive and checkout instructions](course-project.md)

The course pipeline combines Qwen2.5-7B annotations with DeBERTa predictions
and fits a scorer to estimate whether each Qwen label is correct. [System diagram](https://github.com/Daveyuwang/caat/blob/24ca74c32f910b38e18fd6593fc3d6bb5b4f57ab/pics/systemarch.png)

## DeBERTa Predictions

Contains complete test set predictions and calibration metrics for three datasets.

### Prediction Files

Each dataset has a predictions JSON file with the following structure:

- `example_id`: Sequential identifier
- `text`: Input text
- `gold_label`: Ground truth label (integer)
- `predicted_label`: Model prediction (integer)
- `logits`: Raw model outputs (before softmax)
- `probabilities`: Softmax probabilities for each class
- `entropy`: Prediction entropy (uncertainty measure)
- `max_prob`: Maximum probability (confidence score)
- `correct`: Boolean indicating if prediction matches gold label

**Files:**
- `imdb_deberta_predictions.json` (12,500 samples)
- `jigsaw_deberta_predictions.json` (20,000 samples, stratified sample)
- `fever_deberta_predictions.json` (9,999 samples)

### Metrics Files

Each dataset has a metrics JSON file containing:

- Overall performance: accuracy, precision, recall, F1-score
- Calibration metrics: ECE (Expected Calibration Error), Brier score
- Per-class metrics: precision, recall, F1-score, support for each class
- Average entropy and confidence scores

**Files:**
- `imdb_deberta_metrics.json`
- `jigsaw_deberta_metrics.json`
- `fever_deberta_metrics.json`

### Calibration Plots

Contains reliability diagrams and confusion matrices for each dataset:

- Calibration plots show model confidence vs actual accuracy
- Confusion matrices show prediction patterns across classes

**Files:**
- `{dataset}_calibration.png` - Reliability diagrams
- `{dataset}_confusion_matrix.png` - Normalized confusion matrices

### Performance Summary

| Dataset | Accuracy | F1-Score | ECE | Brier Score |
|---------|----------|----------|-----|-------------|
| IMDb    | 95.14%   | 0.9518   | 0.0371 | 0.0430   |
| Jigsaw  | 95.09%   | 0.6772   | 0.0129 | 0.0374   |
| FEVER   | 64.52%   | 0.6449   | 0.0566 | 0.1563   |

Lower ECE and Brier scores indicate better calibration.

---

## LLM Annotations (Qwen2.5-7B-Instruct)

Complete LLM annotation results for three datasets using Qwen2.5-7B-Instruct model.

### Dataset Results

Each dataset directory contains:
- `{dataset}_llm_annotations.json` - Complete annotation results
- `{dataset}_agreement_metrics.json` - Performance metrics
- `{dataset}_hard_cases.json` - Difficult samples for analysis

**Datasets:**
- `imdb/` - Sentiment analysis (12,500 samples, 95.7% accuracy)
- `fever/` - Fact verification (9,999 samples, 53.8% accuracy)  
- `jigsaw/` - Toxicity detection (20,000 samples, 77.6% accuracy)

### Annotation Format

Each annotation includes:
- `row_id`: Example identifier
- `text`: Input text (or preview for hard cases)
- `gold_label`: Ground truth label
- `llm_label`: LLM prediction
- `llm_confidence`: Verbal confidence (low/medium/high)
- `llm_confidence_numeric`: Numeric confidence (0.3/0.6/0.9)
- `llm_rationale`: Explanation (empty in current version)
- `is_valid`: JSON parsing success
- `raw_response`: Raw LLM output

### Jigsaw Stratified Sample

The stratified Jigsaw sample includes 18,115 original training rows, 944
development rows, and 941 test rows. [Sampling record](https://github.com/Daveyuwang/caat/blob/24ca74c32f910b38e18fd6593fc3d6bb5b4f57ab/results/llm_annotations_7b/jigsaw/jigsaw_sample/jigsaw_20k_sampling_report.json)
- Location: `jigsaw/jigsaw_sample/`
- Files:
  - `jigsaw_sample20k.csv` - Full sample data
  - `jigsaw_sample20k_ids.csv` - Sample IDs only
  - `jigsaw_20k_sampling_report.json` - Sampling statistics
  - `jigsaw_20k_sampling_comparison.png` - Label distribution visualization

### Performance Summary

| Dataset | Samples | Accuracy | F1-Score | Cohen's Kappa | Avg Confidence |
|---------|---------|----------|----------|---------------|----------------|
| IMDb    | 12,500  | 95.7%    | 0.957    | 0.915         | 0.873          |
| Jigsaw  | 20,000  | 77.6%    | 0.768    | 0.481         | 0.838          |
| FEVER   | 9,999   | 53.8%    | 0.543    | 0.308         | 0.615          |

### Hard Cases

Hard cases are identified based on LLM-gold label disagreement. These samples are particularly useful for:
- Understanding model failure modes
- Manual review and annotation refinement
- Dashboard visualization and filtering

---

## Trust Scorer

Complete trust score classifier implementation for predicting LLM annotation reliability.

### Model Architecture

- **Primary Model**: Logistic Regression (LR) with isotonic calibration
- **Comparison Model**: Random Forest (RF)
- **Features**: 10 features combining LLM confidence, DeBERTa predictions, and text characteristics

[Scorer diagram](https://github.com/Daveyuwang/caat/blob/24ca74c32f910b38e18fd6593fc3d6bb5b4f57ab/pics/trustscore.png) · [Original coverage and accuracy plots](https://github.com/Daveyuwang/caat/blob/24ca74c32f910b38e18fd6593fc3d6bb5b4f57ab/pics/coverage_accuracy_combined.png)

### Features

The trust scorer uses 10 features to predict annotation reliability:

**Continuous Features (6):**
1. `llm_confidence_numeric` - LLM confidence level (0.3/0.6/0.9)
2. `deberta_p_max` - Maximum DeBERTa probability
3. `deberta_margin` - Difference between top two probabilities
4. `deberta_entropy` - Prediction entropy (uncertainty)
5. `token_count` - Text length in tokens
6. `punctuation_density` - Punctuation ratio

**Binary Features (4):**
1. `llm_deberta_disagree` - LLM and DeBERTa predict different labels
2. `negation_present` - Text contains negation words
3. `confidence_clash_high_low` - LLM high confidence with DeBERTa low confidence
4. `confidence_clash_low_high` - LLM low confidence with DeBERTa high confidence

### Policy Thresholds

Each model provides three decision policies optimized on the development set:

**1. High-Precision Policy**
- Goal: Maximize accepted accuracy
- Constraint: Coverage >= 30%
- Use case: Critical applications requiring high reliability

**2. Balanced Policy**
- Goal: Maximize F1 score
- Constraint: None
- Use case: Most production scenarios

**3. High-Coverage Policy**
- Goal: Maximize coverage
- Constraint: Accepted accuracy >= 80%
- Use case: Large-scale annotation projects

### Performance Summary

**Logistic Regression (Primary Model):**

| Dataset | ROC-AUC | PR-AUC | ECE | Balanced Policy |
|---------|---------|--------|-----|-----------------|
| IMDb    | 89.4%   | 99.2%  | 0.6% | 95.7% acc @ 99.4% cov |
| Jigsaw  | 97.4%   | 98.9%  | 0.3% | 97.3% acc @ 77.0% cov |
| FEVER   | 73.6%   | 73.7%  | 4.0% | 71.5% acc @ 55.9% cov |

**Random Forest (Comparison):**

| Dataset | ROC-AUC | PR-AUC | ECE | Balanced Policy |
|---------|---------|--------|-----|-----------------|
| IMDb    | 94.6%   | 99.5%  | 0.9% | 97.1% acc @ 97.1% cov |
| Jigsaw  | 97.8%   | 99.0%  | 0.5% | 97.3% acc @ 77.1% cov |
| FEVER   | 75.8%   | 76.0%  | 5.6% | 70.4% acc @ 59.3% cov |

**Key Findings:**
- LR provides better calibration quality (lower ECE) than RF
- LR is more stable, especially on multi-class tasks (FEVER)
- Both models achieve similar discriminative performance (AUC)
- Isotonic calibration reduces ECE by 82-97% for LR

### Predictions Format

Each predictions CSV file contains per-example trust scores and decisions:

- `example_id`: Example identifier
- `text`: Input text
- `gold_label`: Ground truth label
- `llm_label`: LLM prediction
- `llm_confidence`: LLM confidence level
- `prob_class{index}`: DeBERTa probability for the corresponding class
- `trust_score_calibrated`: Calibrated trust score [0, 1]
- `policy_high_precision_decision`: High-precision policy decision (1 for accept, 0 for review)
- `policy_balanced_decision`: Balanced policy decision
- `policy_high_coverage_decision`: High-coverage policy decision

Use the CLI to evaluate the saved predictions at a confidence threshold.
[Jigsaw CLI replay](course-project.md#jigsaw-cli-replay)

### Quality Assurance

The trust scorer has been validated through multiple quality checks:

- **Negative Control Test**: Shuffled labels produce AUC ~ 0.50 (no data leakage)
- **Threshold Stability**: Balanced policy thresholds stable across dev set variations
- **Monotonicity**: Coverage-accuracy curves show expected monotonic relationship
- **Calibration**: ECE significantly reduced after isotonic calibration

### Course model export

The `final_export/` directory contains the course model artifacts.
- Pre-trained LR pipelines for all datasets
- Policy threshold configurations
- Comprehensive performance report

The random forest comparisons use saved predictions and metrics. To recover
the trained random forest pipelines, run this from the repo root in a full
clone.

```bash
git restore --source=12d0752a601546f33c1b34570a08680998cbec28 --worktree -- "results/trust_scorer/models/*_rf_pipeline.joblib"
```

---

## Reproducing Results

Run the training and inference commands from the repository root in a
separate environment with the course dependencies.

```bash
python -m pip install -r requirements.txt
```

### DeBERTa Predictions

To train a baseline for one dataset, pass its name.

```bash
python scripts/train_baseline_models.py --dataset imdb
```

The prediction exporter expects trained models in
`outputs/{dataset}/deberta-v3-base/run1/best_model_calibrated/`.

```bash
python scripts/save_deberta_predictions.py
```

### LLM Annotations

```bash
# All datasets
python scripts/run_llm_annotations_7b.py --datasets all

# Specific dataset
python scripts/run_llm_annotations_7b.py --datasets imdb
```

### Trust Scorer

The archived Jigsaw run used scikit-learn 1.7.1. Use the recorded version to
retrain the course scorer, which uses `cv='prefit'` for calibration. [Run config](https://github.com/Daveyuwang/caat/blob/24ca74c32f910b38e18fd6593fc3d6bb5b4f57ab/results/trust_scorer/models/jigsaw_lr_config.json)

```bash
# Train all models
python scripts/train_trust_scorer.py --dataset all --model all

# Train specific dataset and model
python scripts/train_trust_scorer.py --dataset imdb --model lr
```

## Course dashboard

From the repository root, install the dashboard dependencies and start Dash.

```bash
python -m pip install dash dash-bootstrap-components numpy pandas plotly
python -m dashboard.app
```

Open [localhost:8050](http://127.0.0.1:8050/) to browse the saved results.
[Annotation explorer](https://github.com/Daveyuwang/caat/blob/24ca74c32f910b38e18fd6593fc3d6bb5b4f57ab/pics/annotation.png) · [Confidence regions](https://github.com/Daveyuwang/caat/blob/24ca74c32f910b38e18fd6593fc3d6bb5b4f57ab/pics/confidence.png) · [Policy thresholds](https://github.com/Daveyuwang/caat/blob/24ca74c32f910b38e18fd6593fc3d6bb5b4f57ab/pics/policythresh.png)

---

## Model Configuration

### DeBERTa Baseline

- **Model**: microsoft/deberta-v3-base
- **Training**: Fine-tuned on each dataset
- **Calibration**: Temperature scaling
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Epochs**: 3

### LLM Annotations

- **Model**: Qwen2.5-7B-Instruct (7B parameters)
- **Inference**: Local with transformers or vLLM
- **Batch Size**: 32 samples
- **Temperature**: 0.1 (near-deterministic)
- **Max Tokens**: 12-24 (task-dependent)
- **Output Format**: Structured JSON with validation

### Trust Scorer

- **Logistic Regression**:
  - Solver: lbfgs
  - C: 1.0
  - Class Weight: balanced
  - Calibration: Isotonic regression

- **Random Forest**:
  - N Estimators: 400
  - Max Depth: None
  - Min Samples Leaf: 5
  - Max Features: sqrt
  - Class Weight: balanced

- **Training Split**: 70% train / 15% dev / 15% test (stratified)
- **Random Seed**: 42 (all experiments)

---

## Dataset Summary

| Dataset | Source | Task | Classes | Samples | Train | Dev | Test |
|---------|--------|------|---------|---------|-------|-----|------|
| IMDb | Sentiment | Binary | 2 | 12,101 | 8,470 | 1,815 | 1,816 |
| Jigsaw | Toxicity | Binary | 2 | 20,000* | 14,000 | 3,000 | 3,000 |
| FEVER | Fact Check | Multi | 3 | 9,999 | 6,999 | 1,500 | 1,500 |

*Jigsaw uses the stratified 20,000-row course sample. The split columns in this table refer to the trust-scorer splits.
