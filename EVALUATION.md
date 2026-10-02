# Local evaluation configuration

Use Python 3.11 or later. Install requirements.txt, copy .env.example to .env, and configure your own API key, endpoint and image-capable model. No endpoint or credential is supplied. Independent JUDGE_* settings are optional; blank fields inherit OPENAI_* settings. Set JUDGE_MODEL to gpt-6-astra to match the published adjudication model where available.

```sh
python -m pip install -r requirements.txt
cp .env.example .env
python -m scripts.evaluation.run --dataset data/benchmark --output local_runs/model --smoke --workers 1
python -m scripts.evaluation.run --dataset data/benchmark --output local_runs/model --workers 3 --llm-judge
```

The full dataset is embedded as ordinary files, with no remote download or Git LFS dependency. Each localized PNG is referenced relative to the QA file. Frozen rendering scripts embed numerical data and translated labels. Rendering requires a locally configured Unicode font and fallback fonts; pass --font to the standalone renderers and configure MVISQA_FONT and MVISQA_FALLBACK_FONTS as needed. Evaluation of existing PNGs does not require rendering.

The candidate split has 8,960 records. val.jsonl and val.needs_review.jsonl retain the original automated admission status. Automated review is not human certification. All image/question/answer languages and seed identities are preserved. Source datasets are attributed by name in the data and README; upstream distribution terms remain applicable to their content. The software license does not relicense upstream data.

The software retains the benchmark construction and scoring modules. Rebuilding from upstream sources requires locally supplied source files and configuration. Publishing integrations, hosted pages, download shortcuts and development history are omitted.

results/models contains locally available semantic scores, per-example judgments and result rows for the five models in the paper table. results/paired_analysis contains the existing four-model paired analysis, including subgroup estimates and bootstrap intervals. No missing experiment results are synthesized. Aggregate metrics are unchanged by anonymization. Historical input fingerprints refer to the original evaluation records; sanitized files have their own checksums in MANIFEST.sha256.

All 70 configurations contribute equally to AVG. XQA-ZH and XQA-EN each average 23 off-diagonal configurations. Exact matching accepts normalized numeric equality or exact normalized text; other completed answers are adjudicated by the configured text-only judge. Failures and unconfirmed equivalence count as incorrect. Judge identity and inference failures remain in the result files. The main table has five models; the paired analysis has four because only those local paired results were available.

To run offline checks:

```sh
python -m unittest discover -s tests
python scripts/verify_release.py
```
