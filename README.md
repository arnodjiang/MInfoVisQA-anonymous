# MInfoVisQA

## Dataset

The dataset is included in this repository at [data/benchmark](data/benchmark).

- Evaluation records: [val.candidates.jsonl](data/benchmark/validation_release/val.candidates.jsonl).
- Localized images and standalone rendering scripts: [cases](data/benchmark/cases).

Use `data/benchmark` as the dataset directory. Image paths are relative to the evaluation-record file. No separate download is required.

## Translation

Benchmark translation uses **Google Translate API (Google Cloud Translation Basic v2, NMT)** for visual labels, question/answer templates and source-context prose. The LLM translator is retained only as an optional backend.

```dotenv
TRANSLATION_BACKEND=google
GOOGLE_TRANSLATE_API_KEY=YOUR_GOOGLE_CLOUD_API_KEY
GOOGLE_TRANSLATE_ENDPOINT=YOUR_GOOGLE_CLOUD_TRANSLATION_BASIC_V2_TRANSLATE_ENDPOINT
```

Enable Cloud Translation in your Google Cloud project and configure the Basic v2 translate endpoint locally. Numerical tokens and label references are protected; dictionary keys, paragraph boundaries and table structure are retained. Translation requests use text only. Credentials are not saved in artifacts.

For a JSON tree of text labels and QA templates:

```sh
python -m scripts.translation --input labels_and_qa.json --output local_runs/translation/fr.json --language fr
```

Use `TRANSLATION_BACKEND=llm` or `--translation-backend llm` to opt into model translation with your `OPENAI_*` settings. Google translation never silently falls back to the LLM. Start a new output revision when changing backends or when old caches lack provider identity. Existing dataset images can be evaluated without running translation or configuring Google credentials.

## Evaluation

Use Python 3.11 or later. From the repository root:

```sh
python -m pip install -r requirements.txt
cp .env.example .env
```

Set these fields in your local `.env`:

```dotenv
OPENAI_API_KEY=YOUR_API_KEY
OPENAI_BASE_URL=YOUR_API_BASE_URL
OPENAI_MODEL=YOUR_IMAGE_CAPABLE_MODEL
JUDGE_API_KEY=YOUR_JUDGE_API_KEY
JUDGE_BASE_URL=YOUR_JUDGE_API_BASE_URL
JUDGE_MODEL=YOUR_JUDGE_MODEL
```

The inference endpoint must support the Responses API with image input. The judge uses text-only equivalence adjudication. Blank `JUDGE_*` fields inherit the corresponding `OPENAI_*` settings. Configure endpoints and credentials yourself; keep `.env` local.

Run a smoke test, then resume the same directory for the full evaluation:

```sh
python -m scripts.evaluation.run --dataset data/benchmark --output local_runs/model --smoke --workers 1
python -m scripts.evaluation.run --dataset data/benchmark --output local_runs/model --workers 3 --llm-judge
```

Completed predictions are reused when resuming. Use a new output directory when changing the model or evaluation configuration. Existing dataset images can be evaluated without rendering or font setup.

Predictions and strict-matching scores are saved under `local_runs/model`. Semantic scores are saved under `local_runs/model/llm_judge_text_v2`. LQA uses matching visual and question languages; XQA-EN and XQA-ZH average their respective 23 cross-language configurations. AVG averages all 70 configurations. Exact matches are accepted directly; other completed answers are checked by the configured judge. Failed requests and unconfirmed equivalence count as incorrect.

To score existing predictions or run judging separately:

```sh
python -m scripts.evaluation.score --run local_runs/model
python -m scripts.evaluation.judge --run local_runs/model --workers 2
```

To verify the embedded dataset and file checksums:

```sh
python scripts/verify_release.py
```
