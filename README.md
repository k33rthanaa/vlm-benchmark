# VLM Benchmark

A Python project to benchmark vision–language models (CLIP, BLIP, etc.) on tasks like image–text retrieval and VQA.

## Problem Description

Benchmarking VLMs on technical/diagnostic images and text.

Generic VLMs often struggle with specialized domains like medical imaging, industrial diagnostics, or technical diagrams. This benchmark suite allows for systematic evaluation of model performance on such specialized datasets, measuring their ability to:
1.  Retrieve relevant technical documentation or cases given a diagnostic image.
2.  Answer specific questions about anomalies or features in technical imagery.

## Models Compared

The benchmark supports a variety of models via Hugging Face Transformers, including:

*   **CLIP (Contrastive Language-Image Pre-training)**:
    *   `openai/clip-vit-base-patch32` (Default)
    *   Other variants like `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
*   **VQA Models**:
    *   **ViLT (Vision-and-Language Transformer)**: `dandelin/vilt-b32-finetuned-vqa`
    *   **BLIP (Bootstrapping Language-Image Pre-training)**: `Salesforce/blip-vqa-base`

Users can easily plug in other Hugging Face models by specifying the `model_name` in the configuration.

## Metrics

### Image-Text Retrieval
*   **Recall@K (R@K)**: Measures the percentage of queries where the correct document/image is found within the top K results (K=1, 5, 10). High R@1 indicates strong fine-grained understanding.

### Visual Question Answering (VQA)
*   **VQA Accuracy**: Percentage of questions answered correctly. Answers are normalized (lowercased, punctuation removed) before comparison to exact ground truth labels.

## Limitations and Failure Cases

While large pretrained models are powerful, they exhibit notable limitations in this technical domain:

1.  **Domain Shift**: Models trained on internet-scale data (e.g., LAION, COCO) often fail to understand technical jargon or specific diagnostic features (e.g., distinguishing between similar-looking industrial defects).
2.  **OCR Dependency**: Many technical images contain vital text labels. Models without robust OCR capabilities or text-aware pretraining often fail to answer questions requiring text reading.
3.  **Detailed Reasoning**: CLIP-style models are excellent at broad semantic matching but often fail at counting objects or understanding complex spatial relationships (e.g., "is the crack *above* or *below* the valve?").
4.  **Hallucination**: In VQA tasks, models may confidently generate plausible-sounding but factually incorrect diagnostic statements, which is critical in high-stakes environments.

## Project Structure

```
.
├── pyproject.toml       # Dependency management and project metadata
├── README.md            # Project documentation
├── configs/             # Configuration files
├── scripts/
│   └── run_benchmark.py # CLI entry point
├── tests/               # Unit tests
└── src/
    └── vlm_benchmark/
        ├── __init__.py
        ├── data.py      # Dataset loading logic
        ├── evaluate.py  # Evaluation orchestrator
        ├── metrics.py   # Scoring functions
        ├── models.py    # Model wrappers (CLIP, etc.)
        └── tasks.py     # Task definitions
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd vlm_benchmark
   ```

2. Install dependencies:
   ```bash
   pip install .
   ```
   Or for development:
   ```bash
   pip install -e .
   ```

## Usage

Run the benchmark using the provided script:

```bash
python scripts/run_benchmark.py --config configs/default.yaml
```

Or via CLI arguments (legacy):

```bash
python scripts/run_benchmark.py --model_type clip --task retrieval --dataset data/example/annotations.jsonl
```

### Arguments

- `--config`: Path to a YAML configuration file (recommended).
- `--model_type`: Type of model to use (default: `clip`).
- `--model_name`: Specific Hugging Face model name.
- `--task`: Task to evaluate (`retrieval` or `vqa`).
- `--dataset`: Dataset to use (HF name or path to .jsonl).

## Extending

- **Add a Model**: Implement the `VLM` interface in `src/vlm_benchmark/models.py`.
- **Add a Task**: Implement the `Task` interface in `src/vlm_benchmark/tasks.py`.
- **Add Metrics**: Add static methods to `src/vlm_benchmark/metrics.py`.
