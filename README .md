# Automated Curation, Vision Training & LLM Data Auditing
**Augle AI — AI/ML Engineering Intern Assignment**

---

## Overview

An end-to-end intelligent data pipeline that:
1. Automatically collects images and generates COCO-format annotations using Grounding DINO + SAM (Phase 1)
2. Trains a custom CNN (AugleNet / CCNN) on the annotated dataset with TensorBoard tracking (Phase 2)
3. Provides a natural-language RAG auditor backed by a FAISS vector index and Groq-hosted LLaMA-3 with tool-calling (Phase 3)

---

## Repository Structure

```
augle_ai/
├── dataset_creation.py   # Phase 1 — image collection, zero-shot annotation, COCO export
├── train.py              # Phase 2 — custom CNN, augmentation pipeline, training loop
├── llm_auditor.py        # Phase 3 — RAG pipeline, FAISS index, LLM tool-calling
├── requirements.txt      # Pinned dependencies
└── README.md
```

All scripts are designed to run on **Google Colab** with a GPU runtime and Google Drive mounted at `/content/drive/MyDrive/augle_ai`.

---

## Setup

### 1. Mount Drive and set base path

```python
from google.colab import drive
drive.mount('/content/drive')
BASE = '/content/drive/MyDrive/augle_ai'
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or run the install cells at the top of each script — they are self-contained.

### 3. API keys

Store your Groq API key as a Colab secret named `GROQ_API_KEY` (via the 🔑 Secrets panel). `llm_auditor.py` reads it automatically:

```python
from google.colab import userdata
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
```

---

## Phase 1 — Automated Collection & Zero-Shot Annotation (`dataset_creation.py`)

### Image Collection

Images are gathered from two sources:

| Source | Method | Per category | Total |
|---|---|---|---|
| Unsplash | Playwright (headless Chromium) | 100 | 300 |
| Open Images V7 | FiftyOne zoo | 200 | 600 |
| **Total** | | **300** | **900** |

Categories: `person`, `car`, `dog`

The Playwright scraper scrolls Unsplash search results pages programmatically and downloads full-resolution JPEGs. FiftyOne pulls from Open Images V7 using its dataset zoo API.

### Zero-Shot Annotation Pipeline

**Grounding DINO** (SwinT-OGC backbone) performs open-vocabulary object detection given a text prompt (`"person . car . dog ."`), producing bounding boxes with confidence scores.

**Segment Anything Model** (SAM ViT-H) takes the Grounding DINO boxes as prompts and generates per-instance segmentation masks.

```
Image → Grounding DINO (text-prompted) → BBoxes
                                            ↓
                                 SAM (box-prompted) → Masks
                                            ↓
                               COCO JSON annotation
```

### COCO Output

The pipeline outputs a single `instances.json` compliant with the COCO detection format:

```json
{
  "images": [...],
  "annotations": [{"id", "image_id", "category_id", "bbox", "segmentation", "area", "iscrowd", "score"}],
  "categories": [{"id", "name", "supercategory"}]
}
```

Schema compliance is validated before Phase 2 begins.

### OpenCV Preprocessing

All transforms propagate correctly to bounding boxes and masks:

| Transform | Details |
|---|---|
| Resize | `cv2.INTER_LINEAR` for images; `cv2.INTER_NEAREST` for masks; bbox scaled by `(W_new/W_old, H_new/H_old)` |
| Normalize | ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` |
| Horizontal flip | Boxes mirrored as `x_new = W − x − w` |
| Random affine | Rotation ±15°, translate ±10%, scale 0.9–1.1; box corners individually transformed then re-fitted to axis-aligned bbox |
| Color jitter | Random brightness/contrast + HSV saturation jitter |
| Random crop | Retains ≥ 90% of each dimension; drops boxes clipped below 5 px in either dimension |

---

## Phase 2 — Custom PyTorch Training (`train.py`)

### Model Architecture — CCNN (AugleNet)

A custom 4-stage CNN built from scratch. No pre-built backbone is used.

```
Input (3 × 224 × 224)
    │
  Stem  ──  ConvBnReLU(3→32, 3×3)
    │
Stage 1 ── ResidualBlock(32→64,  skip_proj=True,  pool=False)  →  [B, 64, 224, 224]
    │
Stage 2 ── ResidualBlock(64→128, skip_proj=True,  pool=True)   →  [B, 128, 112, 112]
    │
Stage 3 ── ResidualBlock(128→256, skip_proj=True, pool=True)   →  [B, 256,  56,  56]
    │
Stage 4 ── ResidualBlock(256→256, skip_proj=False, pool=True)  →  [B, 256,  28,  28]
    │
  GAP  ──  AdaptiveAvgPool2d(1)                                 →  [B, 256]
    │
Dropout(0.4)
    │
  FC   ──  Linear(256 → num_classes)
    │
LogSoftmax
```

**Architectural decisions:**

- **Stem 3×3 conv** captures fine-grained low-level features before downsampling begins.
- **Residual connections** with 1×1 projection shortcuts (when channel dimensions change) allow gradients to flow unimpeded across all four stages, enabling stable training without batch normalisation collapse.
- **Progressive channel doubling** (32 → 64 → 128 → 256) increases representational capacity with depth, following established CNN design principles.
- **MaxPool only in stages 2–4** preserves spatial resolution in stage 1, beneficial for small-object detection classes like `dog` at variable scale.
- **Global Average Pooling** instead of a flattened FC head reduces parameter count, avoids overfitting on a ~900-image dataset, and makes the model input-size agnostic.
- **Dropout(0.4)** before the classifier head provides additional regularisation for the small dataset.
- **Kaiming initialization** for conv layers; Xavier for the linear head — matched to ReLU activations.

### Dataset Class

`COCOObjectDataset` loads annotations and returns:

```
image   FloatTensor [3, H, W]   ImageNet-normalised
label   LongTensor  []           0-indexed primary category
bboxes  FloatTensor [N, 4]       x, y, w, h pixel coords
masks   BoolTensor  [N, H, W]
```

RLE-encoded segmentation masks are decoded into dense boolean arrays via a custom `rle_to_mask` function. A custom `collate_fn` handles variable annotation counts per image.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Batch size | 16 |
| Image size | 224 × 224 |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealingLR (η_min=1e-6) |
| Loss | NLLLoss (paired with LogSoftmax) |
| Dropout | 0.4 |
| Val split | 15% (seed=42) |

### Experiment Tracking — TensorBoard

The following are logged per epoch:

- `train/step_loss` — per-step training loss
- `loss/train`, `loss/val` — epoch-level loss curves
- `acc/train`, `acc/val` — top-1 accuracy
- `val/top3_acc` — top-3 accuracy
- `lr` — current learning rate
- `augmented_samples` — grid of augmented training images (epoch 1, then every 5 epochs)

Best checkpoint is saved to Drive whenever `val_acc` improves.

```bash
# Launch TensorBoard in Colab
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/augle_ai/runs
```

---

## Phase 3 — Agentic RAG Data Auditor (`llm_auditor.py`)

### Architecture

```
User Query (natural language)
        │
  Embedding (all-MiniLM-L6-v2)
        │
  FAISS IndexFlatIP  ──  top-5 cosine-similar chunks
        │
  Groq LLaMA-3.3-70B (system prompt + retrieved context)
        │
  Tool call decision
     ├── compute_bbox_stats()     ── flags tiny boxes, per-category area stats
     └── compute_class_distribution()  ── balance score, entropy, category counts
        │
  Tool result → back to LLM (agentic loop, max 5 rounds)
        │
  Final grounded answer
```

### RAG Index

Each COCO image is converted to a text chunk describing its annotations:

```
Image 000042.jpg (id=42, 640x480px) contains 3 annotation(s):
  person at (120,85) size 95x210 area=19950px2 score=0.87;
  car at (300,200) size 180x110 area=19800px2 score=0.91; ...
```

A global dataset-summary chunk is prepended. All chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS `IndexFlatIP` (inner-product, exact cosine similarity for L2-normalised vectors). The index is persisted to Drive and reloaded on subsequent runs.

### Tools

**Tool 1 — `compute_bbox_stats`**
Parses all annotations and computes per-category bounding-box area statistics (count, mean, min, max, std). Flags any annotation whose area falls below a configurable pixel threshold as suspicious.

Parameters:
- `category_filter` (optional string) — restrict to one category
- `size_threshold_px` (int, default 500) — area threshold in px²

**Tool 2 — `compute_class_distribution`**
Returns annotation counts per category, a balance score (min_count / max_count, range 0–1), and normalised Shannon entropy. Provides a plain-English imbalance interpretation.

### Demo Queries

**Query 1 — Suspicious bounding boxes**
```
Are there any suspiciously small bounding boxes in the dataset?
Flag anything under 500 square pixels and tell me which categories are affected.
```
*→ Triggers `compute_bbox_stats` with `size_threshold_px=500`.*

**Query 2 — Class balance**
```
Is the dataset class-balanced? Show me the distribution of annotations per category
and give a recommendation on whether I need to collect more data for any class.
```
*→ Triggers `compute_class_distribution`.*

**Query 3 — General health report (RAG-only)**
```
Give me a high-level health report of this dataset: how many images,
annotations, and categories are there, and what does the annotation density look like?
```
*→ Answered directly from retrieved chunks; no tool call needed.*

### Sample Output

```
Query: Is the dataset class-balanced? ...

[Retrieved 5 chunks]
  1. (score=0.821) Dataset dets: 847 images, 2341 annotations, 3 categories...
  2. (score=0.703) Image 000001.jpg (id=1, 640x480px) contains 2 annotation(s)...
  ...

[Tool Call] compute_class_distribution({})
[Tool Result] {
  "total_annotations": 2341,
  "num_categories": 3,
  "per_category": {"person": 1102, "car": 891, "dog": 348},
  "balance_score": 0.3160,
  "normalized_entropy": 0.9421,
  "interpretation": "Moderate imbalance"
}

[Answer]
The dataset has moderate class imbalance. `person` dominates with 1102 annotations
(47%), followed by `car` at 891 (38%) and `dog` at 348 (15%). The balance score
of 0.316 and entropy of 0.942 confirm the skew. Recommendation: collect an
additional ~400–500 dog images to bring the distribution closer to parity.
```

---

## Running End-to-End

```bash
# Phase 1 — collect images and annotate
python dataset_creation.py

# Phase 2 — train the model
python train.py

# Phase 3 — launch the auditor (interactive queries at the bottom of the script)
python llm_auditor.py
```

All three scripts are also available as `.ipynb` notebooks for cell-by-cell execution in Colab.

---

## Requirements

See `requirements.txt` for pinned versions. Key dependencies:

```
torch==2.3.0
torchvision==0.18.0
faiss-cpu
sentence-transformers
groq
opencv-python-headless
playwright
fiftyone
numpy
matplotlib
```
