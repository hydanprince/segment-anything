# Segment Anything (SAM) Python Package

Promptable image segmentation with Meta AI's Segment Anything Model (SAM).

## Installation

### Development install

```bash
# Clone repository
# git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything

# Uninstall any existing versions
python -m pip uninstall -y segment-anything segment_anything

# Install in development mode
python -m pip install -e .
```

### Production install

```bash
# Install from GitHub
pip install "git+https://github.com/facebookresearch/segment-anything.git@main"
```

### Optional extras

```bash
# Mask post-processing, COCO export, notebooks, ONNX export
pip install "segment_anything[all]"
```

## Verify installation

```bash
python -c "import segment_anything as sa; print(sa.__version__)"
```

## Package structure

```
segment_anything/
├── __init__.py
├── __version__.py
├── automatic_mask_generator.py
├── build_sam.py
├── predictor.py
├── utils/
└── modeling/
```

## Quick start

```python
from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry["<model_type>"](checkpoint="/path/to/checkpoint")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)

masks, scores, logits = predictor.predict(<input_prompts>)
```
