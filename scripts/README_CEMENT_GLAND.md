# Cement Gland Detection for Head/Tail Mapping

## Overview

The cement gland is a dark spot on the embryo's chin (ventral head region) that can be used as a **sanity check** for head/tail identification. Even in dim images, it should be visible if brightness is increased enough.

## Quick Start

### 1. Install Dependencies

```bash
# Required
pip install numpy pillow

# Optional but recommended
pip install opencv-python  # Better blob detection
pip install tifffile       # Better TIF support
pip install scipy         # Advanced image processing
```

### 2. Check Cement Gland in Single Image

```bash
python3 scripts/detect_cement_gland.py examples/Cement\ Gland.tif \
    --output data/cement_gland_results \
    --brightness 2.0
```

### 3. Use in Mapping Workflow

```python
from scripts.detect_cement_gland import check_cement_gland_for_mapping
from pathlib import Path

# Check cement gland before/after Parande labeling
image_path = Path("examples/Cement Gland.tif")
result = check_cement_gland_for_mapping(image_path, brightness_factor=2.0)

if result['found']:
    print(f"✓ Cement gland found at {result['location']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    # Use this to validate head/tail labels
else:
    print("⚠ Cement gland not found - manual verification needed")
```

## Integration with Parande Labeling

The cement gland check should be performed **first** as a sanity check:

1. **Before Parande labeling**: Check if cement gland is visible
2. **After Parande labeling**: Validate that head region aligns with cement gland location

### Example Workflow

```python
from scripts.mapping_with_cement_gland_check import map_embryo_with_cement_gland_check
from pathlib import Path

# Your Parande labels (from labeling tool)
parande_labels = {
    'head': (100, 50),  # (x, y) coordinates
    'tail': (200, 150),
    'confidence': 0.95
}

# Map with cement gland check
result = map_embryo_with_cement_gland_check(
    image_path=Path("embryo_image.tif"),
    parande_labels=parande_labels,
    brightness_factor=2.0
)

# Check results
if result['sanity_check_passed']:
    print("✓ Mapping validated by cement gland")
else:
    print("⚠ Warning: Cement gland doesn't align with head - verify labels")
```

## What Gets Detected

The script detects:
- **Location**: Centroid coordinates of cement gland
- **Size**: Area in pixels
- **Shape**: Circularity (cement gland should be roughly circular)
- **Confidence**: Detection confidence score

## Output

### Single Image Analysis

```json
{
  "image_path": "examples/Cement Gland.tif",
  "image_name": "Cement Gland.tif",
  "cement_gland": {
    "found": true,
    "centroid": [150, 75],
    "bbox": [140, 65, 20, 20],
    "area": 314,
    "circularity": 0.85,
    "confidence": 0.92,
    "method": "opencv_contours"
  },
  "image_shape": [300, 400]
}
```

### Mapping Results

```json
{
  "image_path": "embryo.tif",
  "cement_gland_check": {
    "found": true,
    "location": [150, 75],
    "confidence": 0.92,
    "note": "Cement gland detected at (150, 75) (confidence: 0.92)"
  },
  "parande_labels": {
    "head": [145, 70],
    "tail": [200, 150]
  },
  "sanity_check_passed": true,
  "mapping_notes": [
    "✓ Cement gland found: Cement gland detected at (150, 75) (confidence: 0.92)",
    "✓ Sanity check PASSED: Cement gland near head region (distance: 7.1 pixels)"
  ]
}
```

## Parameters

### Brightness Factor
- **Default**: 2.0
- **Range**: 1.0 - 5.0
- **Effect**: Higher values make dark features more visible
- **Recommendation**: Start with 2.0, increase if cement gland not found

### Size Constraints
- **Min size**: 5 pixels (default)
- **Max size**: 50 pixels (default)
- **Adjust**: Based on image resolution and embryo size

### Darkness Threshold
- **Default**: 0.3 (30th percentile)
- **Range**: 0.1 - 0.5
- **Effect**: Lower values detect darker regions

## Notes for Mapping

1. **Check cement gland FIRST** - Before or immediately after Parande labeling
2. **Note when found** - Log cement gland detection in mapping results
3. **Validate alignment** - Ensure head region is near cement gland location
4. **Manual verification** - If cement gland not found, may need manual check

## Troubleshooting

### Cement Gland Not Found

1. **Increase brightness**: Try `--brightness 3.0` or `4.0`
2. **Check image quality**: Ensure image is not too dark or corrupted
3. **Verify stage**: Cement gland may not be visible at all developmental stages
4. **Manual check**: May need to visually inspect image

### False Positives

1. **Adjust size constraints**: Reduce max_size if detecting other dark spots
2. **Increase circularity threshold**: Cement gland should be roughly circular
3. **Check location**: Cement gland should be on ventral (bottom) side of head

## Integration Points

The cement gland check can be integrated at:

1. **Pre-labeling**: Check visibility before Parande labeling
2. **Post-labeling**: Validate Parande labels against cement gland
3. **Batch processing**: Check all images in a directory
4. **Quality control**: Flag images that need manual review

## Files

- `scripts/detect_cement_gland.py` - Main detection script
- `scripts/mapping_with_cement_gland_check.py` - Integration example
- `examples/Cement Gland.tif` - Example image with cement gland
- `examples/Head_tail identification.tif` - Example image for mapping

