#!/usr/bin/env python3
"""
Detect cement gland in embryo images as a sanity check for head/tail identification.

The cement gland is a dark spot on the embryo's chin (ventral head region) that
is used to stick to rocks in the wild. Even in dim images, it should be visible
if brightness is increased enough.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import json

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL/Pillow not available. Install with: pip install pillow")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("Warning: tifffile not available. Install with: pip install tifffile")


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """Load an image file, supporting TIF, PNG, JPEG formats."""
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return None
    
    try:
        # Try tifffile first for TIF files (better handling of multi-page TIFs)
        if image_path.suffix.lower() in ['.tif', '.tiff'] and HAS_TIFFFILE:
            img = tifffile.imread(str(image_path))
            # If multi-page, take first page
            if len(img.shape) == 3 and img.shape[0] > 1:
                img = img[0]
            return img
        
        # Fall back to PIL/Pillow
        if HAS_PIL:
            img = Image.open(image_path)
            # Convert to grayscale if needed
            if img.mode != 'L':
                img = img.convert('L')
            return np.array(img)
        
        # Fall back to OpenCV
        if HAS_CV2:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
        
        print(f"Error: Could not load image {image_path}. Install PIL or OpenCV.")
        return None
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def enhance_brightness(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Enhance image brightness to make dark features more visible."""
    # Normalize to 0-1 range
    img_normalized = image.astype(np.float32) / 255.0
    
    # Apply brightness enhancement
    enhanced = np.clip(img_normalized * factor, 0, 1)
    
    # Convert back to 0-255 range
    return (enhanced * 255).astype(np.uint8)


def detect_cement_gland(
    image: np.ndarray,
    brightness_factor: float = 2.0,
    min_size: int = 5,
    max_size: int = 50,
    darkness_threshold: float = 0.3
) -> Optional[Dict]:
    """
    Detect cement gland in embryo image.
    
    The cement gland is a dark spot on the ventral (bottom) side of the head region.
    
    Args:
        image: Grayscale image as numpy array
        brightness_factor: Factor to enhance brightness (higher = brighter)
        min_size: Minimum size of detected region in pixels
        max_size: Maximum size of detected region in pixels
        darkness_threshold: Threshold for dark regions (0-1, lower = darker)
    
    Returns:
        Dictionary with detection results, or None if not found
    """
    if image is None:
        return None
    
    # Enhance brightness to make dark features visible
    enhanced = enhance_brightness(image, brightness_factor)
    
    # Normalize for thresholding
    normalized = enhanced.astype(np.float32) / 255.0
    
    # Find dark regions (cement gland should be darker than surrounding tissue)
    # Use adaptive thresholding or simple threshold
    threshold_value = np.percentile(normalized, darkness_threshold * 100)
    dark_mask = normalized < threshold_value
    
    # Use OpenCV for better blob detection if available
    if HAS_CV2:
        # Convert to uint8 for OpenCV
        enhanced_uint8 = enhanced
        
        # Apply threshold
        _, thresh = cv2.threshold(enhanced_uint8, 
                                  int(threshold_value * 255), 
                                  255, 
                                  cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and find the most likely cement gland
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_size <= area <= max_size:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate circularity (cement gland should be roughly circular)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        candidates.append({
                            'area': area,
                            'centroid': (cx, cy),
                            'bbox': (x, y, w, h),
                            'circularity': circularity,
                            'contour': contour
                        })
        
        # Select best candidate (most circular, appropriate size)
        if candidates:
            # Sort by circularity (higher is better) and size appropriateness
            candidates.sort(key=lambda c: (
                c['circularity'],  # Prefer circular shapes
                -abs(c['area'] - (min_size + max_size) / 2)  # Prefer medium-sized
            ), reverse=True)
            
            best = candidates[0]
            
            return {
                'found': True,
                'centroid': best['centroid'],
                'bbox': best['bbox'],
                'area': best['area'],
                'circularity': best['circularity'],
                'confidence': min(best['circularity'] * 0.5 + 0.5, 1.0),  # Normalize confidence
                'method': 'opencv_contours'
            }
    
    # Fallback: Simple blob detection using NumPy
    # Find connected dark regions
    from scipy import ndimage
    
    try:
        labeled, num_features = ndimage.label(dark_mask)
        
        # Analyze each labeled region
        candidates = []
        for i in range(1, num_features + 1):
            region_mask = labeled == i
            area = np.sum(region_mask)
            
            if min_size <= area <= max_size:
                # Get centroid
                y_coords, x_coords = np.where(region_mask)
                cx = int(np.mean(x_coords))
                cy = int(np.mean(y_coords))
                
                # Get bounding box
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                w = x_max - x_min + 1
                h = y_max - y_min + 1
                
                # Calculate circularity
                perimeter = np.sum(ndimage.binary_erosion(region_mask) != region_mask)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    candidates.append({
                        'area': area,
                        'centroid': (cx, cy),
                        'bbox': (x_min, y_min, w, h),
                        'circularity': circularity
                    })
        
        if candidates:
            # Select best candidate
            candidates.sort(key=lambda c: (
                c['circularity'],
                -abs(c['area'] - (min_size + max_size) / 2)
            ), reverse=True)
            
            best = candidates[0]
            
            return {
                'found': True,
                'centroid': best['centroid'],
                'bbox': best['bbox'],
                'area': best['area'],
                'circularity': best['circularity'],
                'confidence': min(best['circularity'] * 0.5 + 0.5, 1.0),
                'method': 'scipy_labeling'
            }
    except ImportError:
        pass
    
    return {
        'found': False,
        'confidence': 0.0,
        'method': 'none'
    }


def check_cement_gland_for_mapping(
    image_path: Path,
    brightness_factor: float = 2.0
) -> Dict:
    """
    Quick check for cement gland - designed to be called during mapping workflow.
    
    Returns a simple dict with 'found' boolean and 'note' string for logging.
    This is the function to call from your mapping script.
    
    Args:
        image_path: Path to embryo image
        brightness_factor: Brightness enhancement factor
    
    Returns:
        Dict with 'found' (bool), 'note' (str), and optional 'location' (tuple)
    """
    result = {
        'found': False,
        'note': 'Cement gland not detected',
        'location': None,
        'confidence': 0.0
    }
    
    # Load and detect
    image = load_image(image_path)
    if image is None:
        result['note'] = 'Failed to load image for cement gland check'
        return result
    
    detection = detect_cement_gland(image, brightness_factor=brightness_factor)
    
    if detection and detection.get('found'):
        result['found'] = True
        result['location'] = detection.get('centroid')
        result['confidence'] = detection.get('confidence', 0.0)
        result['note'] = (
            f"Cement gland detected at {detection['centroid']} "
            f"(confidence: {detection['confidence']:.2f})"
        )
    else:
        result['note'] = 'Cement gland not found - may need manual verification'
    
    return result


def analyze_embryo_image(
    image_path: Path,
    brightness_factor: float = 2.0,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Analyze an embryo image to detect cement gland.
    
    Args:
        image_path: Path to image file
        brightness_factor: Brightness enhancement factor
        output_dir: Optional directory to save annotated images
    
    Returns:
        Dictionary with analysis results
    """
    result = {
        'image_path': str(image_path),
        'image_name': image_path.name,
        'cement_gland': None,
        'image_shape': None,
        'error': None
    }
    
    # Load image
    image = load_image(image_path)
    if image is None:
        result['error'] = 'Failed to load image'
        return result
    
    result['image_shape'] = image.shape
    
    # Detect cement gland
    detection = detect_cement_gland(image, brightness_factor=brightness_factor)
    
    if detection and detection.get('found'):
        result['cement_gland'] = detection
        
        # Save annotated image if output directory specified
        if output_dir and HAS_CV2:
            output_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = output_dir / f"{image_path.stem}_cement_gland_detected.png"
            
            # Create annotated image
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Draw bounding box
            x, y, w, h = detection['bbox']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw centroid
            cx, cy = detection['centroid']
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            
            # Add label
            label = f"Cement Gland (conf: {detection['confidence']:.2f})"
            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(str(annotated_path), annotated)
            result['annotated_image'] = str(annotated_path)
    else:
        result['cement_gland'] = {'found': False, 'confidence': 0.0}
    
    return result


def process_image_directory(
    image_dir: Path,
    output_dir: Optional[Path] = None,
    brightness_factor: float = 2.0
) -> Dict:
    """
    Process all images in a directory.
    
    Args:
        image_dir: Directory containing images
        output_dir: Optional directory for output
        brightness_factor: Brightness enhancement factor
    
    Returns:
        Dictionary with results for all images
    """
    results = {
        'total_images': 0,
        'cement_gland_found': 0,
        'cement_gland_not_found': 0,
        'errors': 0,
        'images': []
    }
    
    # Find image files
    image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
    
    results['total_images'] = len(image_files)
    
    for image_path in image_files:
        print(f"Processing: {image_path.name}")
        analysis = analyze_embryo_image(image_path, brightness_factor, output_dir)
        
        if analysis.get('error'):
            results['errors'] += 1
        elif analysis.get('cement_gland', {}).get('found'):
            results['cement_gland_found'] += 1
        else:
            results['cement_gland_not_found'] += 1
        
        results['images'].append(analysis)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detect cement gland in embryo images'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to image file or directory'
    )
    parser.add_argument(
        '--brightness',
        type=float,
        default=2.0,
        help='Brightness enhancement factor (default: 2.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for annotated images and results'
    )
    parser.add_argument(
        '--min-size',
        type=int,
        default=5,
        help='Minimum cement gland size in pixels (default: 5)'
    )
    parser.add_argument(
        '--max-size',
        type=int,
        default=50,
        help='Maximum cement gland size in pixels (default: 50)'
    )
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    output_dir = Path(args.output) if args.output else None
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process single image or directory
    if image_path.is_file():
        print(f"Analyzing image: {image_path}")
        result = analyze_embryo_image(image_path, args.brightness, output_dir)
        
        print("\nResults:")
        print(f"  Image: {result['image_name']}")
        print(f"  Shape: {result['image_shape']}")
        
        if result.get('cement_gland', {}).get('found'):
            cg = result['cement_gland']
            print(f"  ✓ Cement gland FOUND")
            print(f"    Centroid: {cg['centroid']}")
            print(f"    Bounding box: {cg['bbox']}")
            print(f"    Area: {cg['area']} pixels")
            print(f"    Circularity: {cg['circularity']:.3f}")
            print(f"    Confidence: {cg['confidence']:.3f}")
        else:
            print(f"  ✗ Cement gland NOT FOUND")
        
        if output_dir:
            # Save results JSON
            json_path = output_dir / f"{image_path.stem}_results.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {json_path}")
    
    elif image_path.is_dir():
        print(f"Processing directory: {image_path}")
        results = process_image_directory(image_path, output_dir, args.brightness)
        
        print("\nSummary:")
        print(f"  Total images: {results['total_images']}")
        print(f"  Cement gland found: {results['cement_gland_found']}")
        print(f"  Cement gland not found: {results['cement_gland_not_found']}")
        print(f"  Errors: {results['errors']}")
        
        if output_dir:
            # Save summary JSON
            json_path = output_dir / 'cement_gland_analysis_summary.json'
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSummary saved to: {json_path}")
    else:
        print(f"Error: Path not found: {image_path}")
        exit(1)

