#!/usr/bin/env python3
"""
Example integration: Head/tail mapping with cement gland sanity check.

This script demonstrates how to integrate cement gland detection into a mapping workflow.
The cement gland check is performed first as a sanity check before/after Parande labeling.
"""

from pathlib import Path
from typing import Dict, Optional
import json

# Import the cement gland detection function
try:
    from detect_cement_gland import check_cement_gland_for_mapping
    CEMENT_GLAND_AVAILABLE = True
except ImportError:
    CEMENT_GLAND_AVAILABLE = False
    print("Warning: Cement gland detection not available. Install dependencies.")


def map_embryo_with_cement_gland_check(
    image_path: Path,
    parande_labels: Optional[Dict] = None,
    brightness_factor: float = 2.0
) -> Dict:
    """
    Map embryo head/tail with cement gland sanity check.
    
    Args:
        image_path: Path to embryo image
        parande_labels: Optional dict with Parande labeling results
                       Expected keys: 'head', 'tail', 'confidence', etc.
        brightness_factor: Brightness factor for cement gland detection
    
    Returns:
        Dict with mapping results including cement gland check
    """
    result = {
        'image_path': str(image_path),
        'image_name': image_path.name,
        'cement_gland_check': None,
        'parande_labels': parande_labels,
        'mapping_notes': [],
        'sanity_check_passed': False
    }
    
    # Step 1: Check for cement gland first (sanity check)
    if CEMENT_GLAND_AVAILABLE:
        print(f"Checking for cement gland in {image_path.name}...")
        cement_gland_result = check_cement_gland_for_mapping(
            image_path, 
            brightness_factor=brightness_factor
        )
        result['cement_gland_check'] = cement_gland_result
        
        if cement_gland_result['found']:
            result['mapping_notes'].append(
                f"✓ Cement gland found: {cement_gland_result['note']}"
            )
            print(f"  {cement_gland_result['note']}")
        else:
            result['mapping_notes'].append(
                f"⚠ {cement_gland_result['note']}"
            )
            print(f"  {cement_gland_result['note']}")
    else:
        result['mapping_notes'].append(
            "⚠ Cement gland detection not available - skipping sanity check"
        )
        print("  Cement gland detection not available")
    
    # Step 2: Use Parande labels (if provided)
    if parande_labels:
        result['parande_labels'] = parande_labels
        
        # Validate Parande labels against cement gland location
        if (result['cement_gland_check'] and 
            result['cement_gland_check']['found'] and
            'head' in parande_labels):
            
            cg_location = result['cement_gland_check']['location']
            head_location = parande_labels.get('head')
            
            # Check if cement gland is near head region
            if cg_location and head_location:
                # Simple distance check (adjust threshold as needed)
                distance = ((cg_location[0] - head_location[0])**2 + 
                           (cg_location[1] - head_location[1])**2)**0.5
                
                if distance < 100:  # Threshold in pixels
                    result['sanity_check_passed'] = True
                    result['mapping_notes'].append(
                        f"✓ Sanity check PASSED: Cement gland near head region "
                        f"(distance: {distance:.1f} pixels)"
                    )
                    print(f"  ✓ Sanity check PASSED: Cement gland aligns with head")
                else:
                    result['mapping_notes'].append(
                        f"⚠ Sanity check WARNING: Cement gland far from head "
                        f"(distance: {distance:.1f} pixels) - verify labeling"
                    )
                    print(f"  ⚠ Sanity check WARNING: Large distance between "
                          f"cement gland and head - verify Parande labels")
        
        result['mapping_notes'].append(
            f"Parande labels: head={parande_labels.get('head')}, "
            f"tail={parande_labels.get('tail')}"
        )
    else:
        result['mapping_notes'].append(
            "No Parande labels provided - manual labeling needed"
        )
    
    return result


def process_mapping_batch(
    image_dir: Path,
    output_file: Optional[Path] = None,
    brightness_factor: float = 2.0
) -> Dict:
    """
    Process a batch of images for mapping with cement gland checks.
    
    Args:
        image_dir: Directory containing embryo images
        output_file: Optional path to save results JSON
        brightness_factor: Brightness factor for cement gland detection
    
    Returns:
        Dict with batch processing results
    """
    results = {
        'total_images': 0,
        'cement_gland_found': 0,
        'cement_gland_not_found': 0,
        'sanity_checks_passed': 0,
        'sanity_checks_failed': 0,
        'images': []
    }
    
    # Find image files
    image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
    
    results['total_images'] = len(image_files)
    
    for image_path in image_files:
        print(f"\nProcessing: {image_path.name}")
        
        # For now, no Parande labels (would come from your labeling tool)
        mapping_result = map_embryo_with_cement_gland_check(
            image_path,
            parande_labels=None,  # Replace with actual Parande labels
            brightness_factor=brightness_factor
        )
        
        # Update statistics
        if mapping_result['cement_gland_check']:
            if mapping_result['cement_gland_check']['found']:
                results['cement_gland_found'] += 1
            else:
                results['cement_gland_not_found'] += 1
        
        if mapping_result['sanity_check_passed']:
            results['sanity_checks_passed'] += 1
        elif mapping_result['cement_gland_check'] and mapping_result['parande_labels']:
            results['sanity_checks_failed'] += 1
        
        results['images'].append(mapping_result)
    
    # Save results if output file specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Map embryo head/tail with cement gland sanity check'
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
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--parande-labels',
        type=str,
        default=None,
        help='JSON file with Parande labels (optional)'
    )
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    output_file = Path(args.output) if args.output else None
    
    # Load Parande labels if provided
    parande_labels = None
    if args.parande_labels:
        with open(args.parande_labels, 'r') as f:
            parande_labels = json.load(f)
    
    if image_path.is_file():
        print(f"Mapping embryo: {image_path.name}")
        result = map_embryo_with_cement_gland_check(
            image_path,
            parande_labels=parande_labels,
            brightness_factor=args.brightness
        )
        
        print("\n=== Mapping Results ===")
        for note in result['mapping_notes']:
            print(f"  {note}")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {output_file}")
    
    elif image_path.is_dir():
        print(f"Processing directory: {image_path}")
        results = process_mapping_batch(
            image_path,
            output_file,
            brightness_factor=args.brightness
        )
        
        print("\n=== Batch Summary ===")
        print(f"  Total images: {results['total_images']}")
        print(f"  Cement gland found: {results['cement_gland_found']}")
        print(f"  Cement gland not found: {results['cement_gland_not_found']}")
        print(f"  Sanity checks passed: {results['sanity_checks_passed']}")
        print(f"  Sanity checks failed: {results['sanity_checks_failed']}")
    
    else:
        print(f"Error: Path not found: {image_path}")
        exit(1)

