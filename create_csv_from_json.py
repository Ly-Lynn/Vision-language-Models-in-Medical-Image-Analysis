#!/usr/bin/env python3
"""
Script to create CSV file similar to entrep-train-meta.csv format from JSON input.

Input JSON format:
{
    "image1.png": "label1",
    "image2.png": "label2",
    ...
}

Output CSV format:
,image_path,vocal-throat,nose,ear,throat,text
0,path/to/image1.png,0,1,0,0,nose.description
1,path/to/image2.png,1,0,0,0,vocal-throat.description
"""

import json
import csv
import argparse
from pathlib import Path


# Label mapping to columns and text prefix
LABEL_MAPPING = {
    "vocal-throat": {"vocal-throat": 1, "nose": 0, "ear": 0, "throat": 0},
    "vc-open": {"vocal-throat": 1, "nose": 0, "ear": 0, "throat": 0},
    "vc-closed": {"vocal-throat": 1, "nose": 0, "ear": 0, "throat": 0},
    "nose": {"vocal-throat": 0, "nose": 1, "ear": 0, "throat": 0},
    "nose-right": {"vocal-throat": 0, "nose": 1, "ear": 0, "throat": 0},
    "nose-left": {"vocal-throat": 0, "nose": 1, "ear": 0, "throat": 0},
    "ear": {"vocal-throat": 0, "nose": 0, "ear": 1, "throat": 0},
    "ear-right": {"vocal-throat": 0, "nose": 0, "ear": 1, "throat": 0},
    "ear-left": {"vocal-throat": 0, "nose": 0, "ear": 1, "throat": 0},
    "throat": {"vocal-throat": 0, "nose": 0, "ear": 0, "throat": 1},
}

# Default descriptions for each label type
DEFAULT_DESCRIPTIONS = {
    "vocal-throat": "vocal-throat.examination image",
    "vc-open": "vocal-throat.vocal cords open position",
    "vc-closed": "vocal-throat.vocal cords closed position",
    "nose": "nose.nasal cavity examination",
    "nose-right": "nose.right nasal cavity examination",
    "nose-left": "nose.left nasal cavity examination",
    "ear": "ear.ear examination",
    "ear-right": "ear.right ear examination",
    "ear-left": "ear.left ear examination",
    "throat": "throat.throat examination",
}


def load_json(json_path):
    """Load JSON file with image paths and labels."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_label_columns(label):
    """
    Get the column values for a given label.
    Returns a dict with vocal-throat, nose, ear, throat values.
    """
    # Normalize label (lowercase, strip whitespace)
    label_normalized = label.lower().strip()
    
    # Check if label exists in mapping
    if label_normalized in LABEL_MAPPING:
        return LABEL_MAPPING[label_normalized]
    
    # Default: try to match partial labels
    if "vocal" in label_normalized or "vc" in label_normalized:
        return LABEL_MAPPING["vocal-throat"]
    elif "nose" in label_normalized:
        return LABEL_MAPPING["nose"]
    elif "ear" in label_normalized:
        return LABEL_MAPPING["ear"]
    elif "throat" in label_normalized:
        return LABEL_MAPPING["throat"]
    else:
        # Default to nose if unknown
        print(f"Warning: Unknown label '{label}', defaulting to 'nose'")
        return LABEL_MAPPING["nose"]


def get_text_description(label, custom_description=None):
    """
    Get the text description for a given label.
    If custom_description is provided, use it with the appropriate prefix.
    """
    if custom_description:
        # Determine the prefix based on label
        label_normalized = label.lower().strip()
        if "vocal" in label_normalized or "vc" in label_normalized:
            prefix = "vocal-throat"
        elif "nose" in label_normalized:
            prefix = "nose"
        elif "ear" in label_normalized:
            prefix = "ear"
        elif "throat" in label_normalized:
            prefix = "throat"
        else:
            prefix = "nose"
        
        return f"{prefix}.{custom_description}"
    
    # Use default description
    label_normalized = label.lower().strip()
    if label_normalized in DEFAULT_DESCRIPTIONS:
        return DEFAULT_DESCRIPTIONS[label_normalized]
    
    # Fallback
    return f"{label}.examination image"


def create_csv_from_json(json_path, output_csv, image_base_path=None):
    """
    Create CSV file from JSON input.
    
    Args:
        json_path: Path to input JSON file
        output_csv: Path to output CSV file
        image_base_path: Optional base path to prepend to image paths
    """
    # Load JSON data
    data = load_json(json_path)
    
    # Prepare CSV data
    csv_data = []
    
    for idx, (image_path, label) in enumerate(data.items()):
        # Optionally prepend base path
        if image_base_path:
            full_image_path = str(Path(image_base_path) / image_path)
        else:
            full_image_path = image_path
        
        # Get label columns
        label_cols = get_label_columns(label)
        
        # Get text description
        text_description = get_text_description(label)
        
        # Create row
        row = {
            '': idx,  # Index column
            'image_path': full_image_path,
            'vocal-throat': label_cols['vocal-throat'],
            'nose': label_cols['nose'],
            'ear': label_cols['ear'],
            'throat': label_cols['throat'],
            'text': text_description
        }
        
        csv_data.append(row)
    
    # Write CSV
    fieldnames = ['', 'image_path', 'vocal-throat', 'nose', 'ear', 'throat', 'text']
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"✓ Created CSV file: {output_csv}")
    print(f"✓ Total entries: {len(csv_data)}")


def main():
    parser = argparse.ArgumentParser(
        description='Create CSV file from JSON with image paths and labels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example JSON format:
{
    "v13c2f10e-d822-4721-b520-9a426e1a3552.png": "vc-open",
    "078c91ff-9899-436c-854c-4227df8c1229.png": "nose-right",
    "9e566a09-0695-418b-b11e-12e2f3ece4c6.png": "vc-closed"
}

Supported labels:
- vocal-throat, vc-open, vc-closed
- nose, nose-right, nose-left
- ear, ear-right, ear-left
- throat

Example usage:
    python create_csv_from_json.py input.json -o output.csv
    python create_csv_from_json.py input.json -o output.csv -b local_data/entrep/images
        """
    )
    
    parser.add_argument('json_file', help='Input JSON file with image paths and labels')
    parser.add_argument('-o', '--output', default='output.csv', help='Output CSV file (default: output.csv)')
    parser.add_argument('-b', '--base-path', help='Base path to prepend to image paths')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.json_file).exists():
        print(f"Error: Input file '{args.json_file}' not found!")
        return 1
    
    # Create CSV
    try:
        create_csv_from_json(args.json_file, args.output, args.base_path)
        print(f"\n✓ Success! CSV file created at: {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

