#!/usr/bin/env python3
"""
flatten_MARVEL_jsonl.py: Flatten and reformat MARVEL label JSONs into a single JSONL file following the MARVEL_data_transformation schema.
"""

import json
import glob
import os
import argparse

def flatten_json_to_jsonl(input_pattern: str, output_path: str, limit: int = None) -> None:
    """
    Reads all label JSON files, reformats each record according to MARVEL_data_transformation schema,
    and writes each formatted object as a single line in output_path (JSONL).
    """
    files = glob.glob(input_pattern, recursive=True)
    if not files:
        print(f"Warning: No files found matching pattern: {input_pattern}")
        return

    # Sort files by numeric ID extracted from filename
    def extract_id(fp):
        try:
            return int(os.path.basename(fp).split('_')[0])
        except ValueError:
            return float('inf')
    files.sort(key=extract_id)

    # Apply optional limit
    if limit is not None:
        files = files[:limit]
        print(f"Limiting to first {limit} files based on ID")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as out_f:
        for file_path in files:
            # load raw label JSON
            with open(file_path, 'r') as in_f:
                data = json.load(in_f)
            # build image path assuming PNG named by id
            image_dir = os.path.dirname(file_path)
            image_filename = f"{data['id']}.png"
            image_path = os.path.join(image_dir, image_filename)
            # format record
            formatted = {
                "id": str(data["id"]),
                "question": data.get("avr_question"),
                "correct_answer": str(data.get("answer")),
                "image_path": image_path
            }
            out_f.write(json.dumps(formatted))
            out_f.write(os.linesep)

    print(f"Flattened {len(files)} files into {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flatten JSON label files into a JSONL.')
    parser.add_argument('--input-pattern', '-i', default='/mnt/weka/aisg/ob1/raw_datasets/MARVEL_AVR/Json_data/*/*_label.json', help='Glob pattern for input JSON files.')
    parser.add_argument('--output-file', '-o', default='/mnt/weka/aisg/ob1/MM-PRM/formatted_data/MARVEL_AVR/MARVEL_AVR_flattened_full.jsonl', help='Path to output JSONL file.')
    parser.add_argument('--limit', '-n', type=int, default=None, help='Optional max number of files to process sequentially by ID.')
    args = parser.parse_args()
    flatten_json_to_jsonl(args.input_pattern, args.output_file, args.limit) 