import onnxruntime
import numpy as np
from onnx_helper import SentimentDetectionModel
import argparse
from pathlib import Path
import json
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    help="Path to the input JSON file. Expected to contain a single array of strings",
)
parser.add_argument("--output", type=str, help="Path to the output file")
args = parser.parse_args()
input_path = args.input
output_path = args.output

model = SentimentDetectionModel("distilbert.onnx")

outputs = model.predict_from_json(input_path)
pprint(outputs)
with open(output_path, "w") as f:
    json.dump(outputs, f)
print(f"Predictions saved to {output_path}")
