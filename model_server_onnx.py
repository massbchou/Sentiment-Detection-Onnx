from onnx_helper import SentimentDetectionModel
import warnings
from typing import TypedDict
import csv
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    DirectoryInput,
    FileResponse,
    BatchTextInput,
    InputSchema,
    InputType,
    ResponseBody,
    TaskSchema,
    FileInput,
)
from pathlib import Path
import argparse
import random
import torch

warnings.filterwarnings("ignore")


# Configure UI Elements in RescueBox Desktop
def create_transform_case_task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="input_dataset",
        label="Path to the input JSON file. Expected to contain a single array of strings",
        input_type=InputType.FILE,
    )
    output_schema = InputSchema(
        key="output_file",
        label="Path to the output directory",
        input_type=InputType.DIRECTORY,
    )
    return TaskSchema(inputs=[input_schema, output_schema], parameters=[])


class Inputs(TypedDict):
    input_dataset: FileInput
    output_file: DirectoryInput


class Parameters(TypedDict):
    pass


server = MLServer(__name__)

server.add_app_metadata(
    name="Sentiment Detection Model",
    author="Umass Rescue",
    version="0.1.0",
    info=load_file_as_string("img_info.md"),
)

model = SentimentDetectionModel("distilbert.onnx")


@server.route("/predict", task_schema_func=create_transform_case_task_schema)
def give_prediction(inputs: Inputs, parameters: Parameters) -> ResponseBody:
    input_path = inputs["input_dataset"].path
    out = Path(inputs["output_file"].path)
    out = str(out / (f"predictions_" + str(random.randint(0, 1000)) + ".csv"))

    res_list = model.predict_from_json(input_path)
    with open(out, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "sentiment"])
        writer.writeheader()
        for i in res_list:
            writer.writerow(i)
    return ResponseBody(FileResponse(path=out, file_type="csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Sentiment Detection Model")
    parser = argparse.ArgumentParser(description="Run a server.")
    parser.add_argument(
        "--port", type=int, help="Port number to run the server", default=5000
    )
    args = parser.parse_args()
    print(
        "CUDA is available." if torch.cuda.is_available() else "CUDA is not available."
    )
    server.run(port=args.port)
