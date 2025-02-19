from transformers import AutoTokenizer
import torch
import onnxruntime
import json
import numpy as np


class SentimentDetectionProcessing:
    def __init__(self, max_len=None, tokenizer=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        if tokenizer is None:
            print("No external tokenizer provided, using bhadresh-savani/distilbert-base-uncased-emotion")
            self.tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
        if max_len is None:
            print("No max_len provided, using the model's default max_len")
            self.max_len = self.tokenizer.model_max_length

    def preprocess(self, texts):
        # Tokenize the texts
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return tokens
    
    def load_from_json(self, path): #json file is expected to be a dictionary containing an array
        with open(path, 'r') as f:
            data = json.load(f)
        #get all keys in the dictionary
        keys = list(data.keys())
        #get the first key that is an array
        for key in keys:
            if isinstance(data[key], list):
                data = data[key]
                break
        if not isinstance(data, list):
            raise ValueError("json file does not contain an array")
        return data

    def postprocess(self, predictions, texts):
        # Get the predicted label
        #print(predictions)
        sentiments = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        res = []
        for ind, i in enumerate(predictions):
            #Scores are currently unformalized and possibly negative.
            #Normalize the scores to be between 0 and 1
            scores = torch.nn.functional.softmax(torch.tensor(i), dim=0).numpy()
            #print(scores)
            pred = sentiments[np.argmax(scores)]
            res.append({"text": texts[ind], "sentiment": pred})
        return res

class SentimentDetectionModel:
    def __init__(self, model_path, tokenizer=None):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.model = onnxruntime.InferenceSession(self.model_path, providers=["CUDAExecutionProvider, CPUExecutionProvider"])
        self.processing = SentimentDetectionProcessing(None, tokenizer)

    def predict(self, texts):
        # Preprocess the input data
        input_data = self.processing.preprocess(texts)
        #print(input_data)
        # Run the model
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        #print(input_name, output_name)
        onnx_inputs = {name: input_data[name].numpy() for name in input_data}
        #print(onnx_inputs)
        #print(input_data)
        raw_output = self.model.run([output_name], onnx_inputs)[0]
        # Postprocess the raw output
        return self.processing.postprocess(raw_output, texts)

    def predict_from_json(self, path):
        data = self.processing.load_from_json(path)
        return self.predict(data)