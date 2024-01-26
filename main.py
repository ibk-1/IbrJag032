from typing import Union, List
from fastapi import FastAPI, HTTPException
import nltk
import re
from nltk.stem.snowball import GermanStemmer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib
import json
from pydantic import BaseModel
from typing import List, Union, Dict

app = FastAPI()
nltk.download("punkt")


class TextRequest(BaseModel):
    text: Union[str, List[str]]


class TextResponse(BaseModel):
    prediction: Union[str, List[Dict[str, str]]]


class DataPreProcessing:
    @staticmethod
    def preprocess_german_text(text: str) -> str:
        # Convert to Lowercase
        text = text.lower()

        # Handle German umlauts and special characters
        replacements = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}
        for k, v in replacements.items():
            text = text.replace(k, v)

        # Remove E-Mails
        text = re.sub(r"\S+@\S+", "", text)
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        # Remove punctuation and special characters
        text = re.sub(r"[^\w\s]", "", text)
        # Tokenization
        tokens = nltk.word_tokenize(text, language="german")
        # Stemming
        stemmer = GermanStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in tokens]

        return " ".join(stemmed_tokens)


class NavieBayes:
    def __init__(self) -> None:
        self.model = joblib.load("./Models/German_Text_Classifier_NB_Model.joblib")

    def predict(self, text: str) -> str:
        # Preprocess the text
        processed_text = DataPreProcessing.preprocess_german_text(text)
        prediction = self.model.predict([processed_text])
        return prediction[0]


class SVMClassifier:
    def __init__(self) -> None:
        self.model = joblib.load("./Models/German_Text_Classifier_SVM_Model.joblib")

    def predict(self, text: str) -> str:
        # Preprocess the text
        processed_text = DataPreProcessing.preprocess_german_text(text)
        prediction = self.model.predict([processed_text])
        return prediction[0]


class LSTMNeuralNetwork:
    def __init__(self) -> None:
        self.model = load_model("./Models/German_Text_Classification_LSTM_NN.h5")
        self.encoder = joblib.load("./Models/encoder.joblib")
        self.tokenizer = self.load_tokenizer()

    @staticmethod
    def load_tokenizer():
        with open("tokenizer.json") as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        return tokenizer

    def predict(self, text: str) -> str:
        # Preprocess the text
        processed_text = DataPreProcessing.preprocess_german_text(text)
        sequence = lstm_model.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=93, padding="post")
        prediction = lstm_model.model.predict(padded_sequence)
        # Inverse transform to get the label
        predicted_label = lstm_model.encoder.inverse_transform(prediction)[0][0]
        return predicted_label


# Initialize model instances
nb_model = NavieBayes()
svm_model = SVMClassifier()
lstm_model = LSTMNeuralNetwork()


@app.get("/")
def Home():
    return {"Message": "Use API to predict German Short Phrases.."}


@app.post("/model/nb", response_model=TextResponse)
def get_predicted_class_nb(request: TextRequest):
    """
    This function is a POST endpoint that takes a text input and returns the predicted class using
    a Naive Bayes model.

    If the input is a single text, it returns a single prediction. If the input is a list of texts, it returns a
    dictionary of predictions for each text.

    :param request: The `request` parameter is of type `TextRequest`. It represents the HTTP request
    made to the `/model/nb` endpoint. It contains the text input that needs to be classified
    :type request: TextRequest
    :return: The function `get_predicted_class_nb` returns a `TextResponse` object.
    """
    input_text = request.text
    print(input_text, type(input_text))
    if isinstance(input_text, str):
        # Single text input
        predicted_label = nb_model.predict(input_text)
        print(predicted_label, "NB")
        return TextResponse(prediction=predicted_label)
    else:
        # List of text inputs
        predictions = [nb_model.predict(text) for text in input_text]
        prediction_dict = [{text: pred} for text, pred in zip(input_text, predictions)]
        return TextResponse(prediction=prediction_dict)


@app.post("/model/svm", response_model=TextResponse)
def get_predicted_class_svm(request: TextRequest):
    """
    This function is a POST endpoint that takes a text input and returns the predicted class using
    a Support Vector Machine (SVM) model.

    If the input is a single text, it returns a single prediction. If the input is a list of texts, it returns a
    dictionary of predictions for each text.

    :param request: The `request` parameter is of type `TextRequest`, which is a data model representing
    the request body for the `/model/svm` endpoint. It contains a single attribute `text` which
    represents the input text or a list of input texts
    :type request: TextRequest
    :return: a response object of type `TextResponse`. If the input is a single text, the function
    returns a `TextResponse` object with the predicted label. If the input is a list of texts, the
    function returns a `TextResponse` object with a dictionary containing the texts as keys and their
    corresponding predicted labels as values.
    """
    input_text = request.text
    if isinstance(input_text, str):
        # Single text input
        predicted_label = svm_model.predict(input_text)
        return TextResponse(prediction=predicted_label)
    else:
        # List of text inputs
        predictions = [svm_model.predict(text) for text in input_text]
        prediction_dict = [{text: pred} for text, pred in zip(input_text, predictions)]
        return TextResponse(prediction=prediction_dict)


@app.post("/model/lstm_nn", response_model=TextResponse)
def get_predicted_class_lstm_nn(request: TextRequest):
    """
    This function is a POST endpoint that takes a text input and returns the predicted class using
    a LSTM neural network model.

    If the input is a single text, it returns a single prediction. If the input is a list of texts, it returns a
    dictionary of predictions for each text

    :param request: The `request` parameter is of type `TextRequest`. It represents the incoming request
    to the `/model/lstm_nn` endpoint. It contains the text input that needs to be classified
    :type request: TextRequest
    :return: The function `get_predicted_class_lstm_nn` returns a `TextResponse` object. If the input is
    a single text, the function returns a `TextResponse` object with the predicted label. If the input
    is a list of texts, the function returns a `TextResponse` object with a dictionary of
    text-prediction pairs.
    """
    input_text = request.text
    if isinstance(input_text, str):
        # Single text input
        predicted_label = lstm_model.predict(input_text)
        return TextResponse(prediction=predicted_label)
    else:
        # List of text inputs
        predictions = [lstm_model.predict(text) for text in input_text]
        prediction_dict = [{text: pred} for text, pred in zip(input_text, predictions)]
        return TextResponse(prediction=prediction_dict)
