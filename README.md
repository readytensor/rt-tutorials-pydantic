## Introduction

This repository demonstrates how to use Pydantic for validation of request data to an inference service for a random forest binary classifier model.

This repository is part of a tutorial series on Ready Tensor, a web platform for AI developers and users.

## Repository Contents

The `app/` folder in the repository contains the following key folders/sub-folders:

- `data_management/` will all files related to handling and preprocessing data.
- `inputs/` contains the input files related to the _titanic_ dataset.
- `outputs/` is a folder to save model artifacts and other result files. Within this folder:
  - `artifacts/` is location to save model artifacts (i.e. the saved model including the trained preprocessing pipeline)
  - `outputs/` is used to contain the predictions or other results files (functionality not included in this repo)

See the following repository for information on the use of the data schema that is provided in the path `./app/inputs/`.

- [https://github.com/readytensor/rt-tutorials-data-schema](https://github.com/readytensor/rt-tutorials-data-schema)

See the following repository for more information on the data proprocessing logic defined in the path `./app/data_management/`.

- [https://github.com/readytensor/rt-tutorials-data-preprocessing](https://github.com/readytensor/rt-tutorials-data-preprocessing)

See the following repository for more information on the **Classifier** class defined in the script called `classifier.py` in the path `./app/model/` and the **ModelServer** class defined in the script called `model_server.py` in the path `./app/`.

- [https://github.com/readytensor/rt-tutorials-oop-ml](https://github.com/readytensor/rt-tutorials-oop-ml)

## Usage

- Create your virtual environment and install dependencies listed in `requirements.txt`.
- Place the following 3 input files in the path `./app/inputs/`:
  - Train data, which must be a CSV file.
  - Test data, which must be a CSV file.
  - The schema file in JSON format. The schema conforms to Ready Tensor specification for the **Binary Classification-Base** category.
- Update the file paths in the `paths.py` file in `./app/`.
- Run the script `train.py` to train the random forest classifier model. This will save the model artifacts, including the preprocessing pipeline and label encoder, in the path `./app/outputs/artifacts/`.
- Run the script `test.py` to run test predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./app/outputs/predictions/`.
- Run the script `serve.py` to start the inference service, which can be queried using the `/ping` and `/infer` endpoints.
- Send a POST request to the endpoint `/infer` using curl. See sample curl command:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "instances": [
    {
      "PassengerId": "879",
      "Pclass": 3,
      "Name": "Laleff, Mr. Kristo",
      "Sex": "male",
      "Age": null,
      "SibSp": 0,
      "Parch": 0,
      "Ticket": "349217",
      "Fare": 7.8958,
      "Cabin": null,
      "Embarked": "S"
    }
  ]
}' http://localhost:8080/infer
```

The key `instances` contains a list of objects, each of which is a sample for which the prediction is requested. The server will respond with a JSON object containing the predicted probabilities for each input record:

```json
{
  "status": "success",
  "message": null,
  "predictions": [
    {
      "id": "879",
      "label": "0",
      "probabilities": {
        "0": 0.9975,
        "1": 0.0025
      }
    }
  ]
}
```

## OpenAPI

Since the service is implemented using FastAPI, we get automatic documentation of the APIs offered by the service. Visit the docs at `http://localhost:8080/docs`.

## Requirements

The code requires Python 3 and the following libraries:

```makefile
fastapi==0.70.0
uvicorn==0.15.0
pydantic==1.8.2
pandas==1.5.5
numpy==1.19.5
scikit-learn==1.0
feature-engine==1.1.1
```

These packages can be installed by running the following command:

```python
pip install -r requirements.txt
```
