
import paths
from model_server import ModelServer
from data_management.schema_provider import BinaryClassificationSchema
from data_management.data_reader import read_data


def run_test():
    
    # instantiate schem provider which loads the schema
    data_schema = BinaryClassificationSchema(paths.SCHEMA_FPATH)

    # load test data
    test_data = read_data(data_path=paths.TEST_DATA_FPATH, data_schema=data_schema)

    # load model server
    model_server = ModelServer(model_path=paths.MODEL_ARTIFACTS_PATH, data_schema=data_schema)

    # make predictions - these are predicted class probabilities
    predictions = model_server.predict_proba(data=test_data)

    # save_predictions
    predictions.to_csv(paths.MODEL_PREDICTIONS_PATH, index=False, float_format='%.4f')
    
    print("Test predictions completed successfully")


if __name__ == "__main__": 
    run_test()