
import paths
from model_server import ModelServer
from data_management.schema_provider import BinaryClassificationSchema
from data_management.data_utils import read_json_in_directory, read_csv_in_directory, save_csv_to_directory


def run_test():
    
    # load the json file schema into a dictionary and use it to instantiate the schema provider
    schema_dict = read_json_in_directory(paths.SCHEMA_DIR)
    data_schema = BinaryClassificationSchema(schema_dict)

    # load test data
    test_data = read_csv_in_directory(file_dir_path=paths.TEST_DIR)

    # load model server
    model_server = ModelServer(model_path=paths.MODEL_ARTIFACTS_PATH, data_schema=data_schema)

    # make predictions - these are predicted class probabilities
    predictions = model_server.predict_proba(data=test_data)

    # save_predictions
    save_csv_to_directory(
        df=predictions, 
        file_dir_path=paths.TEST_OUTPUTS_DIR,
        file_name=paths.PREDICTIONS_FILE_NAME
        )
    
    print("Test predictions completed successfully")


if __name__ == "__main__": 
    run_test()