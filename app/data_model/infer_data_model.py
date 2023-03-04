

from typing import List
from pydantic import BaseModel, validator


def get_infer_request_model(schema: dict):
    """ Returns appropriate pydantic model to verify input data for inference
        using the given schema

    Args:
        schema (dict): schema for the binary classification problem

    Returns:
        pydantic model: pydantic model to verify input data for inference
    """

    class InferenceRequest(BaseModel): 
        instances: List[dict]

        @validator("instances", pre=True, each_item=True)
        def has_id_field(cls, v): 
            ''' Check that each sample has the expected id field'''
            assert schema.id_field in v.keys(), f"Required ID field '{schema.id_field}' missing in input sample {v}"
            return v

        @validator("instances", pre=True, each_item=True)
        def has_all_required_features(cls, v): 
            ''' Check that each sample has all the required features'''
            keys_ = v.keys()
            for k in schema.features: 
                assert k in keys_, f"Required feature '{k}' missing in input sample {v}"
            return v

        @validator("instances", pre=True, each_item=True)
        def has_correct_data_types_for_features(cls, v): 
            ''' Check that each feature is of correct type'''    
            for f in schema.numeric_features: 
                assert isinstance(v[f], (int, float)) or v[f] is None, f"Type error: Data type of feature {f} should be one of [int, float, NoneType]. Given value {v[f]} is of type {type(v[f])}"
                    
            for f in schema.categorical_features:  
                # we check for str and int or float because the data may be encoded as int or float
                assert isinstance(v[f], (str, int, float)) or v[f] is None, f"Type error: Data type of feature {f} should be one of [str, int, float, NoneType]. Given value {v[f]} is of type {type(v[f])}"               
            
            return v
        
    return InferenceRequest


