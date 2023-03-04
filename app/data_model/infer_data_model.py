from typing import List, Dict, Any
from pydantic import BaseModel, validator
from data_management.schema_provider import BinaryClassificationSchema


def get_infer_request_model(schema: BinaryClassificationSchema) -> BaseModel:
    """ Returns Pydantic  model to verify input data for inference using the given schema

    Args:
        schema (BinaryClassificationSchema): schema for the binary classification problem

    Returns:
        Pydantic model: Pydantic  model to verify input data for inference
    """
    class InferenceRequest(BaseModel): 
        instances: List[Dict[str, Any]]

        @validator("instances", pre=True)
        def instances_must_not_be_empty(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Check that the list of instances is not empty."""
            if v is None:
                raise ValueError("'instances' cannot be None")
            if not isinstance(v, list):
                raise TypeError("'instances' must be a list")
            if len(v) == 0:
                raise ValueError("'instances' cannot be an empty list")
            return v

        @validator("instances", each_item=True)
        def has_id_field(cls, v: Dict[str, Any]) -> Dict[str, Any]:
            """ Check that each sample has the expected ID field"""
            if schema.id_field not in v.keys():
                raise ValueError(f"Required ID field '{schema.id_field}' missing in input sample: {v}")
            return v

        @validator("instances", each_item=True)
        def has_all_required_features(cls, v: Dict[str, Any]) -> Dict[str, Any]:
            """ Check that each sample has all the required features"""
            keys_ = v.keys()
            for k in schema.features: 
                if k not in keys_:
                    raise ValueError(f"Required feature '{k}' missing in input sample: {v}")
            return v

        @validator("instances", each_item=True)
        def has_correct_data_types_for_features(cls, v: Dict[str, Any]) -> Dict[str, Any]:
            """ Check that each feature is of correct type"""    
            for f in schema.numeric_features: 
                if not isinstance(v[f], (int, float)) and v[f] is not None:
                    raise TypeError(f"Type error: Data type of feature {f} should be one of [int, float] or value can be None. Given value {v[f]} is of type {type(v[f])}")                    
            for f in schema.categorical_features:  
                if not isinstance(v[f], (str, int, float)) and v[f] is not None:
                    raise TypeError(f"Type error: Data type of feature {f} should be one of [str, int, float] or value can be None. Given value {v[f]} is of type {type(v[f])}")            
            return v
        
    return InferenceRequest


