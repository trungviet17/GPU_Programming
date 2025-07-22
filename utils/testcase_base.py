import numpy as np 
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import json 


class TestCase(BaseModel):
    name: str = Field(default="", description="Name of the test case")
    input: np.ndarray = Field(default=np.ndarray([]), description="Input data for the test case")
    output: np.ndarray = Field(default=np.ndarray([]), description="Expected output data for the test case")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the test case") 


    def __str__(self): 
        return f"TestCase(input_shape={self.input.shape}, output_shape={self.output.shape}, metadata={self.metadata})"
    

class BaseTestCaseManager: 

    def __init__(self): 

        self.test_cases : List[TestCase] = []
    
    def add_test_case(self, test_case: TestCase):
        self.test_cases.append(test_case)

    def get_test_cases(self, idx: int = None) -> List[TestCase]:
        if idx is not None:
            return [self.test_cases[idx]] if 0 <= idx < len(self.test_cases) else []
        return self.test_cases


    def save_test_cases(self, file_path: str):
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([test_case.model_dump() for test_case in self.test_cases], f, ensure_ascii=False, indent=4)

        print(f"Test cases saved to {file_path}")


    def load_test_cases(self, file_path: str): 
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.test_cases = [TestCase(**item) for item in data] if data else []
        
    def generate_test_case(self, input_shape: tuple, output_shape: tuple, metadata: Dict[str, Any] = None) -> TestCase:
        input_data = np.random.rand(*input_shape)
        output_data = np.random.rand(*output_shape)
        metadata = metadata or {}
        
        test_case = TestCase(
            name=f"TestCase_{len(self.test_cases) + 1}",
            input=input_data,
            output=output_data,
            metadata=metadata
        )
        
        self.add_test_case(test_case)
        return test_case


    
    