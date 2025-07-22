import numpy as np 
from typing import List, Dict, Any
from dataclasses import dataclass, asdict, field 
import json 
import torch 
from abc import abstractmethod


@dataclass
class TestCase:
    name: str = ""
    input: torch.Tensor = field(default_factory=lambda: np.array([]))
    output: torch.Tensor = field(default_factory=lambda: np.array([]))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"TestCase(input_shape={self.input.shape}, output_shape={self.output.shape}, metadata={self.metadata})"


class BaseTestCaseManager: 

    def __init__(self): 
        self.name = "Base"
        self.test_cases : List[TestCase] = []
    
    def add_test_case(self, test_case: TestCase):
        self.test_cases.append(test_case)

    def get_test_cases(self, idx: int = None) -> List[TestCase]:
        if idx is not None:
            return [self.test_cases[idx]] if 0 <= idx < len(self.test_cases) else []
        return self.test_cases


    def save_test_cases(self, file_path: str):
        
        test_case_data = []

        for test_case in self.test_cases:
            data = asdict(test_case)
            data['input'] = test_case.input.tolist()  
            data['output'] = test_case.output.tolist()
            test_case_data.append(data)
        
        torch.save(test_case_data, file_path)

        json_file_path = file_path.replace('.pt', '.json')
        with open(json_file_path, 'w', encoding = "utf-8") as json_file:
            json.dump(test_case_data, json_file, indent=4)

        print(f"Test cases saved to {file_path} and {json_file_path}")


    def load_test_cases(self, file_path: str): 
        
        test_case_data = torch.load(file_path)

        for data in test_case_data:
            test_case = TestCase(
                name=data['name'],
                input=torch.tensor(data['input']),
                output=torch.tensor(data['output']),
                metadata=data.get('metadata', {})
            )
            self.add_test_case(test_case)


    @abstractmethod
    def generate_test_case(self, input_shape: tuple, output_shape: tuple, metadata: Dict[str, Any] = None) -> TestCase:
        input_data = torch.rand(*input_shape, device = "cuda" if torch.cuda.is_available() else "cpu")
        output_data = torch.rand(*output_shape, device = "cuda" if torch.cuda.is_available() else "cpu")
        metadata = metadata or {}
        
        test_case = TestCase(
            name=f"TestCase_{self.name}_{len(self.test_cases) + 1}",
            input=input_data,
            output=output_data,
            metadata=metadata
        )
        
        self.add_test_case(test_case)
        return test_case
    

    def generate_test_cases(self, input_shapes: List[tuple], output_shapes: List[tuple], metadata: Dict[str, Any] = None) -> List[TestCase]:
        if len(input_shapes) != len(output_shapes):
            raise ValueError("Input shapes and output shapes must have the same length")
        
        for input_shape, output_shape in zip(input_shapes, output_shapes):
            test_case = self.generate_test_case(input_shape, output_shape, metadata)
            self.test_cases.append(test_case)

        
        
        


if __name__ == "__main__":
    manager = BaseTestCaseManager()
    test_case = manager.generate_test_case((3, 3), (1, 3), {"description": "Sample test case"})
    print(test_case)

    manager.save_test_cases("test_cases.pt")

    new_manager = BaseTestCaseManager()
    new_manager.load_test_cases("test_cases.pt")
    print(new_manager.get_test_cases())  
    
    