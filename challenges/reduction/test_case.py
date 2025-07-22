import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 
import torch 
from utils.testcase_base import TestCase, BaseTestCaseManager
from typing import Dict, Any, List


class ReductionTestCaseManager(BaseTestCaseManager):

    def __init__(self):
        # generate testcase 
        super().__init__()
        self.name = "reduction"
        folder = os.path.dirname(os.path.abspath(__file__))
        if "reduction_test_cases.pt" in os.listdir(folder):
            self.load_test_cases(os.path.join(folder, "reduction_test_cases.pt"))

    def generate_test_case(self, input_shape, output_shape = None, metadata = None):

        input_data = torch.rand(*input_shape, device="cuda" if torch.cuda.is_available() else "cpu")
        output_data = torch.sum(input_data).unsqueeze(0) 

        metadata = {
            "N" : input_shape[0]
        }

        test_case = TestCase(
            name=f"ReductionTestCase_{len(self.test_cases) + 1}",
            input=input_data.squeeze(),
            output=output_data,
            metadata=metadata
        )

        return test_case 
    
    def generate_test_cases(self, input_shapes: List[tuple], output_shapes: List[tuple] = None, metadata: Dict[str, Any] = None) -> List[TestCase]:
        if output_shapes is None:
            output_shapes = [(1,) for _ in input_shapes]

        super().generate_test_cases(input_shapes, output_shapes, metadata)


if __name__ == "__main__": 

    test_case_manager = ReductionTestCaseManager()
    input_shapes = [(5,1), (100,1), (1000,1), (1024,1), (2048,1), (4096,1), (8192,1)]
    test_case_manager.generate_test_cases(input_shapes)
    test_case_manager.save_test_cases("reduction_test_cases.pt")





    