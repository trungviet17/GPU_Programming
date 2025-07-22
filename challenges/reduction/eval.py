import sys, os 
sys.path.append( os.path.dirname( os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 

import torch 
import time 
import ctypes 
from utils.testcase_base import TestCase, BaseTestCaseManager
from utils.evaluator import CUDAEvaluator
from challenges.reduction.test_case import ReductionTestCaseManager
from utils.evaluator import compile_cuda_code


class ReductionEvaluator(CUDAEvaluator):

    def __init__(self, cuda_lib_path: str = None): 
        super().__init__(cuda_lib_path)
        self.name = "ReductionEvaluator"
        self.cuda_lib.solve.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


    def _run_cuda(self, input_data: torch.Tensor, metadata = None):

        input_data = input_data.to("cuda" if torch.cuda.is_available() else "cpu")

        input_ptr = input_data.data_ptr()
        output_tensor = torch.zeros(1, dtype=input_data.dtype, device=input_data.device)
        output_ptr = output_tensor.data_ptr()
        

        start_time = time.time()
        self.cuda_lib.solve(ctypes.c_void_p(input_ptr), ctypes.c_void_p(output_ptr), len(input_data))
        torch.cuda.synchronize()
        end_time = time.time()

    
        return end_time - start_time, output_tensor
    

    def _run_torch(self, input_data: torch.Tensor, metadata = None):

        input_data = input_data.to("cuda" if torch.cuda.is_available() else "cpu")
        # warm up 
        _ = torch.sum(input_data)
        torch.cuda.synchronize()


        start_time = time.time()
        output = torch.sum(input_data).unsqueeze(0)
        torch.cuda.synchronize()  
        end_time = time.time()

        return  end_time - start_time, output



if __name__ == "__main__":

    abs_path = os.path.dirname(os.path.abspath(__file__))
    cuda_file = os.path.join(abs_path, "reduction.cu")
    output_lib = os.path.join(abs_path, "reduction.so")

    compile_cuda_code(cuda_file = cuda_file, output_lib = output_lib)
    evaluator = ReductionEvaluator(cuda_lib_path=output_lib)
    test_case_manager = ReductionTestCaseManager()

    test_case_manager.add_test_case(
        TestCase(
            name="SampleTestCase",
            input=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda" if torch.cuda.is_available() else "cpu"),
            output=torch.tensor([36.0], device="cuda" if torch.cuda.is_available() else "cpu"),
            metadata={"N": 8}
        )
    )
    
    evaluator.eval_all_test_cases(test_case_manager)

