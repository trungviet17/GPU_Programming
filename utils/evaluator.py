import time 
import numpy as np 
import torch 
import sys, os 
import ctypes
import subprocess
from typing import Dict, Any, List 
from abc import abstractmethod

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.testcase_base import TestCase, BaseTestCaseManager


class CUDAEvaluator: 


    def __init__(self, cuda_lib_path: str):
        abs_path = os.path.abspath(cuda_lib_path)
        self.cuda_lib = ctypes.CDLL(abs_path)
        
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Shared library not found: {abs_path}")

        self.cuda_lib.solve.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.cuda_lib.solve.restype = None 
    
    def eval_testcase(self, test_case: TestCase) -> Dict[str, float] : 

        input_data = test_case.input.to("cuda" if torch.cuda.is_available() else "cpu")
        expected_output = test_case.output.to("cuda" if torch.cuda.is_available() else "cpu")
        metadata = test_case.metadata

        cuda_time, cuda_output = self._run_cuda(input_data = input_data, metadata=metadata)
        torch_time, torch_output = self._run_torch(input_data = input_data, metadata=metadata)

        cuda_acc = self._calculate_acc(cuda_output, expected_output)
        torch_acc = self._calculate_acc(torch_output, expected_output)

        speedup = cuda_time / torch_time if torch_time > 0 else float('inf')
        print(f"CUDA Time: {cuda_time:.6f}s, Torch Time: {torch_time:.6f}s, Speedup: {speedup:.2f}x", 
              f"CUDA Accuracy: {cuda_acc:.4f}, Torch Accuracy: {torch_acc:.4f}", 
              f"CUDA Output: {cuda_output}, Torch Output: {torch_output}")
        return {
            'test_case_name': test_case.name,
            'cuda_time': cuda_time,
            'torch_time': torch_time,
            'speedup': speedup,
            'cuda_accuracy': cuda_acc,
            'cuda_output': cuda_output,
            'torch_output': torch_output,
            'torch_accuracy': torch_acc
        }
        

    @abstractmethod
    def _run_cuda(self, input_data: torch.Tensor, metadata: Dict[str, Any] = None) : 
        pass 

    @abstractmethod
    def _run_torch(self, input_data: torch.Tensor, metadata: Dict[str, Any] = None) : 
        pass 
    

    def _calculate_acc(self, output: torch.Tensor, expected: torch.Tensor, atol = 1e-5, rtol = 1e-5) -> float:
        if output.shape != expected.shape:
            raise ValueError(f"Output shape {output.shape} does not match expected shape {expected.shape}")
        
        return torch.isclose(output, expected, atol=atol, rtol=rtol).float().mean().item()

    def eval_all_test_cases(self, test_case_manager: BaseTestCaseManager) -> List: 

        results = []
        for test_case in test_case_manager.get_test_cases():
            if not isinstance(test_case, TestCase):
                raise TypeError(f"Expected TestCase, got {type(test_case)}")
            
            result = self.eval_testcase(test_case)
            results.append(result)
            print(f"Test case: {test_case.name}")

        with open(f"performance_results.txt", "w") as f:
            for result in results:
                f.write(f"{result['test_case_name']}: CUDA Time: {result['cuda_time']:.6f}s, "
                        f"Torch Time: {result['torch_time']:.6f}s, Speedup: {result['speedup']:.2f}x, "
                        f"CUDA Accuracy: {result['cuda_accuracy']:.4f}, Torch Accuracy: {result['torch_accuracy']:.4f}, "
                        f"CUDA Output: {result['cuda_output']}, Torch Output: {result['torch_output']}\n")
        
        print("Performance results saved to performance_results.txt")
        return results 


def compile_cuda_code(cuda_file: str, output_lib: str) : 


    cmd = f"nvcc -O3 --compiler-options '-fPIC' -shared {cuda_file} -o {output_lib}"    

    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Compiled {cuda_file} to {output_lib} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling {cuda_file}: {e}")
        raise RuntimeError(f"Failed to compile CUDA code: {e}")
    




