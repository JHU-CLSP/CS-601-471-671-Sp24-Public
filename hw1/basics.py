import torch
import torch.nn as nn

torch.manual_seed(42)

"""
Tensors
"""


def tensor_creation():
    # Creating two tensors from lists
    x1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)

    # Just like in NumPy, we can access the shape and datatype of the tensors.
    print(f'shape of x1: {x1.shape}, after transpose: {x1.T.shape}')
    print(f'dtype of x1: {x1.dtype}, after casting to torch.long: {x1.long().dtype}')

    dim1, dim2 = x1.size()
    print("Size:", dim1, dim2)

    # Similar to NumPy you can also create tensors that are all 1s or 0s. Here are examples:
    # creating a tensor of 3 rows and 2 columns consisting of ones
    x = torch.ones(3, 2)
    print('\ntorch.ones(3, 2)')
    print(x)
    # creating a tensor of 3 rows and 2 columns consisting of zeros
    x = torch.zeros(3, 2)
    print('\ntorch.zeros(3, 2)')
    print(x)
    # Creating a tensor with random values
    x = torch.rand(3, 2)
    print('\ntorch.rand(3, 2)')
    print(x)
    # Generating tensor randomly from normal distribution
    x = torch.randn(3, 3)

    print('\ntorch.randn(3, 3)')
    print(x)


"""
Tensor Operations
"""


def tensor_operations():
    # Slicing of Tensors
    print(f"{'-' * 10} Slicing of Tensors {'-' * 10}")
    # You can slice PyTorch tensors the same way you slice `ndarrays` in NumPy
    # create a tensor
    x = torch.tensor([[1, 2],
                      [3, 4],
                      [5, 6]])
    print(x[:, 1])  # Every row, only the last column
    print(x[0, :])  # Every column in first row
    y = x[1, 1]  # take the element in first row and first column and create another tensor
    print(y)

    # Reshape Tensor
    print(f"{'-' * 10} Reshape Tensor {'-' * 10}")
    # A common operation aims at changing the shape of a tensor.
    # A tensor of size (2,3) can be re-organized to any other shape with the same number of elements (e.g. a tensor of size (6), or (3,2), ...).
    # Create a tensor from a (nested) list
    x = torch.Tensor([[1, 2], [3, 4]])
    print("x: ", x)
    x = torch.arange(6)
    print("\nx after .arange(6)")
    print(x)
    x = x.view(2, 3)
    print("\nx after .view(2, 3)")
    print(x)
    x = x.permute(1, 0)  # Swapping dimension 0 and 1
    print("\nx after .permute(1, 0)")
    print(x)

    # Use of `-1` to reshape the tensors.
    print(f"{'-' * 10} Use of `-1` to Reshape the Tensors {'-' * 10}")
    # `-1` indicates that the shape will be inferred from previous dimensions.
    # In the below code snippet `x.view(6,-1)` will result in a tensor of shape `6x1` because we have fixed the size of rows to be 6,
    # Pytorch will now infer the best possible dimension for the column such that it will be able to accommodate all the values present in the tensor.
    x = torch.tensor([[1, 2],
                      [3, 4],
                      [5, 6]])  # (3 rows and 2 columns)
    print(x)
    y = x.view(6, -1)  # y shape will be 6x1
    print(y)


"""
Mathematical Operations
"""


def math_operations():
    # Create two new tensors
    x1 = torch.ones([3, 2])
    x2 = torch.ones([3, 2])

    # Compute the element-wise sum of x1 and x2
    print(f"{'-' * 10} Element-wise Sum {'-' * 10}")
    x_sum = x1 + x2
    print('Addition\n', x_sum)

    # Compute the element-wise multiplication of x1 and x2
    print(f"{'-' * 10} Element-wise Mul {'-' * 10}")
    x_prod = x1 * x2
    print('\nElement-wise multiplication\n', x_prod)

    # Inplace Operations
    # In PyTorch all operations on the tensor that operate in-place on it will have an `_` postfix.
    # For example, `add` is the out-of-place version, and `add_` is the in-place version.
    print(f"{'-' * 10} Inplace Operations {'-' * 10}")
    print(f"How it started: \n {x1} \n {x2}")
    x2.add_(x1)  # tensor y added with x and result will be stored in y
    print(f"How it's going: \n {x1} \n {x2}")


"""
PyTorch and Numpy Bridge
"""
import numpy as np


def torch_numpy():
    print(f"{'-' * 10} PyTorch and Numpy Bridge {'-' * 10}")
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    x = torch.from_numpy(arr)
    arr = x.numpy()
    print(type(x), type(arr))


def run_all_basics_demo():
    tensor_creation()
    tensor_operations()
    math_operations()
    torch_numpy()
