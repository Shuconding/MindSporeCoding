import mindspore.numpy as np
import mindspore.ops.operations as P
from mindspore import dtype as mstype


def calculate_laplacian_with_self_loop(matrix, matmul):
    matrix = matrix + P.Eye()(matrix.shape[0], matrix.shape[0], mstype.float32)
    row_sum = matrix.sum(1)
    d_inv_sqrt = P.Pow()(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_laplacian = matmul(matmul(matrix, d_mat_inv_sqrt).transpose(0, 1), d_mat_inv_sqrt)
    return normalized_laplacian
