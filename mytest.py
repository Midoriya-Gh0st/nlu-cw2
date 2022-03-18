import torch
import numpy as np


# a = np.array([    [[1, 2, 12, 0],  # 方向0
#                    [3, 4, 34, 0],
#                    [3, 4, 34, 0],
#                    [3, 4, 34, 0],
#                    [5, 6, 56, 0]],
#
#                   [[7, 8, 78, 0],  # 方向1
#                    [9, 10, 910, 0],
#                    [9, 10, 910, 0],
#                    [9, 10, 910, 0],
#                    [11, 12, 1112, 0]]])
#
# a = torch.from_numpy(a)  # torch.DoubleTensor
# print(a.size())


def combine_directions(outs):
    s = torch.cat([outs[0: outs.size(0): 2],
                   outs[1: outs.size(0): 2]],
                  dim=2)  # 在第2维度上合并
    print(f"[0]: {outs[0: outs.size(0): 2]}")
    print(f"[1]: {outs[1: outs.size(0): 2]}")

    return s


# print("size 0:", a.size(0))
# x = combine_directions(a)
# print(x.size())
# print(x)

attn_weight = np.array([[2.2286e-03, 2.2955e-03, 1.0147e-03, 9.0388e-04, 1.8968e-03,
                         2.1466e-02, 1.3945e-02, 3.7814e-02, 4.4589e-02, 2.8069e-02,
                         3.5019e-02, 1.6987e-02, 5.0772e-02, 5.2090e-02, 3.4147e-02,
                         8.6123e-02, 1.3572e-01, 1.9611e-01, 2.1003e-01, 2.1200e-02,
                         7.5836e-03, 0.0000e+00]])
attn_weight = torch.from_numpy(attn_weight)
# print(attn_weight.size())
# print(sum(attn_weight[0]))

test = np.array([1, 2, 3])
test = torch.from_numpy(test)
test1 = test.unsqueeze(dim=0)
test2 = test1.squeeze()
# print(test)
# print(test.size())
#
# print(test1)
# print(test1.size())
#
# print(test2)
# print(test2.size())

print(type(test))
print(type(test.data))
test_data = test.data
test_1 = test
print("[test_data]:", test_data)

test_data[0] = 100
print(test_data)
print(test)

print(id(test))
print(id(test_1))
print(id(test_data))


