#!/usr/bin/env python
# Confidential and Proprietary Information of Hudson River Trading LLC
# checkpy: PYLINT

import argparse
import functools
import time
from typing import Dict, Optional, Tuple

import pandas as pd
import torch

from torch import nn
import os


#x = torch.randn(20, 120000, 1).cuda()
#print(x.size())
#tmp = torch.cat((x, x, x), 0).cuda()
#print(tmp.size())


# import necessary libraries

# define a tensor
#cuda0 = torch.device('cuda:0')
for _ in range(10):
    # A = torch.ones([20, 120000], dtype=torch.bfloat16, device=cuda0)
    # A = torch.tensor(5., requires_grad=True).cuda()

    # define a tensor
    A = torch.tensor(5., requires_grad=True, dtype=torch.bfloat16).cuda()
    print("Tensor-A:", A)

    # define a function using above defined
    # tensor
    x = A**3
    print("x:", x)

    # print the gradient using .grad
    print("A.grad:", A.grad)
