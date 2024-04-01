#./workload_a_bench.py
#!/usr/bin/env python
# Confidential and Proprietary Information of Hudson River Trading LLC
# checkpy: PYLINT

import argparse
import functools
import time
from typing import Dict, Optional, Tuple

import pandas as pd
import torch

#==========================================================================================
from torch import nn
import os

USE_ROCMLINEAR = os.environ.get('USE_ROCMLINEAR')
if not USE_ROCMLINEAR:
    FC_CLASS_REGISTRY = {
            'torch': nn.Linear
        }
else:
    from custom_linear import customLinear as rocmLinear
    FC_CLASS_REGISTRY = {
            'torch': rocmLinear
        }

#===========================================================================================
# We always have tf32 enabled
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"])
    parser.add_argument("--networks", nargs="+", default=["small", "medium", "large"])
    args = parser.parse_args()

    results = pd.DataFrame(index=args.networks, columns=args.dtypes)
    for dtype in args.dtypes:
        for network in args.networks:
            mintime = test(network, dtype)
            results[dtype][network] = mintime

    print("\n===== RESULTS =====")
    print(results)


def test(
    network: str,
    dtype: str,
    length: int = 120_000,
    width: int = 12,
    n: int = 10,
) -> None:
    torch.manual_seed(17)

    dtype = getattr(torch, dtype)
    parameters = NETWORKS[network].copy()
    batch = parameters.pop("batch_size")
    print('batch_size=', batch)
    model = BaseModel(in_width=width, out_width=1, dtype=dtype, **parameters).cuda()
    inputs = torch.empty((batch, length, width), device="cuda", dtype=dtype).normal_()
    print('inputs=', inputs.size())
    from torchsummary import summary
    summary(model, verbose=2, input_data=(120000, 12), dtypes=[torch.bfloat16, torch.bfloat16])

    autocast = functools.partial(
        torch.autocast,
        device_type="cuda",
        dtype=dtype,
        enabled=dtype in [torch.bfloat16, torch.float16],
    )
    with autocast():
        outputs = model(inputs)
    del outputs

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.001,
    )

    times = [time.time()]
    for _ in range(1):
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            fwd = model(inputs)
            #fwd1= model(inputs)
        #print(fwd.size())
        outputs = fwd.mean().backward()
        #outputs1 = fwd1.mean().backward()
        optimizer.step()
        torch.cuda.synchronize()
        duration = time.time() - start
        print(f"({dtype}, {network}) took {duration}s")
        times.append(duration)

    return min(times)


def elemwise_concat(*tensors: torch.Tensor) -> torch.Tensor:
    """Concat two tensors along their final axis."""
    return torch.cat(tensors, dim=-1)


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        num_blocks: int,
        predecay_width: int,
        block_pass_width: int,
        block_buffer_width: int,
        block_out_width: int,
        buffer_width: int,
        model_width: int,
        in_width: int,
        out_width: int,
        dtype: str,
        relu_leak: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.relu_leak = relu_leak
        self.blocks = torch.nn.ModuleList(
            [
                Block(
                    in_width=in_width if block_num == 0 else block_out_width,
                    decay_in_width=(
                        in_width if block_num == 0 else block_out_width + predecay_width
                    ),
                    predecay_width=predecay_width,
                    block_pass_width=block_pass_width,
                    block_buffer_width=block_buffer_width,
                    block_out_width=block_out_width,
                    relu_leak=self.relu_leak,
                    dtype=dtype,
                )
                for block_num in range(num_blocks)
            ]
        )
        self.final_push = FC_CLASS_REGISTRY["torch"](
            in_features=in_width + block_out_width,
            out_features=buffer_width,
            dtype=dtype,
        )
        self.final_predecay = FC_CLASS_REGISTRY["torch"](
            in_features=in_width + block_out_width + predecay_width,
            out_features=predecay_width,
            dtype=dtype,
        )
        self.final_decay = FC_CLASS_REGISTRY["torch"](
            in_features=predecay_width,
            out_features=buffer_width,
            dtype=dtype,
        )
        self.model = FC_CLASS_REGISTRY["torch"](
            in_features=buffer_width,
            out_features=model_width,
            dtype=dtype,
        )
        self.output = FC_CLASS_REGISTRY["torch"](
            in_features=model_width,
            out_features=out_width,
            dtype=dtype,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        block_in, predecay = inputs, None
        for block in self.blocks:
            block_in, predecay = block(block_in, predecay)

        leaky_relu = functools.partial(
            torch.nn.functional.leaky_relu, negative_slope=self.relu_leak
        )

        decay_in = elemwise_concat(inputs, block_in, predecay)
        predecay: torch.Tensor = leaky_relu(self.final_predecay(decay_in))

        push_in = elemwise_concat(inputs, block_in)
        buffer = self.final_push(push_in) * self.final_decay(predecay)
        model = leaky_relu(self.model(buffer))
        return self.output(model)
class Block(torch.nn.Module):
    def __init__(
        self,
        in_width: int,
        decay_in_width: int,
        predecay_width: int,
        block_pass_width: int,
        block_buffer_width: int,
        block_out_width: int,
        relu_leak: float,
        dtype: str,
    ) -> None:
        super().__init__()
        self.relu_leak = relu_leak
        self.pass_ = FC_CLASS_REGISTRY["torch"](
            in_features=in_width,
            out_features=block_pass_width,
            dtype=dtype,
        )
        self.push = FC_CLASS_REGISTRY["torch"](
            in_features=in_width,
            out_features=block_buffer_width,
            dtype=dtype,
        )
        self.predecay = FC_CLASS_REGISTRY["torch"](
            in_features=decay_in_width,
            out_features=predecay_width,
            dtype=dtype,
        )
        self.decay = FC_CLASS_REGISTRY["torch"](
            in_features=predecay_width,
            out_features=block_buffer_width,
            dtype=dtype,
        )
        self.output = FC_CLASS_REGISTRY["torch"](
            in_features=block_pass_width + block_buffer_width,
            out_features=block_out_width,
            dtype=dtype,
        )

    def forward(
        self, inputs: torch.Tensor, predecay: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        leaky_relu = functools.partial(
            torch.nn.functional.leaky_relu, negative_slope=self.relu_leak
        )
        pass_: torch.Tensor = leaky_relu(self.pass_(inputs))

        if predecay is not None:
            decay_in = elemwise_concat(inputs, predecay)
        else:
            decay_in = inputs
        predecay: torch.Tensor = leaky_relu(self.predecay(decay_in))
        buffer = self.push(inputs) * self.decay(predecay)
        output: torch.Tensor = leaky_relu(self.output(elemwise_concat(pass_, buffer)))
        return output, predecay

NETWORKS: Dict[str, dict] = {
    # 2x
    "small": {
        "batch_size": 60,
        "num_blocks": 2,
        "predecay_width": 8,
        "block_pass_width": 16,
        "block_buffer_width": 32,
        "block_out_width": 16,
        "buffer_width": 16,
        "model_width": 8,
    },
    # 40x
    "medium": {
        "batch_size": 40,
        "num_blocks": 3,
        "predecay_width": 32,
        "block_pass_width": 64,
        "block_buffer_width": 128,
        "block_out_width": 64,
        "buffer_width": 64,
        "model_width": 32,
    },
    # 200x
    "large": {
        "batch_size": 20,
        "num_blocks": 4,
        "predecay_width": 64,
        "block_pass_width": 64,
        "block_buffer_width": 256,
        "block_out_width": 128,
        "buffer_width": 128,
        "model_width": 32,
    },
}

if __name__ == "__main__":
    main()
