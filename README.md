# Deep Gradient Compression on Vietnamese Dataset

### This source code is modified to use Vietnamese datasets for DGC

## Note for MACOS user
Macos must install libuv & tensorflow-macos & pkg-config first \
```
brew install libuv pkg-config
pip install tensorflow-macos
```

Then run command `export PKG_CONFIG_PATH=/opt/homebrew/opt/libuv/lib/pkgconfig` \

Then install setuptoools: `pip install setuptools==58.2.01` \

Require g++ version:

```
g++ --version
    Configured with: --prefix=/Library/Developer/CommandLineTools/usr --with-gxx-include-dir=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/4.2.1
    Apple clang version 12.0.5 (clang-1205.0.22.11)
    Target: arm64-apple-darwin20.6.0
    Thread model: posix
    InstalledDir: /Library/Developer/CommandLineTools/usr/bin
```

## Content
- [Prerequisites](#prerequisites)
- [Code](#code)
- [Dataset](#dataset)
- [Training](#training)

## Prerequisites

The code is built with following libraries (see [requirements.txt](requirements.txt)):
- Python >= 3.7
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.5
- [Horovod](https://github.com/horovod/horovod) == 0.28.4
- [numpy](https://github.com/numpy/numpy)
- [tensorboardX](https://github.com/lanpa/tensorboardX) >= 1.2
- [tqdm](https://github.com/tqdm/tqdm)
- [openmpi](https://www.open-mpi.org/software/ompi/) == 5.0.5
- torchvision == 0.12.0
- torchtext == 0.12.0
- torchaudio == 0.11.0
- torch == 1.11.0

## Code

The core code to implement DGC is in [dgc/compression.py](dgc/compression.py) and [dgc/memory.py](dgc/memory.py).

- Gradient Accumulation and Momentum Correction
```python
    mmt = self.momentums[name]
    vec = self.velocities[name]
    if self.nesterov:
        mmt.add_(grad).mul_(self.momentum)
        vec.add_(mmt).add_(grad)
    else:
        mmt.mul_(self.momentum).add_(grad)
        vec.add_(mmt)
    return vec
```

- Sparsification
```python
    importance = tensor.abs()
    # sampling
    sample_start = random.randint(0, sample_stride - 1)
    samples = importance[sample_start::sample_stride]
    # thresholding
    threshold = torch.min(torch.topk(samples, top_k_samples, 0, largest=True, sorted=False)[0])
    mask = torch.ge(importance, threshold)
    indices = mask.nonzero().view(-1)
```

## Dataset
Download dataset at: `https://drive.google.com/drive/folders/138HHR0pFgAdry57rKCYqEutTV31lUhDa?usp=sharing`
All dataset need to be extracted and located in `data` folder. In `data` folder has 3 sub-folders:
- `vietnameseaudio`: contains audios data
- `vietnameseimage`: contains images data
- `vietnamesetext`: contains text data

## Training
We use [Horovod](https://github.com/horovod/horovod) to run distributed training:
- run on a machine with *N* GPUs,
```bash
horovodrun -np N python train.py --configs [config files]
```
e.g., resnet-20 on cifar-10 dataset with 8 GPUs:
```bash
# fp16 values, int32 indices
# warmup coeff: [0.25, 0.063, 0.015, 0.004, 0.001] -> 0.001
horovodrun -np 8 python train.py --configs configs/cifar/resnet20.py \
    configs/dgc/wm5.py configs/dgc/fp16.py configs/dgc/int32.py
```
- run on *K* machines with *N* GPUs each,
```bash
mpirun -np [K*N] -H server0:N,server1:N,...,serverK:N \
    -bind-to none -map-by slot -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -mca pml ob1 \
    -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo \
    python train.py --configs [config files]
```
e.g., resnet-50 on ImageNet dataset with 4 machines with 8 GPUs each,
```bash
# fp32 values, int64 indices, no warmup
mpirun -np 32 -H server0:8,server1:8,server2:8,server3:8 \
    -bind-to none -map-by slot -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH -x PATH -mca pml ob1 \
    -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo \
    python train.py --configs configs/imagenet/resnet50.py \
    configs/dgc/wm0.py
```
For more information on horovodrun, please read horovod documentations.

Here are some reproduce results using **0.1%** compression ratio (*i.e.*, `configs.train.compression.compress_ratio = 0.001`):
| #GPUs | Batch Size | #Sparsified Nodes | ResNet-50 | VGG-16-BN | LR Scheduler |
|:-----:|:----------:|:-----------------:|:---------:|:---------:|:------------:|
| -     | -          | -                 |  [76.2](https://pytorch.org/docs/stable/torchvision/models.html) | [73.4](https://pytorch.org/docs/stable/torchvision/models.html) | - |
| 8     | 256        | 8                 |   76.6    |   74.1    | MultiStep    |
| 16    | 512        | 16                |   76.5    |   73.8    | MultiStep    |
| 32    | 1024       | 32                |   76.3    |   73.3    | MultiStep    |
| 32    | 1024       | 32                |   76.7    |   74.4    | Cosine       |
| 64    | 2048       | 64                |   76.8    |   74.2    | Cosine       |
| 64    | 2048       | 8                 |   76.6    |   73.8    | Cosine       |
| 128   | 4096       | 16                |   76.4    |   73.1    | Cosine       |
| 256   | 8192       | 32                |   75.9    |   71.7    | Cosine       |

## License

This repository is released under the Apache license. See [LICENSE](LICENSE) for additional details.


## Acknowledgement
- Our implementation is modified from [grace](https://github.com/sands-lab/grace) which is an unified framework for all sorts of compressed distributed training algorithms.
