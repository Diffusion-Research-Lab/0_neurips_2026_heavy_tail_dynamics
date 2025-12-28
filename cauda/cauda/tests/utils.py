"""Unittests utiles."""

# Authors: Hamza Cherkaoui

import torch


def _devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs
