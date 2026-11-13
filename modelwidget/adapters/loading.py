"""Shared ``from_pretrained`` call for a snapshot already stored on disk.

Transformers 5 takes ``dtype``. ``AutoModel.from_pretrained`` is wrapped, so
its signature is ``**kwargs`` and a parameter-name check cannot see ``dtype``.
"""

from __future__ import annotations

import torch


def from_pretrained(factory, snapshot, *, dtype: torch.dtype, low_cpu_mem_usage: bool):
    return factory.from_pretrained(
        snapshot,
        local_files_only=True,
        low_cpu_mem_usage=low_cpu_mem_usage,
        dtype=dtype,
    )
