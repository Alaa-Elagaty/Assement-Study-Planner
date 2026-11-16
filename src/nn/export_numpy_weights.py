"""Export the trained PyTorch weights to a plain .npz file (no torch needed).

Run once from the project root:  python src/nn/export_numpy_weights.py
It reads models/study_planner_nn.pth directly (the file is a zip of raw
tensors), so it works even when PyTorch is not installed.
"""
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PTH = ROOT / "models" / "study_planner_nn.pth"
NPZ = ROOT / "models" / "study_planner_nn.npz"

_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64}


class _Storage:
    def __init__(self, dtype):
        self.dtype = dtype


def _rebuild_tensor(storage, offset, size, stride, *args):
    return ("tensor", storage, offset, tuple(size), tuple(stride))


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh, zf, prefix):
        super().__init__(fh)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if name == "OrderedDict":
            return OrderedDict
        if name in _DTYPES:
            return _Storage(_DTYPES[name])
        return super().find_class(module, name)

    def persistent_load(self, pid):
        _, storage_type, key, _loc, _numel = pid
        raw = self.zf.read(f"{self.prefix}/data/{key}")
        return np.frombuffer(raw, dtype=storage_type.dtype)


def load_state_dict(path=PTH):
    with zipfile.ZipFile(path) as zf:
        prefix = zf.namelist()[0].split("/")[0]
        with zf.open(f"{prefix}/data.pkl") as fh:
            raw = _Unpickler(fh, zf, prefix).load()
    state = {}
    for name, (_, flat, offset, size, stride) in raw.items():
        n = int(np.prod(size)) if size else 1
        state[name] = np.array(flat[offset:offset + n]).reshape(size)
    return state


def main():
    state = load_state_dict()
    np.savez(NPZ, **{k.replace(".", "_"): v for k, v in state.items()})
    print("Saved", NPZ, {k: v.shape for k, v in state.items()})


if __name__ == "__main__":
    main()
