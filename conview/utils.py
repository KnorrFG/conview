import numpy as np
import nibabel as nib
from functools import lru_cache
import importlib.resources

@lru_cache(4)
def load_bg_template(name):
    with importlib.resources.path("conview.data", f"{name}.nii.gz") as p:
        return nib.load(str(p))


def mni_2_index(affine: np.ndarray, mni: tuple)-> tuple:
    return tuple(nib.affines.apply_affine(
        np.linalg.inv(affine), mni).astype(int))


def index_2_mni(affine: np.ndarray, indices: tuple)-> tuple:
    return tuple(nib.affines.apply_affine(affine, indices).astype(int))