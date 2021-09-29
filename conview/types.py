from dataclasses import dataclass
from typing import NamedTuple

import nibabel as nib
import numpy as np
import nilearn.image as nili
import matplotlib as mpl
import pygame as pg

@dataclass
class State:
    atlasPath: str
    atlasImg: nib.Nifti1Image
    atlasMat: np.ndarray
    img4DPath: str
    img4D: nib.Nifti1Image
    current3DVol: nib.Nifti1Image
    current3DMat: np.ndarray
    bgImg: nib.Nifti1Image
    mni_coords_L: tuple[int]
    mni_coords_R: tuple[int]
    index_coords_L: tuple[int]
    index_coords_R: tuple[int]
    volumeIdx: int
    cmap_L: str
    cmap_R: str
    vmin_L: float
    vmax_L: float
    normalizer_L: mpl.colors.Normalize
    vmin_R: float
    vmax_R: float
    normalizer_R: mpl.colors.Normalize
    threshold_L: float
    threshold_R: float
    smooth_L: bool
    smooth_R: bool
    last_active_side: str
    show_info: bool


class ImgCoords(NamedTuple):
    x: int
    y: int


class MatCoords(NamedTuple):
    row: int
    col: int


class ImgSize(NamedTuple):
    w: int
    h: int

    @staticmethod
    def from_(x):
        if type(x) == pg.Surface:
            return ImgSize(x.get_width(), x.get_height())
        else:
            raise ValueError(f"invalid argument: {x}")


class MatShape(NamedTuple):
    rows: int
    cols: int


class SliceClick(NamedTuple):
    index: int
    coords: ImgCoords
    rect: pg.Rect
    side_identifier: str