import numpy as np
import sys
import nibabel as nib
from functools import lru_cache
import importlib.resources
import itertools as itt
import matplotlib as mpl

import pyparadigm as pp

from .types import (State, ImgSize, ImgCoords, MatShape, MatCoords)

@lru_cache(4)
def load_bg_template(name):
    with importlib.resources.path("conview.data", f"{name}.nii.gz") as p:
        return nib.load(str(p))


def mni_2_index(affine: np.ndarray, mni: tuple)-> tuple:
    return tuple(nib.affines.apply_affine(
        np.linalg.inv(affine), mni).astype(int))


def index_2_mni(affine: np.ndarray, indices: tuple)-> tuple:
    return tuple(nib.affines.apply_affine(affine, indices).astype(int))


loaded_font = None

@lru_cache()
def text(x, align="left"): 
    global loaded_font
    if loaded_font is None:
        loaded_font = pp.Font("dejavusans", size=40)
    
    # for some reason, text in pygame tends to be very ugly. The best option I
    # found to make it look nice is too make it way to big, and then scale it
    # down
    return pp.Surface(scale=1, margin=pp.Margin(left=0.03))(
        pp.Text(x, loaded_font, align=align))


def partition_all(n, iter):
    offset = 0
    while True:
        vals = list(itt.islice(iter, offset, offset + n))
        len_vals = len(vals)
        if len_vals > 0:
            yield vals
        else:
            return
        if len_vals < n:
            return
        offset += n


def error(msg: str):
    print(msg, file=sys.stderr)
    exit(1)


def load_nii_or_error(file: str):
    try:
        return nib.load(file)
    except Exception as e:
        error(str(e))


def threshold_slice(slice, thresh):
    if thresh is None or thresh == 0: 
        return slice
    x = slice.copy()
    mask = np.abs(x) < thresh
    x[mask] = np.nan
    return x


def get_slices(mat, split_coord):
    return (
        mat[split_coord[0], :, :],
        mat[:, split_coord[1], :],
        mat[:, :, split_coord[2]])


def cut(iter, index):
    return (x for i, x in enumerate(iter) if i != index)


def image_coords_2_mat_coords(imgCoord: ImgCoords, imgSize: ImgSize,
                              matSize: MatShape):
    relX = imgCoord.x / imgSize.w
    relY = imgCoord.y / imgSize.h
    # When creating an image from a mat, it is rotated by 90 degree, the coords
    # need to be adjusted for that
    relRow = relX
    relCol = 1 - relY
    row = relRow * matSize.rows
    col = relCol * matSize.cols
    return MatCoords(round(row), round(col))


def mat_coords_2_img_coords(matCoord: MatCoords, matSize: MatShape,
                            imgSize: ImgSize):
    relRow = matCoord.row / matSize.rows
    relCol = matCoord.col / matSize.cols
    # When creating an image from a mat, it is rotated by 90 degree, the coords
    # need to be adjusted for that
    relX = relRow
    relY = 1 - relCol
    x = relX * imgSize.w
    y = relY * imgSize.h
    return ImgCoords(round(x), round(y))

