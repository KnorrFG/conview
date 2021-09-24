from dataclasses import dataclass
from typing import NamedTuple
import sys

import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import nilearn.image as nili
import numpy as np
from numpy.core.numeric import cross
from pygame.constants import RESIZABLE
import pyparadigm as pp
import pygame as pg
from pyparadigm.eventlistener import EventListener
from pyparadigm.surface_composition import LLItem
import scipy.ndimage as spi
import click
from scipy.ndimage.interpolation import zoom

from . import utils


@dataclass
class State:
    atlasPath: str
    atlasImg: nib.Nifti1Image
    img4DPath: str
    img4D: nib.Nifti1Image
    current3Dvol: nib.Nifti1Image
    bgImg: nib.Nifti1Image
    mni_coords_L: tuple[int]
    mni_coords_R: tuple[int]
    index_coords_L: tuple[int]
    index_coords_R: tuple[int]
    volumeIdx: int
    cmap_L: str
    cmap_R: str
    vrange_L: tuple[float]
    vrange_R: tuple[float]
    threshold_L: float
    threshold_R: float
    smooth_L: bool
    smooth_R: bool

    def update_3d_vol(self, idx):
        if not 0 <= idx < self.img4D.shape[3]:
            raise ValueError("Invalid 4D index")
        self.volumeIdx = idx
        mat = self.img4D.dataobj[:, :, :, idx]
        img = Nifti1Image(mat, self.img4D.affine)
        self.current3Dvol = nili.resample_to_img(
            img, self.bgImg, interpolation='nearest')


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


def error(msg: str):
    print(msg, file=sys.stderr)
    exit(1)


def load_nii_or_error(file: str):
    try:
        return nib.load(file)
    except Exception as e:
        error(str(e))


nslice = slice(None, None, None)


@click.command()
@click.argument("atlas")
@click.argument("image")
def main(atlas:str, image: str):
    pp.init((1400, 700), pg.RESIZABLE, display_pos=(100, 100))
    el = pp.EventListener((
        pp.Handler.resize_event_handler(),
        pp.Handler.quit_event_handler()))
    atlasImg = load_nii_or_error(atlas)
    bgImg = utils.load_bg_template("ch2better")
    atlasImg = nili.resample_to_img(atlasImg, bgImg, interpolation='nearest')
    cmap = 'coolwarm'
    mni_coords = (36, -27, 66)
    vrange = None
    threshold = 20
    smooth = False
    index_coords = utils.mni_2_index(bgImg.affine, mni_coords)
    img4D = load_nii_or_error(image)
    state = State(atlas, atlasImg, image, img4D, None, bgImg, mni_coords, mni_coords,
                  index_coords, index_coords, 0, cmap, cmap, vrange, vrange, 
                  threshold, threshold, smooth, smooth)
    state.update_3d_vol(state.volumeIdx)

    run = True
    while run:
        pp.display(pp.compose(pp.empty_surface(0xFFFFFF), pp.LinLayout("h"))(
            render_view(el, state.atlasImg, state.bgImg, state.index_coords_L,
                        state.cmap_L, state.vrange_L, state.threshold_L,
                        state.smooth_L, "L"),
            LLItem(0)(pp.Line("v")),
            render_view(el, state.current3Dvol, 
                        state.bgImg, state.index_coords_R,
                        state.cmap_R, state.vrange_R, state.threshold_R,
                        state.smooth_R, "R")
        ))

        event = el.wait_for_keys(pg.K_q, sleeptime=0.005)
        run = handle_event(event, state)


def render_view(el, fg_img, bg_img, index_coords, cmap, vrange, threshold, 
                smooth, event_identifier):
    fg_splits = get_slices(fg_img.get_fdata(), index_coords)
    bg_splits = get_slices(bg_img.get_fdata(), index_coords)
    imgs = [make_overlayed_img(fg, bg, cmap, vrange, threshold, smooth)
        for fg, bg in zip(fg_splits, bg_splits)]

    for i, img in enumerate(imgs):
        imgCoords = compute_image_cross_coord(
            index_coords, i, bg_img.shape, ImgSize.from_(img))
        add_split_cross(img, imgCoords, target=img)

    return compose_3_way_plit_22(imgs, el, event_identifier)


def compute_image_cross_coord(index_coords, slice_dimension, nii_shape, imgShape):
    matCoords = MatCoords(*cut(index_coords, slice_dimension))
    matShape = MatShape(*cut(nii_shape, slice_dimension))
    return mat_coords_2_img_coords(matCoords, matShape, imgShape)


def handle_event(event, state):
    if event in (pg.K_q, pg.QUIT):
        return False
    elif type(event) == SliceClick:
        if pg.key.get_pressed()[pg.K_LCTRL] and event.side_identifier == "L":
            handle_change_target_slice(state, event)
        else:
            handle_slice_click(state, event)
    elif event == pg.VIDEORESIZE:
        # this is unnecessary, but explicit is better than implicit
        pass
    return True


def handle_change_target_slice(state, sc):
    side = sc.side_identifier
    imgSize = ImgSize(sc.rect.w, sc.rect.h)
    matShape = MatShape(*cut(state.bgImg.shape, sc.index))
    matCoords = image_coords_2_mat_coords(sc.coords, imgSize, matShape)
    mat_index = list(matCoords)
    index_coords = getattr(state, f"index_coords_{side}")
    mat_index.insert(sc.index, index_coords[sc.index])
    
    field_val = np.asscalar(state.atlasImg.get_fdata()[tuple(mat_index)])
    state.update_3d_vol(field_val)


def handle_slice_click(state: State, sc: SliceClick):
    side = sc.side_identifier
    imgSize = ImgSize(sc.rect.w, sc.rect.h)
    matShape = MatShape(*cut(state.bgImg.shape, sc.index))
    matCoords = image_coords_2_mat_coords(sc.coords, imgSize, matShape)
    mat_index = list(matCoords)
    index_coords = getattr(state, f"index_coords_{side}")
    mat_index.insert(sc.index, index_coords[sc.index])
    mni_index = utils.index_2_mni(state.bgImg.affine, mat_index)
    setattr(state, f"index_coords_{side}", mat_index)
    setattr(state, f"mni_coords_{side}", mni_index)
    

def cut(iter, index):
    return (x for i, x in enumerate(iter) if i != index)


def add_split_cross(img: pg.Surface, cross_coord: ImgCoords, 
                    color: pg.Color = 0, width: int = 2, target=None):
    target = target or img.copy()
    pg.draw.line(target, color, 
                 (cross_coord.x, 0), 
                 (cross_coord.x, img.get_height()),
                 width)
    pg.draw.line(target, color, 
                 (0, cross_coord.y), 
                 (img.get_width(), cross_coord.y),
                 width)
    return target


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


def slice_click_handler(event, x, y, rect, slice_index, event_identifier):
    if pp.is_left_click(event):
        return SliceClick(slice_index, ImgCoords(x, y), rect, event_identifier)
    else:
        return pp.EventConsumerInfo.DONT_CARE


def compose_3_way_plit_22(imgs, el: EventListener, event_identifier: str):
    assert len(imgs) == 3
    for i, img in enumerate(imgs):
        imgs[i] = pp.make_transparent_by_colorkey(img, img.get_at((0, 0)))

    def field(index):
        return el.mouse_area(lambda event, x, y, rect: slice_click_handler(
                event, x, y, rect, index, event_identifier), 
                ident=f"slice_clickhandler {index}{event_identifier}")(
                pp.Surface(scale=1)(imgs[index]))

    return pp.GridLayout()(
        # the order is due to conventions in the Neuroimaging field
        [field(1), field(0)],
        [field(2), None]
    )


def get_slices(mat, split_coord):
    return (
        mat[split_coord[0], :, :],
        mat[:, split_coord[1], :],
        mat[:, :, split_coord[2]])


def make_overlayed_img(fgMat, bgMat, cmap, vrange, threshold, smooth):
    assert len(fgMat.shape) == len(bgMat.shape) == 2
    img = pp.mat_to_surface(fgMat, pp.apply_color_map(cmap))
    img = pp.make_transparent_by_colorkey(img, img.get_at((0, 0)))
    img = pg.transform.rotate(img, 90)
    bg = pp.mat_to_surface(bgMat)
    bg = pp.make_transparent_by_colorkey(bg, bg.get_at((0, 0)))
    bg = pg.transform.rotate(bg, 90)
    result_size = (max(img.get_width(), bg.get_width()), 
                   max(img.get_height(), bg.get_height()))
    return pp.compose(pp.empty_surface(0xFFFFFF, result_size))(
        pp.Overlay(
            pp.Surface(smooth=smooth, scale=1)(bg),
            pp.Surface(smooth=smooth, scale=1)(img),
        ))


if __name__ == "__main__":
    main()
