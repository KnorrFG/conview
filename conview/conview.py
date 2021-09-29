from time import sleep
import textwrap as tw

import nilearn.image as nili
import nibabel as nib
import numpy as np
from pygame.display import update
import pyparadigm as pp
import pygame as pg
import click
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyparadigm.surface_composition import Padding

from .utils import (
    text, mni_2_index, index_2_mni, load_bg_template, partition_all,
    error, load_nii_or_error, threshold_slice, get_slices, cut,
    image_coords_2_mat_coords, mat_coords_2_img_coords)
from .types import (
    State, ImgCoords, MatCoords, ImgSize, MatShape, SliceClick
)


help_text = """
This is Conview, a tool to get an intuitive grasp of your connectivity matrices.
To view the connectivity of a region on the right side, hold control while you
click on the Region on the Left side.

List of Hotkeys:
(all changes will apply to the side you clicked on last, left if you havent 
clicked yet)

t: set threshold
m: change color map 
i: change vmin
a: change vmax
v: reset vrange to default
g: toggle location and value information
h: show this help

press any key to leave this screen.
"""


menu_bar_heigth = 20
color_bar_height = 50
bar_bg_color = 0x999999
nslice = slice(None, None, None)

@click.command()
@click.argument("atlas")
@click.argument("image")
def main(atlas:str, image: str):
    pp.init((1400, 700), pg.RESIZABLE, display_pos=(100, 100), title="Conview")
    el = pp.EventListener((
        pp.Handler.resize_event_handler(),
        pp.Handler.quit_event_handler()))
    atlasImg = load_nii_or_error(atlas)
    bgImg = load_bg_template("ch2better")
    atlasImg = nili.resample_to_img(atlasImg, bgImg, interpolation='nearest')
    cmap = 'coolwarm'
    mni_coords = (36, -27, 66)
    vrange = (None, None)
    threshold = 0
    smooth = False
    index_coords = mni_2_index(bgImg.affine, mni_coords)
    img4D = load_nii_or_error(image)
    state = State(
        atlas, atlasImg, None, image, img4D, None, None, bgImg, mni_coords,
        mni_coords, index_coords, index_coords, 0, cmap, cmap, *vrange, None,
        *vrange, None, threshold, threshold, smooth, smooth, "L", True)
    update_3d_vol_in_state(state, state.volumeIdx)
    update_mat_in_state(state, "L")

    run = True
    while run:
        disp_surf = pg.display.get_surface()
        disp_surf.fill(0xFFFFFF)
        pp.compose(disp_surf, pp.FreeFloatLayout())(
            pp.FRect(0, 0, 0.5, menu_bar_heigth)(
                compose_bar(
                    state.cmap_L, state.threshold_L, 
                    (state.normalizer_L.vmin, state.normalizer_L.vmax))),
            pp.FRect(0, menu_bar_heigth, 0.5, -(menu_bar_heigth + color_bar_height))(
                pp.Overlay(
                    render_coord_info(
                        state.index_coords_L, state.mni_coords_L, 
                        state.atlasImg.get_fdata()[state.index_coords_L], 
                        state.show_info
                    ),
                    render_view(el, state.atlasMat, state.bgImg, state.index_coords_L,
                                state.cmap_L, state.smooth_L, "L"))),
            pp.FRect(0, -color_bar_height, 0.5, color_bar_height)(
                make_scale(state.normalizer_L, state.cmap_L)),

            pp.FRect(0.5, 0, 0.5, menu_bar_heigth)(
                compose_bar(
                    state.cmap_R, state.threshold_R, 
                    (state.normalizer_R.vmin, state.normalizer_R.vmax))),
            pp.FRect(0.5, menu_bar_heigth, 0.5, -(menu_bar_heigth + color_bar_height))(
                pp.Overlay(
                    render_coord_info(
                        state.index_coords_R, state.mni_coords_R, 
                        state.current3DVol.get_fdata()[state.index_coords_R], 
                        state.show_info
                    ),
                    render_view(el, state.current3DMat,
                                state.bgImg, state.index_coords_R,
                                state.cmap_R, state.smooth_R, "R"))),
            pp.FRect(0.5, -color_bar_height, 0.5, color_bar_height)(
                make_scale(state.normalizer_R, state.cmap_R)),

            pp.FRect(0.5, 0, 0, 1.0)(pp.Line("v")))
        pg.display.flip()

        event = el.wait_for_keys(pg.K_q, pg.K_m, pg.K_t, pg.K_i, pg.K_a, pg.K_v, 
            pg.K_g, pg.K_h,
            sleeptime=0.005)
        run = handle_event(event, state)


def render_coord_info(coords, mni_coords, val, show_info):
    if not show_info:
        return pp.Surface()
    return pp.GridLayout()(
        [None, None], 
        [None,
         pp.Padding.from_scale(0.8)(
             text(tw.dedent(f"""\
            loc: {coords}
            mni: {mni_coords}
            value: {val:.3f}""")))])


def compose_bar(cmap, threshold, vrange):
    return text(f"cmap: {cmap} | threshold: {threshold} | "
                f"vrange: {vrange[0]:.3f}, {vrange[1]:.3f}")


def update_3d_vol_in_state(state, idx):
    """its defined here instead as a method of state, because this way i dont
    get into trouble for circular imorts between utils and types"""
    if not 0 <= idx < state.img4D.shape[3]:
        raise ValueError("Invalid 4D index")
    state.volumeIdx = idx
    mat = state.img4D.dataobj[:, :, :, idx]
    img = nib.Nifti1Image(mat, state.img4D.affine)
    state.current3DVol = nili.resample_to_img(
        img, state.bgImg, interpolation='nearest')
    update_mat_in_state(state, "R")


def render_view(el, fg_mat, bg_img, index_coords, cmap,  
                smooth, event_identifier):
    fg_splits = get_slices(fg_mat, index_coords)
    bg_splits = get_slices(bg_img.get_fdata(), index_coords)
    imgs = [make_overlayed_img(fg, bg, cmap, smooth)
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
    elif event == pg.K_m:
        change_color_map(state, state.last_active_side)
    elif event == pg.K_t:
        update_state_from_user_input(state, "threshold", state.last_active_side)
        update_mat_in_state(state, state.last_active_side)
    elif event == pg.K_i:
        update_state_from_user_input(state, "vmin", state.last_active_side)
        update_mat_in_state(state, state.last_active_side)
    elif event == pg.K_a:
        update_state_from_user_input(state, "vmax", state.last_active_side)
        update_mat_in_state(state, state.last_active_side)
    elif event == pg.K_v:
        reset_vrange(state, state.last_active_side)
        update_mat_in_state(state, state.last_active_side)
    elif event == pg.K_g:
        state.show_info = not state.show_info
    elif event == pg.K_h:
        display_help()
    elif type(event) == SliceClick:
        state.last_active_side = event.side_identifier
        if pg.key.get_pressed()[pg.K_LCTRL] and event.side_identifier == "L":
            handle_change_target_slice(state, event)
        else:
            handle_slice_click(state, event)
    elif event == pg.VIDEORESIZE:
        # this is unnecessary, but explicit is better than implicit
        pass
    return True


def reset_vrange(state, side):
    if side == "L":
        state.vmin_L, state.vmax_L = None, None
    else:
        state.vmin_R, state.vmax_R = None, None

def update_mat_in_state(state, side):
    field, source = ("atlasMat", "atlasImg") if side == "L" \
        else ('current3DMat', 'current3DVol')
    mat = getattr(state, source).get_fdata()
    mat = threshold_slice(mat, getattr(state, "threshold_" + side))
    vrange = (getattr(state, "vmin_" + side), getattr(state, "vmax_" + side))
    normalizer = mpl.colors.Normalize(*vrange)
    masked_mat = np.ma.masked_array(mat, np.isnan(mat))
    mat = normalizer(masked_mat)
    setattr(state, field, mat.data)
    setattr(state, "normalizer_" + side, normalizer)


def handle_change_target_slice(state, sc):
    side = sc.side_identifier
    imgSize = ImgSize(sc.rect.w, sc.rect.h)
    matShape = MatShape(*cut(state.bgImg.shape, sc.index))
    matCoords = image_coords_2_mat_coords(sc.coords, imgSize, matShape)
    mat_index = list(matCoords)
    index_coords = getattr(state, f"index_coords_{side}")
    mat_index.insert(sc.index, index_coords[sc.index])
    
    field_val = np.asscalar(state.atlasImg.get_fdata()[tuple(mat_index)])
    update_3d_vol_in_state(state, field_val)


def handle_slice_click(state: State, sc: SliceClick):
    side = sc.side_identifier
    imgSize = ImgSize(sc.rect.w, sc.rect.h)
    matShape = MatShape(*cut(state.bgImg.shape, sc.index))
    matCoords = image_coords_2_mat_coords(sc.coords, imgSize, matShape)
    mat_index = list(matCoords)
    index_coords = getattr(state, f"index_coords_{side}")
    mat_index.insert(sc.index, index_coords[sc.index])
    mni_index = index_2_mni(state.bgImg.affine, mat_index)
    setattr(state, f"index_coords_{side}", tuple(mat_index))
    setattr(state, f"mni_coords_{side}", mni_index)
    

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


def slice_click_handler(event, x, y, rect, slice_index, event_identifier):
    if pp.is_left_click(event):
        return SliceClick(slice_index, ImgCoords(x, y), rect, event_identifier)
    else:
        return pp.EventConsumerInfo.DONT_CARE


def compose_3_way_plit_22(imgs, el: pp.EventListener, event_identifier: str):
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


def make_overlayed_img(fgMat, bgMat, cmap, smooth):
    assert len(fgMat.shape) == len(bgMat.shape) == 2
    img = pp.mat_to_surface(fgMat, pp.apply_color_map(cmap, normalize=False))
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


def update_state_from_user_input(state, field, side):
    input_error = False
    caption = f"new value for {field}:\n\n\n<ESC>: Cancel   <RETURN>: OK"
    while True:
        val = pp.string_dialog(caption)
        if val is None:
            return
        try:
            setattr(state, f"{field}_{side}", float(val))
            return
        except ValueError:
            if not input_error:
                caption += "\n\n\nThe input must be a number\nuse . as decimal"\
                    + "separator"
                input_error = True


def make_color_map_surface(color_map):
    vals = np.linspace(-1, 1, 256)
    surf = pp.mat_to_surface(np.expand_dims(vals, axis=0), 
                             pp.apply_color_map(color_map))
    return pp.Surface(scale=1, keep_aspect_ratio=False)(surf)


def make_scale(normalizer, color_map):
    return pp.LinLayout("v")(
            pp.LLItem(0.5),
            pp.Padding(0.1, 0.1, 0, 0)(make_color_map_surface(color_map)),
            pp.LLItem(0.5),
            pp.Padding(0.1, 0.1, 0, 0)(
                pp.LinLayout("h")(
                    pp.LLItem(0.1),
                    text(f"{normalizer.vmin:.2f}"),
                    pp.LLItem(4),
                    text(f"{0.5 * (normalizer.vmin + normalizer.vmax):.2f}"),
                    pp.LLItem(4),
                    text(f"{normalizer.vmax:.2f}"))))


def display_help():
    pp.display(pp.compose(pp.empty_surface(0xFFFFFF), pp.LinLayout("v"))(
        text("Help"),
        pp.LLItem(9)(
            pp.Padding.from_scale(0.9)(text(help_text) 
        ))))
    pp.EventListener().wait_for_unicode_char()


#============================================================================== 
# Color Map Stuff
#============================================================================== 
choose_color_map_text = """
Choose a color map by clicking on it. 
Press a number key to change the page.
Press any other key to go back"""

def change_color_map(state, side):
    el = pp.EventListener((pp.Handler.resize_event_handler(),))
    page = 0

    def make_color_map_cell(name):
        return el.mouse_area(lambda e, x, y: name if pp.is_left_click(e)\
                                else pp.EventConsumerInfo.DONT_CARE, page)(
                    pp.Padding.from_scale(0.9)(
                        pp.LinLayout("v")(
                            make_color_map_surface(name),
                            pp.LLItem(0.5),
                            text(name))))

    pages = list(partition_all(26, plt.colormaps()))
    while True:
        sleep(0.001)
        cells = partition_all(2, (make_color_map_cell(name) for name in
                                  pages[page]))
        pp.display(pp.compose(pp.empty_surface(0xFFFFFF), pp.LinLayout("v"))(
            pp.LLItem(4)(text(choose_color_map_text, align="left")),
            pp.LLItem(10)(pp.GridLayout()(*(list(row) for row in cells))),
            page_widget(len(pages), page)))

        res = el.group(page).wait_for_unicode_char()
        if res == pg.VIDEORESIZE:
            continue
        elif len(res) > 1:
            setattr(state, f"cmap_{side}", res)
            return
        else:
            try:
                res_int = int(res)
                if res_int < len(pages):
                    page = res_int
            except ValueError:
                return


def make_color_map_surface(color_map):
    vals = np.linspace(-1, 1, 256)
    surf = pp.mat_to_surface(np.expand_dims(vals, axis=0), 
                             pp.apply_color_map(color_map))
    return pp.Surface(scale=0.9, keep_aspect_ratio=False)(surf)


def page_widget(pages, active):
    texts = [text(str(i)) for i in range(pages)]
    return pp.LinLayout("h")(
            pp.LLItem(2),
            *(t if i != active else 
                                    pp.Border(color=0)(t) 
                               for i, t in enumerate(texts)),
            pp.LLItem(2))


if __name__ == "__main__":
    main()
