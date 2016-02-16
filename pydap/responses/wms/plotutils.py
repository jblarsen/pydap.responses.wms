from __future__ import division
import ctypes
from cStringIO import StringIO

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.colors import BoundaryNorm
from matplotlib import rcParams
from matplotlib.ticker import NullLocator

rcParams['xtick.labelsize'] = 'small'
rcParams['ytick.labelsize'] = 'small'
#rcParams['text.antialiased'] = False

try:
    from PIL import Image
except:
    PIL = None
try:
    from wand.image import Image as WandImage
    from wand.api import library
    # Register C-type arguments
    library.MagickQuantizeImage.argtypes = [ctypes.c_void_p,
                                            ctypes.c_size_t,
                                            ctypes.c_int,
                                            ctypes.c_size_t,
                                            ctypes.c_int,
                                            ctypes.c_int]
    library.MagickQuantizeImage.restype = None
    library.MagickGetImageColors.argtypes = [ctypes.c_void_p]
    library.MagickGetImageColors.restype = ctypes.c_size_t
except:
    wand = None

def make_colorbar(width, height, dpi, grid, orientation, transparent, norm, 
                  cmap, extend, paletted, add_ticks, center_labels):
    dpi = 2*dpi
    figsize = width/dpi, height/dpi
    fig = Figure(figsize=figsize, dpi=dpi, frameon=False)

    fig.set_facecolor('white')
    fig.set_edgecolor('none')

    if orientation == 'vertical':
        if add_ticks:
            ax = fig.add_axes([0.05, 0.05, 0.60, 0.90])
        else:
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    else:
        if add_ticks:
            cbwidth = 0.95 # ratio of width to figsize
            cbratio = 0.10 # ratio of width to height
            cbheight= cbratio*cbwidth*width/float(height)
            ax = fig.add_axes([(1-cbwidth)/2.0, 1-cbheight, cbwidth, cbheight])
        else:
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    if transparent:
        fig.figurePatch.set_alpha(0.0)
        ax.axesPatch.set_alpha(0.5)

    if len(norm.boundaries) <= 12: # label all colors
        nb = norm.boundaries
        if center_labels:
            nb_pos = nb + (nb[1]-nb[0])*0.5
        else:
            nb_pos = nb
    else:
        nb_pos = None
      
    cb = ColorbarBase(ax, cmap=cmap, norm=norm, ticks=nb_pos, drawedges=True,
            orientation=orientation, extend=extend, extendfrac='auto')

    if center_labels:
        cb.ax.set_xticklabels(nb)
        cb.ax.tick_params('both', length=0, width=1)

    fontsize = 0
    #cb.solids.set_antialiased(False)
    #cb.patch.set_antialiased(False)
    #cb.outline.set_antialiased(False)

    if add_ticks:
        #    ticks.set_antialiased(False)
        if orientation == 'vertical':
            for tick in cb.ax.get_yticklabels():
                txt = tick.get_text()
                ntxt = len(txt)
                fontsize = max(int(0.50*width/ntxt), fontsize)
                fontsize = min(14, fontsize)
            for tick in cb.ax.get_yticklabels():
                tick.set_fontsize(fontsize)
                tick.set_color('black')
        else:
            #cb.ax.xaxis.set_ticks()
            ticks = cb.ax.get_xticklabels()
            stxt = 0
            # get total ticks text
            for tick in ticks:
                stxt += len(tick.get_text())
            fontsize = width/(2*(stxt+5.0))
            fontsize = max(min(12, fontsize),7)
            for tick in cb.ax.get_xticklabels():
                tick.set_fontsize(fontsize)
                tick.set_color('black')
    else:
        cb.ax.xaxis.set_major_locator(NullLocator())

    # Save to buffer.
    canvas = FigureCanvas(fig)
    output = StringIO() 
    canvas.print_png(output)
    """
    if paletted:
        # The text on colorbars look bad when we disable antialiasing
        # entirely. We therefore just force the number of paletted
        # colors to a reasonable number
        try:
            # We reduce the number of colors to the number of colors
            # in the color scale plus 20 to allow for some
            # text antialiasing.
            nother = 20 
            ncolors = len(norm.boundaries) + nother
        except AttributeError:
            ncolors = None
        output = convert_paletted(canvas, ncolors=ncolors)
    else:
        output = StringIO() 
        canvas.print_png(output)
    """
    return output

#@profile
def convert_paletted(canvas, ncolors=None, backend='PIL', verbose=False):
    """\
    Convert matplotlib canvas to paletted PNG if there are less
    than 256 colors (if ncolors is not None paletting is always 
    employed. Pixels are considered transparent if the
    alpha channel value is <= 128 and opaque otherwise. 

    Returns a memory file (StringIO object) containing the PNG
    image.
    """
    if backend == 'wand':
        outbuffer = convert_paletted_wand(canvas, ncolors, verbose)
    elif backend == 'PIL':
        outbuffer = convert_paletted_pil(canvas, ncolors, verbose)
    else:
        raise ValueError('Invalid PNG conversion backend specified')
    return outbuffer

#@profile
def DoColorReduction(img, color_count):
    """Reduce image color count"""
    assert isinstance(img, WandImage)
    assert isinstance(color_count, int)
    colorspace = 1 # assuming RGB?
    treedepth = 0 # Autodetect
    dither = 0 # False
    merror = 0 # False
    # NOTE: It is slow to get the number of colors from the image
    ncolors = library.MagickGetImageColors(img.wand)
    library.MagickQuantizeImage(img.wand, color_count, colorspace,
                                         treedepth, dither, merror)

#@profile
def convert_paletted_wand(canvas, ncolors=None, verbose=False):
    """
    ImageMagick based conversion to paletted PNG.
    """
    outbuffer = StringIO() 
   
    # Read image
    inbuffer = StringIO()
    # TODO: If and when wand supports reading raw RGBA we can
    # use the much faster canvas.print_to_buffer() method
    canvas.print_png(inbuffer)
    inbuffer.seek(0)
    with WandImage(file=inbuffer) as im:
        # TODO: Maybe take number of colors from mpl.norm instance
        # to reduce overhead from getting colors from image (30 ms
        # for typical 512x512 image).
        # Find number of colors
        if ncolors is None:
            ncolors = len(im.histogram)
        if verbose:
            print(ncolors)
            print(im.histogram)

        # Only convert if the number of colors is less than 256
        if ncolors < 256:
            DoColorReduction(im, ncolors)
            im.save(file=outbuffer)
        else:
            canvas.print_png(outbuffer)
    return outbuffer

#@profile
def convert_paletted_pil(canvas, ncolors=None, verbose=False):
    """\
    Convert matplotlib canvas to paletted PNG if there are less
    than 256 colors (if ncolors is not None paletting is always 
    employed. Pixels are considered transparent if the
    alpha channel value is <= 128 and opaque otherwise. 

    Returns a memory file (StringIO object) containing the PNG
    image.
    """
    outbuffer = StringIO() 
   
    # Read image
    buf, size = canvas.print_to_buffer()
    im = Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
    # Find number of colors
    if ncolors is None:
        colors = im.getcolors(255)
        if colors is not None:
            ncolors = len(colors)
    if verbose:
        print(colors)
        print(im.histogram())
    # Only convert if the number of colors is less than 256
    if ncolors is not None:
        # Get alpha band
        alpha = im.split()[-1]
        # Convert to paletted image
        im = im.convert("RGB")
        #im = im.convert("P", palette=Image.ADAPTIVE, colors=ncolors)
        im = im.quantize(colors=ncolors, method=2, kmeans=0, palette=None)
        # Set all pixel values below ncolors to 1 and the rest to 0
        mask = Image.eval(alpha, lambda a: 255 if a <=128 else 0)
        # Paste the color of index ncolors and use alpha as a mask
        im.paste(ncolors, mask)
        # Truncate palette to actual size to save space
        im.palette.palette = im.palette.palette[:3*(ncolors+1)]
        im.save(outbuffer, 'png', compress_level=7, transparency=ncolors)
    else:
        canvas.print_png(outbuffer)
    return outbuffer

def modify_contour_levels(levels, extend):
    """\
    Modifies levels for contouring so that we do not contour
    upper and lower bounds.
    """
    outlevels = levels[:]
    if extend in ['min', 'neither']:
        outlevels = outlevels[:-1]
    if extend in ['max', 'neither']:
        outlevels = outlevels[1:]
    return outlevels
