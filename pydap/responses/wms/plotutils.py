from __future__ import division

from cStringIO import StringIO

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.colors import BoundaryNorm
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 'small'
rcParams['ytick.labelsize'] = 'small'

try:
    from PIL import Image
except:
    PIL = None

def make_colorbar(width, height, dpi, grid, orientation, transparent, norm, 
                  cmap, extend, paletted):
    figsize = width/dpi, height/dpi
    fig = Figure(figsize=figsize, dpi=dpi)
    fig.set_facecolor('white')
    fig.set_edgecolor('none')
    if orientation == 'vertical':
        ax = fig.add_axes([0.05, 0.05, 0.35, 0.90])
    else:
        ax = fig.add_axes([0.05, 0.55, 0.90, 0.40])
    if transparent:
        fig.figurePatch.set_alpha(0.0)
        ax.axesPatch.set_alpha(0.5)

    cb = ColorbarBase(ax, cmap=cmap, norm=norm,
            orientation=orientation, extend=extend)
    fontsize = 0
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
        ticks = cb.ax.get_xticklabels()
        for tick in ticks:
            txt = tick.get_text()
            ntxt = len(txt)
            fontsize = max(int(1.25*width/(len(ticks)*ntxt)), fontsize)
            fontsize = min(12, fontsize)
        for tick in cb.ax.get_xticklabels():
            tick.set_fontsize(fontsize)
            tick.set_color('black')

    # Decorate colorbar
    if 'units' in grid.attributes and 'long_name' in grid.attributes:
        units = grid.attributes['units']
        long_name = grid.attributes['long_name'].capitalize()
        if orientation == 'vertical':
            ax.set_ylabel('%s [%s]' % (long_name, units), fontsize=fontsize)
        else:
            ax.set_xlabel('%s [%s]' % (long_name, units), fontsize=12)

    # Save to buffer.
    canvas = FigureCanvas(fig)
    if paletted:
        # Since it does not seem to be possible to easily disable antialiasing
        # for colorbars we manually force the number of paletted colors
        """
        try:
            # We make reduce the number of colors to the number of colors
            # in the color scale plus 20 (arbitrary)
            nother = 20 
            ncolors = len(norm.boundaries) + nother
        except AttributeError:
            ncolors = None
        output = convert_paletted(canvas, ncolors=ncolors)
        """
        output = StringIO() 
        canvas.print_png(output)
    else:
        output = StringIO() 
        canvas.print_png(output)
    return output

def convert_paletted(canvas, ncolors=None, verbose=False):
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
        im = im.convert("P", palette=Image.ADAPTIVE, colors=ncolors)
        # Set all pixel values below ncolors to 1 and the rest to 0
        mask = Image.eval(alpha, lambda a: 255 if a <=128 else 0)
        # Paste the color of index ncolors and use alpha as a mask
        im.paste(ncolors, mask)
        # Truncate palette to actual size to save space
        im.palette.palette = im.palette.palette[:3*(ncolors+1)]
        im.save(outbuffer, 'png', optimize=False, transparency=ncolors)
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
