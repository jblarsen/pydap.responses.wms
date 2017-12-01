"""
Plotting utilities for the pydap WMS response.
"""

# Standard library imports
from io import BytesIO

# External imports
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colorbar import ColorbarBase
from matplotlib import rcParams
from matplotlib.ticker import NullLocator
from PIL import Image


# Matplotlib defaults
rcParams['xtick.labelsize'] = 'small'
rcParams['ytick.labelsize'] = 'small'
#rcParams['text.antialiased'] = False


def make_colorbar(width, height, dpi, orientation, transparent, norm,
                  cmap, extend, add_ticks, center_labels):
    """\
    Return colorbar saved to BytesIO object.
    """
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
            cbheight = cbratio*cbwidth*width/float(height)
            ax = fig.add_axes([(1-cbwidth)/2.0, 1-cbheight, cbwidth, cbheight])
        else:
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    if transparent:
        fig.figurePatch.set_alpha(0.0)
        ax.axesPatch.set_alpha(0.5)

    if len(norm.boundaries) <= 14: # label all colors
        nb = norm.boundaries
        if center_labels:
            nb_pos = nb + (nb[1]-nb[0])*0.5
        else:
            nb_pos = nb
    else:
        nb_pos = None

    cb = ColorbarBase(ax, cmap=cmap, norm=norm, ticks=nb_pos, drawedges=True,
                      orientation=orientation, extend=extend,
                      extendfrac='auto')

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
            newticks = []
            for tick in ticks:
                # Substitute m-minus with hyphen-minus (n-minus)
                newtick = tick.get_text().replace(u'\u2212', u'\u002D')
                stxt += len(newtick)
                newticks.append(newtick)
            cb.ax.set_xticklabels(newticks)
            fontsize = width/(2*(stxt+5.0))
            fontsize = max(min(12, fontsize), 6)
            for tick in cb.ax.get_xticklabels():
                tick.set_fontsize(fontsize)
                tick.set_color('black')
    else:
        cb.ax.xaxis.set_major_locator(NullLocator())

    # Save to buffer.
    canvas = FigureCanvas(fig)
    output = BytesIO()
    canvas.print_png(output)
    return output

#@profile
def convert_paletted(canvas, ncolors=None, verbose=False):
    """\
    Convert matplotlib canvas to paletted PNG if there are less
    than 256 colors (if ncolors is not None paletting is always
    employed. Pixels are considered transparent if the
    alpha channel value is <= 128 and opaque otherwise.

    Returns a memory file (BytesIO object) containing the PNG
    image.
    """
    outbuffer = BytesIO()

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
        mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
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

def mslext2text(ncchars):
    """\
    Convert text of H and L's (high and low pressures) to TeXlike plot
    format
    """
    colors = {'H': 'blue', 'L': 'red'}
    #ncstring = str(netCDF4.chartostring(ncchars)).strip()
    ncstring = ncchars
    if ':' in ncstring:
        sym, val = ncstring.split(':')
        hlt = sym + '\n$\\regular^{\,'+val+'}}$'
        return hlt, colors[sym]
    return ncstring, 'black'

