"""
Support for arrow heads on barb plots. Subclasses matplotlib Barbs class
and overrides _make_barbs method.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy import ma
from matplotlib.quiver import Barbs
import matplotlib.transforms as transforms
from matplotlib.patches import CirclePolygon

def arrow_barbs(ax, *args, **kw):
    """Make arrow barbs."""
    if not ax._hold:
        ax.cla()
    b = ArrowBarbs(ax, *args, **kw)
    ax.add_collection(b, autolim=True)
    ax.autoscale_view()
    return b

# This is a helper function that parses out the various combination of
# arguments for doing colored vector plots.  Pulling it out here
# allows both Quiver and Barbs to use it
def _parse_args(*args):
    X, Y, U, V, C = [None] * 5
    args = list(args)

    # The use of atleast_1d allows for handling scalar arguments while also
    # keeping masked arrays
    if len(args) == 3 or len(args) == 5:
        C = np.atleast_1d(args.pop(-1))
    V = np.atleast_1d(args.pop(-1))
    U = np.atleast_1d(args.pop(-1))
    if U.ndim == 1:
        nr, nc = 1, U.shape[0]
    else:
        nr, nc = U.shape
    if len(args) == 2:  # remaining after removing U,V,C
        X, Y = [np.array(a).ravel() for a in args]
        if len(X) == nc and len(Y) == nr:
            X, Y = [a.ravel() for a in np.meshgrid(X, Y)]
    else:
        indexgrid = np.meshgrid(np.arange(nc), np.arange(nr))
        X, Y = [np.ravel(a) for a in indexgrid]
    return X, Y, U, V, C

class ArrowBarbs(Barbs):
    def _make_barbs(self, u, v, nflags, nbarbs, half_barb, empty_flag, length,
                    pivot, sizes, fill_empty, flip, arrowhead=True):
        '''
        This function actually creates the wind barbs.  *u* and *v*
        are components of the vector in the *x* and *y* directions,
        respectively.

        *nflags*, *nbarbs*, and *half_barb*, empty_flag* are,
        *respectively, the number of flags, number of barbs, flag for
        *half a barb, and flag for empty barb, ostensibly obtained
        *from :meth:`_find_tails`.

        *length* is the length of the barb staff in points.

        *pivot* specifies the point on the barb around which the
        entire barb should be rotated.  Right now, valid options are
        'head' and 'middle'.

        *sizes* is a dictionary of coefficients specifying the ratio
        of a given feature to the length of the barb. These features
        include:

            - *spacing*: space between features (flags, full/half
               barbs)

            - *height*: distance from shaft of top of a flag or full
               barb

            - *width* - width of a flag, twice the width of a full barb

            - *emptybarb* - radius of the circle used for low
               magnitudes

        *fill_empty* specifies whether the circle representing an
        empty barb should be filled or not (this changes the drawing
        of the polygon).

        *flip* is a flag indicating whether the features should be flipped to
        the other side of the barb (useful for winds in the southern
        hemisphere.

        This function returns list of arrays of vertices, defining a polygon
        for each of the wind barbs.  These polygons have been rotated to
        properly align with the vector direction.
        '''

        #These control the spacing and size of barb elements relative to the
        #length of the shaft
        spacing = length * sizes.get('spacing', 0.125)
        full_height = length * sizes.get('height', 0.4)
        full_width = length * sizes.get('width', 0.25)
        empty_rad = length * sizes.get('emptybarb', 0.15)

        #Controls y point where to pivot the barb.
        pivot_points = dict(tip=0.0, middle=-length / 2., tail=-length)

        #Check for flip
        if flip:
            full_height = -full_height

        endx = 0.0
        endy = pivot_points[pivot.lower()]

        # Hardcoded settings for width, spacing, and arrowhead
        # Commented values is the same as quiver defaults
        headaxislength = 2.5 #4.5
        headlength     = 3   #5.0
        headwidth      = 2   #3.0
        width          = 0.15
        width          = width*length
        hwidth         = 0.5*width
        headwidth      = width*(headwidth-1)
        headlength     = width*headlength
        headaxislength = headlength-width*headaxislength
        spacing        = width*1.25

        # Get the appropriate angle for the vector components.  The offset is
        # due to the way the barb is initially drawn, going down the y-axis.
        # This makes sense in a meteorological mode of thinking since there 0
        # degrees corresponds to north (the y-axis traditionally)
        angles = -(ma.arctan2(v, u) + np.pi / 2)

        # Used for low magnitude.  We just get the vertices, so if we make it
        # out here, it can be reused.  The center set here should put the
        # center of the circle at the location(offset), rather than at the
        # same point as the barb pivot; this seems more sensible.
        circ   = CirclePolygon((0, 0), radius=empty_rad).get_verts()
        circ_o = CirclePolygon((0, 0), radius=empty_rad+hwidth).get_verts()
        circ_i = CirclePolygon((0, 0), radius=empty_rad-hwidth).get_verts()
        if fill_empty:
            empty_barb = circ
        else:
            # If we don't want the empty one filled, we make a degenerate
            # polygon that wraps back over itself (not optimal)
            empty_barb = np.concatenate((circ_i, circ_o[::-1]))

        barb_list = []
        for index, angle in np.ndenumerate(angles):
            #If the vector magnitude is too weak to draw anything, plot an
            #empty circle instead
            if empty_flag[index]:
                #We can skip the transform since the circle has no preferred
                #orientation
                barb_list.append(empty_barb)
                continue

            if arrowhead:
                # Fancy wind_barb with arrow
                poly_verts = [(endx, endy),
                              (endx+headwidth/2, endy+headaxislength),
                              (endx-hwidth, endy-headlength+headaxislength),
                              (endx-headwidth/2-width, endy+headaxislength),
                              (endx-width, endy),
                              (endx-width, endy+length)]
            else:
                # Normal wind_barb without arrow
                poly_verts = [(endx, endy),
                              (endx-hwidth, endy-hwidth),
                              (endx-width, endy), (endx-width, endy+length)]
            offset = length

            # Add vertices for each flag
            for i in range(nflags[index]):
                # The spacing that works for the barbs is a little to much for
                # the flags, but this only occurs when we have more than 1
                # flag.
                poly_verts.extend(
                    [[endx, endy + offset],
                     [endx + full_height + hwidth, endy - full_width / 2 + offset],
                     [endx, endy - full_width + offset]])

                offset -= full_width + spacing/4.0

            # Add vertices for each barb.  These really are lines, but works
            # great adding 3 vertices that basically pull the polygon out and
            # back down the line
            for i in range(nbarbs[index]):
                poly_verts.extend(
                    [(endx, endy + offset),
                     (endx + full_height, endy + offset + full_width / 2),
                     (endx + full_height + hwidth, endy + offset + full_width / 2 - hwidth),
                     (endx + full_height, endy + offset + full_width / 2 - width),
                     (endx, endy + offset - width)])

                offset -= spacing

            # Add the vertices for half a barb, if needed
            if half_barb[index]:
                # If the half barb is the first on the staff, traditionally it
                # is offset from the end to make it easy to distinguish from a
                # barb with a full one
                if offset == length:
                    poly_verts.append((endx, endy + offset))
                    offset -= 1.5 * spacing
                poly_verts.extend(
                    [(endx, endy + offset),
                     (endx + full_height / 2, endy + offset + full_width / 4),
                     (endx + full_height / 2 + hwidth, endy + offset + full_width / 4 - hwidth),
                     (endx + full_height / 2, endy + offset + full_width / 4 - width),
                     (endx, endy + offset - width)])

            # Rotate the barb according the angle. Making the barb first and
            # then rotating it made the math for drawing the barb really easy.
            # Also, the transform framework makes doing the rotation simple.
            poly_verts = transforms.Affine2D().rotate(-angle).transform(
                poly_verts)
            barb_list.append(poly_verts)

        return barb_list
