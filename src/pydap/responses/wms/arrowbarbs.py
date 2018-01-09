"""
Support for arrow heads on barb plots. Subclasses matplotlib Barbs class
and overrides _make_barbs method.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy import ma
#from matplotlib.quiver import Barbs
import matplotlib.transforms as transforms
from matplotlib.patches import CirclePolygon
import matplotlib.collections as mcollections
from matplotlib.cbook import delete_masked_points

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

def _check_consistent_shapes(*arrays):
    all_shapes = set(a.shape for a in arrays)
    if len(all_shapes) != 1:
        raise ValueError('The shapes of the passed in arrays do not match.')

class ArrowBarbs(mcollections.PolyCollection):
    '''
    Specialized PolyCollection for barbs.

    The only API method is :meth:`set_UVC`, which can be used to
    change the size, orientation, and color of the arrows.  Locations
    are changed using the :meth:`set_offsets` collection method.
    Possibly this method will be useful in animations.

    There is one internal function :meth:`_find_tails` which finds
    exactly what should be put on the barb given the vector magnitude.
    From there :meth:`_make_barbs` is used to find the vertices of the
    polygon to represent the barb based on this information.
    '''
    # This may be an abuse of polygons here to render what is essentially maybe
    # 1 triangle and a series of lines.  It works fine as far as I can tell
    # however.
    def __init__(self, ax, *args, **kw):
        """
        The constructor takes one required argument, an Axes
        instance, followed by the args and kwargs described
        by the following pylab interface documentation:
        %(barbs_doc)s
        """
        self._pivot = kw.pop('pivot', 'tip')
        self._length = kw.pop('length', 7)
        barbcolor = kw.pop('barbcolor', None)
        flagcolor = kw.pop('flagcolor', None)
        self.sizes = kw.pop('sizes', dict())
        self.fill_empty = kw.pop('fill_empty', False)
        self.barb_increments = kw.pop('barb_increments', dict())
        self.rounding = kw.pop('rounding', True)
        self.flip = kw.pop('flip_barb', False)
        transform = kw.pop('transform', ax.transData)

        # Flagcolor and and barbcolor provide convenience parameters for
        # setting the facecolor and edgecolor, respectively, of the barb
        # polygon.  We also work here to make the flag the same color as the
        # rest of the barb by default

        if None in (barbcolor, flagcolor):
            kw['edgecolors'] = 'face'
            if flagcolor:
                kw['facecolors'] = flagcolor
            elif barbcolor:
                kw['facecolors'] = barbcolor
            else:
                # Set to facecolor passed in or default to black
                kw.setdefault('facecolors', 'k')
        else:
            kw['edgecolors'] = barbcolor
            kw['facecolors'] = flagcolor

        # Explicitly set a line width if we're not given one, otherwise
        # polygons are not outlined and we get no barbs
        if 'linewidth' not in kw and 'lw' not in kw:
            kw['linewidth'] = 1

        # Parse out the data arrays from the various configurations supported
        x, y, u, v, c = _parse_args(*args)
        self.x = x
        self.y = y
        xy = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))

        # Make a collection
        barb_size = self._length ** 2 / 4  # Empirically determined
        mcollections.PolyCollection.__init__(self, [], (barb_size,),
                                             offsets=xy,
                                             transOffset=transform, **kw)
        self.set_transform(transforms.IdentityTransform())

        self.set_UVC(u, v, c)

    def _find_tails(self, mag, rounding=True, half=5, full=10, flag=50):
        '''
        Find how many of each of the tail pieces is necessary.  Flag
        specifies the increment for a flag, barb for a full barb, and half for
        half a barb. Mag should be the magnitude of a vector (i.e., >= 0).

        This returns a tuple of:

            (*number of flags*, *number of barbs*, *half_flag*, *empty_flag*)

        *half_flag* is a boolean whether half of a barb is needed,
        since there should only ever be one half on a given
        barb. *empty_flag* flag is an array of flags to easily tell if
        a barb is empty (too low to plot any barbs/flags.
        '''

        # If rounding, round to the nearest multiple of half, the smallest
        # increment
        if rounding:
            mag = half * (mag / half + 0.5).astype(np.int)

        num_flags = np.floor(mag / flag).astype(np.int)
        mag = np.mod(mag, flag)

        num_barb = np.floor(mag / full).astype(np.int)
        mag = np.mod(mag, full)

        half_flag = mag >= half
        empty_flag = ~(half_flag | (num_flags > 0) | (num_barb > 0))

        return num_flags, num_barb, half_flag, empty_flag

    def _make_barbs(self, u, v, nflags, nbarbs, half_barb, empty_flag, length,
                    pivot, sizes, fill_empty, flip):
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

        # These control the spacing and size of barb elements relative to the
        # length of the shaft
        spacing = length * sizes.get('spacing', 0.125)
        full_height = length * sizes.get('height', 0.4)
        full_width = length * sizes.get('width', 0.25)
        empty_rad = length * sizes.get('emptybarb', 0.15)
        width           = length * sizes.get('width', 0.05)
        headwidth       =  width *(sizes.get('headwidth', 1.0) - 1)
        headlength      =  width * sizes.get('headlength', 1.0)
        headaxislength  =  width * sizes.get('headaxislength', 1.0)

        hwidth          = 0.5*width
        headaxislength  = headlength-headaxislength
        spacing         = max(spacing, width*1.1)

        # Controls y point where to pivot the barb.
        pivot_points = dict(tip=0.0, middle=-length / 2., tail=-length)

        # Check for flip
        if flip:
            full_height = -full_height

        endx = 0.0
        endy = pivot_points[pivot.lower()]

        # Get the appropriate angle for the vector components.  The offset is
        # due to the way the barb is initially drawn, going down the y-axis.
        # This makes sense in a meteorological mode of thinking since there 0
        # degrees corresponds to north (the y-axis traditionally)
        angles = -(ma.arctan2(v, u) + np.pi / 2)

        # Used for low magnitude.  We just get the vertices, so if we make it
        # out here, it can be reused.  The center set here should put the
        # center of the circle at the location(offset), rather than at the
        # same point as the barb pivot; this seems more sensible.
        circ = CirclePolygon((0, 0), radius=empty_rad).get_verts()

        if fill_empty:
            empty_barb = circ
        else:
            # If we don't want the empty one filled, we make a degenerate
            # polygon that wraps back over itself
            empty_barb = np.concatenate((circ, circ[::-1]))

        barb_list = []
        for index, angle in np.ndenumerate(angles):
            # If the vector magnitude is too weak to draw anything, plot an
            # empty circle instead
            if empty_flag[index]:
                # We can skip the transform since the circle has no preferred
                # orientation
                barb_list.append(empty_barb)
                continue

            poly_verts = [(endx, endy),
                          (endx+headwidth/2, endy+headaxislength),
                          (endx-hwidth, endy-headlength+headaxislength),
                          (endx-headwidth/2-width, endy+headaxislength),
                          (endx-width, endy),
                          (endx-width, endy+length)]

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

                offset -= full_width + spacing/2.0

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

    def set_UVC(self, U, V, C=None):
        self.u = ma.masked_invalid(U, copy=False).ravel()
        self.v = ma.masked_invalid(V, copy=False).ravel()
        if C is not None:
            c = ma.masked_invalid(C, copy=False).ravel()
            x, y, u, v, c = delete_masked_points(self.x.ravel(),
                                                 self.y.ravel(),
                                                 self.u, self.v, c)
            _check_consistent_shapes(x, y, u, v, c)
        else:
            x, y, u, v = delete_masked_points(self.x.ravel(), self.y.ravel(),
                                              self.u, self.v)
            _check_consistent_shapes(x, y, u, v)

        magnitude = np.hypot(u, v)
        flags, barbs, halves, empty = self._find_tails(magnitude,
                                                       self.rounding,
                                                       **self.barb_increments)

        # Get the vertices for each of the barbs

        plot_barbs = self._make_barbs(u, v, flags, barbs, halves, empty,
                                      self._length, self._pivot, self.sizes,
                                      self.fill_empty, self.flip)
        self.set_verts(plot_barbs)

        # Set the color array
        if C is not None:
            self.set_array(c)

        # Update the offsets in case the masked data changed
        xy = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
        self._offsets = xy
        self.stale = True

    def set_offsets(self, xy):
        """
        Set the offsets for the barb polygons.  This saves the offsets passed
        in and actually sets version masked as appropriate for the existing
        U/V data. *offsets* should be a sequence.

        ACCEPTS: sequence of pairs of floats
        """
        self.x = xy[:, 0]
        self.y = xy[:, 1]
        x, y, u, v = delete_masked_points(self.x.ravel(), self.y.ravel(),
                                          self.u, self.v)
        _check_consistent_shapes(x, y, u, v)
        xy = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
        mcollections.PolyCollection.set_offsets(self, xy)
        self.stale = True

    set_offsets.__doc__ = mcollections.PolyCollection.set_offsets.__doc__
