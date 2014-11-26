"""
Module containing projection related utilities.
"""

from __future__ import division

import numpy as np
import pyproj

def rotate_vector(p_source,p_target,uin,vin,lons,lats,returnxy=False):
    """
    Rotate a vector field (``uin,vin``) on a rectilinear grid
    with longitudes = ``lons`` and latitudes = ``lats`` from
    geographical (lat/lon) into map projection (x/y) coordinates.

    The vector is returned on the same grid, but rotated into
    x,y coordinates.

    The input vector field is defined in spherical coordinates (it
    has eastward and northward components) while the output
    vector field is rotated to map projection coordinates (relative
    to x and y). The magnitude of the vector is preserved.

    This method is more or less verbatim copied from matplotlib
    basemap.

    .. tabularcolumns:: |l|L|

    ==============   ====================================================
    Arguments        Description
    ==============   ====================================================
    uin, vin         input vector field on a lat/lon grid.
    lons, lats       Arrays containing longitudes and latitudes
                     (in degrees) of input data in increasing order.
                     For non-cylindrical projections (those other than
                     ``cyl``, ``merc``, ``gall`` and ``mill``) lons must
                     fit within range -180 to 180.
    ==============   ====================================================

    Returns ``uout, vout`` (rotated vector field).
    If the optional keyword argument
    ``returnxy`` is True (default is False),
    returns ``uout,vout,x,y`` (where ``x,y`` are the map projection
    coordinates of the grid defined by ``lons,lats``).
    """
    # if lons,lats are 1d and uin,vin are 2d, and
    # lats describes 1st dim of uin,vin, and
    # lons describes 2nd dim of uin,vin, make lons,lats 2d
    # with meshgrid.
    if lons.ndim == lats.ndim == 1 and uin.ndim == vin.ndim == 2 and\
       uin.shape[1] == vin.shape[1] == lons.shape[0] and\
       uin.shape[0] == vin.shape[0] == lats.shape[0]:
        lons, lats = np.meshgrid(lons, lats)
    else:
        if not lons.shape == lats.shape == uin.shape == vin.shape:
            raise TypeError("shapes of lons,lats and uin,vin don't match")
    x, y = pyproj.transform(p_source, p_target, lons, lats)
    # rotate from geographic to map coordinates.
    if np.ma.isMaskedArray(uin):
        mask = np.ma.getmaskarray(uin)
        masked = True
        uin = uin.filled(1)
        vin = vin.filled(1)
    else:
        masked = False

    # Map the (lon, lat) vector in the complex plane.
    uvc = uin + 1j*vin
    uvmag = np.abs(uvc)
    theta = np.angle(uvc)

    # Define a displacement (dlon, dlat) that moves all
    # positions (lons, lats) a small distance in the
    # direction of the original vector.
    dc = 1E-5 * np.exp(theta*1j)
    dlat = dc.imag * np.cos(np.radians(lats))
    dlon = dc.real

    # Deal with displacements that overshoot the North or South Pole.
    farnorth = np.abs(lats+dlat) >= 90.0
    somenorth = farnorth.any()
    if somenorth:
        dlon[farnorth] *= -1.0
        dlat[farnorth] *= -1.0

    # Add displacement to original location and find the native coordinates.
    lon1 = lons + dlon
    lat1 = lats + dlat
    xn, yn = pyproj.transform(p_source, p_target, lon1, lat1)

    # Determine the angle of the displacement in the native coordinates.
    vecangle = np.arctan2(yn-y, xn-x)
    if somenorth:
        vecangle[farnorth] += np.pi

    # Compute the x-y components of the original vector.
    uvcout = uvmag * np.exp(1j*vecangle)
    uout = uvcout.real
    vout = uvcout.imag

    if masked:
        uout = np.ma.array(uout, mask=mask)
        vout = np.ma.array(vout, mask=mask)
    if returnxy:
        return uout,vout,x,y
    else:
        return uout,vout

def project_data(p_source, p_target, bbox, lon, lat, cyclic):
    """\
    Project data and determine increment for going around globe. Input is
    assumed to be in EPSG:4326.

    Arguments:
    p_source -- Source pyproj.Proj instance
    p_target -- Source pyproj.Proj instance
    bbox -- input bounding box - used for normalizing output
    lon -- input longitude coordinates
    lat -- input latitude coordinates
    cyclic -- boolean true if global input coordinates

    Returns:
    x -- projection x coordinates
    y -- projection y coordinates
    dx -- delta for going around globe once in projection coordinates
    do_proj -- boolean true if output projection differs from EPSG:4326 
    """
    # Preconditions
    assert isinstance(p_source, pyproj.Proj)
    assert isinstance(p_target, pyproj.Proj)

    # Perform projection
    do_proj = p_source != p_target
    if do_proj:
        if len(lon.shape) == 1:
            lon, lat = np.meshgrid(lon, lat)
        dx = 2.0*pyproj.transform(p_source, p_target, 180.0, 0.0)[0]
        x, y = pyproj.transform(p_source, p_target, lon, lat)
        # Special handling when request crosses discontinuity
        if bbox[0] > bbox[2]:
        #if bbox[0] > bbox[2] or cyclic:
            x = np.where(x >= 0.0, x, x+dx)
    else:
        x, y = lon, lat
        dx = 360.0
    while np.min(x) > bbox[0]:
        x -= dx
    # Projections can result in inf values - mask them out
    #x = np.ma.masked_invalid(x)
    #y = np.ma.masked_invalid(y)
    # The above is really not a good idea since we should not present data in
    # projections that go towards infinity where they have data
    return x, y, dx, do_proj
