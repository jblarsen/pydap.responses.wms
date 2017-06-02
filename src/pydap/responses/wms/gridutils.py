from __future__ import division

import datetime
import re
import operator

import numpy as np
import iso8601
import netcdftime

from pydap.model import GridType
from pydap.lib import walk

def is_valid(grid, dataset):
    return (get_lon(grid, dataset) is not None and 
            get_lat(grid, dataset) is not None)

def get_lon(grid, dataset):
    def check_attrs(var):
        if (re.match('degrees?_e', var.attributes.get('units', ''), re.IGNORECASE) or
            var.attributes.get('standard_name', '') == 'longitude'):
            return np.asarray(var[:])
        #if (re.match('degrees?_e', var.attributes.get('units', ''), re.IGNORECASE) or
        #    var.attributes.get('axis', '').lower() == 'x' or
        #    var.attributes.get('standard_name', '') == 'longitude'):
        #    return var

    # check maps first
    for dim in grid.maps.values():
        if check_attrs(dim) is not None:
            return np.asarray(dim[:])

    # check curvilinear grids
    if hasattr(grid, 'coordinates'):
        coords = grid.coordinates.split()
        for coord in coords:
            if coord in dataset and check_attrs(dataset[coord].array) is not None:
                return np.asarray(dataset[coord].array[:])

    return None

def get_lat(grid, dataset):
    def check_attrs(var):
        #if (re.match('degrees?_n', var.attributes.get('units', ''), re.IGNORECASE) or
        #    var.attributes.get('axis', '').lower() == 'y' or
        #    var.attributes.get('standard_name', '') == 'latitude'):
        #    return var
        if (re.match('degrees?_n', var.attributes.get('units', ''), re.IGNORECASE) or
            var.attributes.get('standard_name', '') == 'latitude'):
            return np.asarray(var[:])

    # check maps first
    for dim in grid.maps.values():
        if check_attrs(dim) is not None:
            return np.asarray(dim[:])

    # check curvilinear grids
    if hasattr(grid, 'coordinates'):
        coords = grid.coordinates.split()
        for coord in coords:
            if coord in dataset and check_attrs(dataset[coord].array) is not None:
                return np.asarray(dataset[coord].array[:])

    return None

def get_time(grid):
    for dim in grid.maps.values():
        if ' since ' in dim.attributes.get('units', ''):
            calendar = dim.attributes.get('calendar', 'standard')
            try:
                utime = netcdftime.utime(dim.units, calendar=calendar)
                tm = utime.num2date(np.asarray(dim[:]))
                # Discard microseconds
                for i in range(len(tm)):
                    tm[i] = tm[i].replace(microsecond=0)
                return tm
            except:
                pass

    return None

def get_vertical(grid):
    for dim in grid.maps.values():
        if isvertical(dim):
            return np.asarray(dim[:])

    return None

def isvertical(dim):
    """Returns True if input variable is a vertical variable."""
    upressure = ['bar', 'atmosphere', 'atm', 'pascal', 'pa', 'hpa']
    #uother = ['meter', 'metre', 'm', 'kilometer', 'km', 'level', 'layer', \
    #           'sigma_level']
    u = getattr(dim, 'units', '')
    if hasattr(dim, 'positive') or u.lower() in upressure:
        return True
    # GETM hack
    if u.lower() == 'level':
        return True
    return False

#@profile
def fix_data(data, attrs):
    if len(data.shape) > 2 and data.shape[0] == 1:
        data = np.asarray(data)[0]
    if 'missing_value' in attrs:
        data = np.ma.masked_equal(data, attrs['missing_value'])
    elif '_FillValue' in attrs:
        data = np.ma.masked_equal(data, attrs['_FillValue'])

    # When scale_factor or add_offset are present we always assume float array
    if attrs.get('scale_factor'):
        data = data.astype(np.float32)
        data *= attrs['scale_factor']
    if attrs.get('add_offset'):
        data = data.astype(np.float32)
        data += attrs['add_offset']

    while len(data.shape) > 2:
        if data.shape[0] != 1:
            ##data = data[0]
            data = np.ma.mean(data, 0)
        else:
            data = data[0]
    return data

def fix_map_attributes(dataset):
    for grid in walk(dataset, GridType):
        for map_ in grid.maps.values():
            if not map_.attributes and map_.name in dataset:
                map_.attributes = dataset[map_.name].attributes.copy()

def find_containing_bounds(axis, v0, v1):
    """
    Find i0, i1 such that axis[i0:i1] is the minimal array with v0 and v1.

    For example::

        >>> from numpy import *
        >>> a = arange(10)
        >>> i0, i1 = find_containing_bounds(a, 1.5, 6.5)
        >>> print a[i0:i1]
        [1 2 3 4 5 6 7]
        >>> i0, i1 = find_containing_bounds(a, 1, 6)
        >>> print a[i0:i1]
        [1 2 3 4 5 6]
        >>> i0, i1 = find_containing_bounds(a, 4, 12)
        >>> print a[i0:i1]
        [4 5 6 7 8 9]
        >>> i0, i1 = find_containing_bounds(a, 4.5, 12)
        >>> print a[i0:i1]
        [4 5 6 7 8 9]
        >>> i0, i1 = find_containing_bounds(a, -4, 7)
        >>> print a[i0:i1]
        [0 1 2 3 4 5 6 7]
        >>> i0, i1 = find_containing_bounds(a, -4, 12)
        >>> print a[i0:i1]
        [0 1 2 3 4 5 6 7 8 9]
        >>> i0, i1 = find_containing_bounds(a, 12, 19)
        >>> print a[i0:i1]
        []

    It also works with decreasing axes::

        >>> b = a[::-1]
        >>> i0, i1 = find_containing_bounds(b, 1.5, 6.5)
        >>> print b[i0:i1]
        [7 6 5 4 3 2 1]
        >>> i0, i1 = find_containing_bounds(b, 1, 6)
        >>> print b[i0:i1]
        [6 5 4 3 2 1]
        >>> i0, i1 = find_containing_bounds(b, 4, 12)
        >>> print b[i0:i1]
        [9 8 7 6 5 4]
        >>> i0, i1 = find_containing_bounds(b, 4.5, 12)
        >>> print b[i0:i1]
        [9 8 7 6 5 4]
        >>> i0, i1 = find_containing_bounds(b, -4, 7)
        >>> print b[i0:i1]
        [7 6 5 4 3 2 1 0]
        >>> i0, i1 = find_containing_bounds(b, -4, 12)
        >>> print b[i0:i1]
        [9 8 7 6 5 4 3 2 1 0]
        >>> i0, i1 = find_containing_bounds(b, 12, 19)
        >>> print b[i0:i1]
        []
    """
    ascending = axis[1] > axis[0]
    if not ascending: axis = axis[::-1]
    i0 = i1 = len(axis)
    for i, value in enumerate(axis):
        if value > v0 and i0 == len(axis):
            i0 = i-1
        if not v1 > value and i1 == len(axis):
            i1 = i+1
    if not ascending: i0, i1 = len(axis)-i1, len(axis)-i0
    return max(0, i0), min(len(axis), i1)

def time_slice(time, grid, dataset):
    """\
    Slice according to time request (WMS-T). Since it is very expensive to
    convert the entire grid time array to datetime objects we do the opposite
    and convert the requested time steps to the unit used in the grid time
    array.

    If input time is None the nearest timestep is returned.
    """
    if time is None:
        find_nearest = True
    else:
        find_nearest = False

    for dim in grid.maps.values():
        if ' since ' in dim.attributes.get('units', ''):
            calendar = dim.attributes.get('calendar', 'standard')
            try:
                values = np.array(np.asarray(dim[Ellipsis]))
                break
            except:
                pass
    if len(values.shape) == 0:
        # Input field has no time dimension
        l = Ellipsis
    else:
        l = np.zeros(values.shape, bool)  # get no data by default
        if time is None:
            time = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        tokens = time.split(',')
        for token in tokens:
            if '/' in token: # range
                start, end = token.strip().split('/')
                start = iso8601.parse_date(start, default_timezone=None)
                end = iso8601.parse_date(end, default_timezone=None)
                l[(values >= start) & (values <= end)] = True
                # Convert to index (we do not support multiple timesteps)
                l = np.where(l == True)[0][0]
            else:
                instant = iso8601.parse_date(token.strip().rstrip('Z'), default_timezone=None)
                utime = netcdftime.utime(dim.units, calendar=calendar)
                instant = utime.date2num(instant)
                # TODO: Calculate index directly
                epoch = values[0]
                values = values - epoch
                instant = instant - epoch
                if not find_nearest:
                    # Require almost exact match
                    l = np.isclose(values, instant)
                    # Convert array to index
                    l = int(np.where(l == True)[0][0])
                else:
                    l = int((np.abs(values-instant)).argmin())
    return l
