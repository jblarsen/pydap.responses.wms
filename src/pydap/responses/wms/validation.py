"""
Module for validating and converting query string input to the WMS service.
"""

from __future__ import division

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

try:
    from functools import reduce
except ImportError:
    pass
import re
import operator
import json
try:
    import cPickle as pickle
except ImportError:
    import pickle
import time
import calendar
from email.utils import parsedate
from datetime import datetime
try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse
try:
    from urllib.parse import unquote
except ImportError:
    from urllib import unquote
import numpy as np
from paste.request import construct_url, parse_dict_querystring
from webob.exc import HTTPBadRequest, HTTPNotModified
from paste.util.converters import asbool
import pyproj

from pydap.model import *
from pydap.lib import walk

from . import projutils
from . import gridutils

WMS_VERSION = '1.3.0'
WMS_ARGUMENTS = ['request', 'bbox', 'cmap', 'layers', 'width', 'height', 
                 'transparent', 'time', 'level', 'elevation', 'styles',
                 'service', 'version', 'format', 'crs', 'bounds',
                 'exceptions', 'bgcolor', 'expr', 'items', 'size']
MIN_WIDTH=0
MAX_WIDTH=8192
MIN_HEIGHT=0
MAX_HEIGHT=8192

DEFAULT_CRS = 'EPSG:4326'

SUPPORTED_CRS = ['EPSG:4326', 'EPSG:3857', 'EPSG:900913', 'EPSG:3785', 'EPSG:3395']
SUPPORTED_REQUESTS = ['GetMap', 'GetCapabilities', 'GetMetadata', 'GetColorbar']
SUPPORTED_FORMATS = ['image/png']
SUPPORTED_EXCEPTIONS = ['XML']

EXCEPTION_TEMPLATE="""<?xml version='1.0' encoding="UTF-8"?>
<ServiceExceptionReport version="1.3.0"
    xmlns="http://www.opengis.net/ogc"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:py="http://genshi.edgewall.org/"
    xsi:schemaLocation="http://www.opengis.net/ogc http://schemas.opengis.net/wms/1.3.0/exceptions_1_3_0.xsd">
    <py:choose test="error_code">
      <py:when test="None">
        <ServiceException>
          ${message} ${error_code}
        </ServiceException>
      </py:when>
      <py:otherwise>
        <ServiceException code="${error_code}">
          ${message}
        </ServiceException>
      </py:otherwise>
    </py:choose>
</ServiceExceptionReport>"""

ALLOWED_EXCEPTIONS = ['InvalidFormat', 'InvalidCRS', 'LayerNotDefined',
                      'StyleNotDefined', 'MissingDimensionValue',
                      'InvalidDimensionValue', 'OperationNotSupported']

class WMSException(Exception):
    def __init__(self, message, status_code, error_code=None):
        # Call the base class constructor with the parameters it needs
        super(WMSException, self).__init__(message)
        self.status_code = status_code
        self.error_code = error_code

def validate_wms(environ):
    """\
    Common validation for WMS calls.
    """
    query = parse_dict_querystring_lower(environ)

    # Check that REQUEST is present
    if 'request' not in query:
        msg = 'REQUEST not present in query string'
        raise WMSException(msg, 400)

    # Get REQUEST
    type_ = query.get('request')

    # Check that REQUEST is valid
    if type_ not in SUPPORTED_REQUESTS:
        msg = 'REQUEST=%s not supported; valid values=%s' % \
              (type_, SUPPORTED_REQUESTS)
        if type_ == 'GetFeatureInfo':
            error_code = 'OperationNotSupported'
            raise WMSException(msg, 400, error_code)
        else:
            raise WMSException(msg, 400)

    query_valid = {'request': type_}

    return query_valid

def validate_get_capabilities(environ):
    """\
    Validate GetCapabilities call.
    """
    # Input query string parameters
    query = parse_dict_querystring_lower(environ)

    required_args = ['service', 'request']
    for arg in required_args:
        if arg not in query:
            msg = '%s not present in query string' % arg.upper()
            raise WMSException(msg, 400)

    # Check that SERVICE=WMS
    service = query.get('service')
    if service != 'WMS':
        msg = 'SERVICE=WMS not present in query string'
        raise WMSException(msg, 400)

    # Validated output
    query_valid = {'service': service}

    return query_valid


def validate_get_map(environ, dataset, dataset_styles):
    """\
    Validate GetMap call.
    """
    # Validated output
    query_valid = {}

    query = parse_dict_querystring_lower(environ)

    required_args = ['version', 'request', 'layers', 'styles', 'crs', 'bbox',
                     'width', 'height', 'format']
    for arg in required_args:
        if arg not in query:
            msg = '%s not present in query string' % arg.upper()
            raise WMSException(msg, 400)

    # Check that VERSION is VERSION=1.3.0
    version = query.get('version')
    if version != WMS_VERSION:
        msg = 'VERSION=%s not supported; valid value=%s' % \
              (version, WMS_VERSION)
        raise WMSException(msg, 400)
    query_valid['version'] = version

    # Check that FORMAT is supported
    format_ = query.get('format')
    if format_ not in SUPPORTED_FORMATS:
        msg = 'FORMAT=%s not supported; valid values=%s' % \
              (format_, SUPPORTED_FORMATS)
        error_code = 'InvalidFormat'
        raise WMSException(msg, 400, error_code)
    query_valid['format'] = format_

    # Check WIDTH
    w = query.get('width')
    try:
        w = int(w)
    except ValueError:
        msg = 'WIDTH=%s not an integer' % w
        raise WMSException(msg, 400)
 
    if w < MIN_WIDTH or w > MAX_WIDTH:
        msg = 'WIDTH=%s must be in range [%i; %i]' % \
              (w, MIN_WIDTH, MAX_WIDTH)
        raise WMSException(msg, 400)
    query_valid['width'] = w

    # Check HEIGHT
    h = query.get('height')
    try:
        h = int(h)
    except ValueError:
        msg = 'HEIGHT=%s not an integer' % h
        raise WMSException(msg, 400)
    if h < MIN_HEIGHT or h > MAX_HEIGHT:
        msg = 'HEIGHT=%s must be in range [%i; %i]' % \
              (h, MIN_HEIGHT, MAX_HEIGHT)
        raise WMSException(msg, 400)
    query_valid['height'] = h

    # Check CRS
    crs = query.get('crs')
    if crs not in SUPPORTED_CRS:
        msg = 'CRS=%s not supported; valid values=%s' % \
              (crs, SUPPORTED_CRS)
        error_code = 'InvalidCRS'
        raise WMSException(msg, 400, error_code)
    # EPSG:900913 and EPSG:3785 are identical
    if crs == 'EPSG:900913':
        crs = 'EPSG:3785'
    query_valid['crs'] = crs

    # Check BBOX
    bbox_str = query.get('bbox')
    bbox = bbox_str.split(',')
    if len(bbox) != 4:
        msg = 'BBOX=%s must contain 4 comma separated values' % \
              (bbox_str)
        raise WMSException(msg, 400)
    
    try:
        bbox = [float(v) for v in bbox]
    except ValueError:
        msg = 'BBOX=%s does not contain 4 numeric values' % \
              (bbox_str)
        raise WMSException(msg, 400)

    # Reorder bounding box for EPSG:4326 which has lat/lon ordering
    if crs == 'EPSG:4326':
        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]

    query_valid['bbox'] = bbox


    # Check LAYERS and STYLES (basic check)
    layers_str = query.get('layers')
    layers = layers_str.split(',')
    styles_str = unquote(query.get('styles'))
    if len(styles_str) > 0:
        styles = styles_str.split(',')
    else:
        styles = len(layers)*['']
    if len(layers) != len(styles):
        msg = 'LAYERS=%s and STYLES=%s not compatible' % \
              (layers_str, styles_str)
        raise WMSException(msg, 400)

    # Check if layers and styles are defined
    defined_layers = build_layers(dataset, dataset_styles)
    layer_names = [l['name'] for l in defined_layers]
    defined_styles = {l['name']: l['styles'] for l in \
                      defined_layers}
    for layer, style in zip(layers, styles):
        defined_styles_layer = defined_styles[layer]
        if layer not in layer_names:
            msg = '%s in LAYERS=%s not defined; valid values=%s' \
                  % (layer, layers_str, defined_layers)
            error_code = 'LayerNotDefined'
            raise WMSException(msg, 400, error_code)
        if style != '' and style not in defined_styles:
            msg = '%s in STYLE=%s not defined; valid values=%s' \
                  % (style, styles_str, defined_styles_layer)
            error_code = 'StylesNotDefined'
            raise WMSException(msg, 400, error_code)

    query_valid['layers'] = layers
    query_valid['styles'] = styles

    # Check TRANSPARENT
    transparent = query.get('transparent', 'FALSE')
    if transparent not in ['TRUE', 'FALSE']:
        msg = 'TRANSPARENT=%s must be TRUE or FALSE' % \
              (transparent)
        raise WMSException(msg, 400)
    # Convert to bool
    transparent = asbool(transparent)
    query_valid['transparent'] = transparent

    # Check BGCOLOR (TODO: add check)
    bgcolor = query.get('bgcolor', '0xFFFFFF')
    query_valid['bgcolor'] = bgcolor

    # Check EXCEPTIONS
    exceptions = query.get('exceptions', 'XML')
    if exceptions not in SUPPORTED_EXCEPTIONS:
        msg = 'EXCEPTIONS=%s not supported; valid values=%s' % \
              (exceptions, SUPPORTED_EXCEPTIONS)
        raise WMSException(msg, 400)
    query_valid['exceptions'] = exceptions

    # Check TIME (TODO: add check)
    time = query.get('time', None)
    # Time; if time is None we will use the nearest timestep available
    if time == 'current': time = None
    query_valid['time'] = time

    # Check ELEVATION (TODO: add check)
    elevation = query.get('elevation', None)
    if elevation is not None:
        try:
            elevation = float(elevation)
        except ValueError:
            msg = 'ELEVATION=%s not a float' % elevation
            raise WMSException(msg, 400)
    query_valid['elevation'] = elevation

    return query_valid


def validate_get_colorbar(environ):
    """\
    Validate GetColorbar call.
    """
    query = parse_dict_querystring_lower(environ)

    required_args = []
    for arg in required_args:
        if arg not in query:
            msg = '%s not present in query string' % arg.upper()
            raise WMSException(msg, 400)

    # Check CMAP
    # TODO: check whether it is valid
    cmapname = query.get('cmap', None)

    # Check SIZE
    size = int(query.get('size', 200))
    min_size = max(MIN_WIDTH, MIN_HEIGHT)
    max_size = min(MAX_WIDTH, MAX_HEIGHT)
    if size < min_size or size > max_size:
        msg = 'SIZE=%s must be in range [%i; %i]' % \
              (size, min_size, max_size)
        raise WMSException(msg, 400)

    # Get more query string parameters
    styles = query.get('styles', 'vertical').split(',')
    if 'horizontal' in styles:
        orientation = 'horizontal'
    else:
        orientation = 'vertical'
    if 'noticks' in styles:
        add_ticks = False
    else:
        add_ticks = True
    if 'centerlabels' in styles:
        center_labels = True
    else:
        center_labels = False

    # TODO: return real values
    query_valid = {}
    return query_valid

def validate(environ, dataset, styles):
    """\
    Validates and converts query string given WSGI environ input.
    Returns a dict containing the converted parameters. Throws
    a WMSException if a validation error is encountered.
    """

    query = parse_dict_querystring_lower(environ)

    query_valid_master = validate_wms(environ)

    # Get REQUEST
    type_ = query.get('request')

    if type_ == 'GetCapabilities':
        query_valid = validate_get_capabilities(environ)
    elif type_ == 'GetMap':
        query_valid = validate_get_map(environ, dataset, styles)
    elif type_ == 'GetColorbar':
        query_valid = validate_get_colorbar(environ)
    elif type_ == 'GetMetadata':
        # TODO: Move GetMetadata validation here
        query_valid = {}
    else:
        msg = 'Internal Error'
        raise WMSException(msg, 500)

    query_valid_master.update(query_valid)
    return query_valid_master


def parse_dict_querystring_lower(environ):
    """Parses query string into dict with keys in lower case."""
    query = parse_dict_querystring(environ)
    # Convert WMS argument keys to lower case
    lowerlist = []
    for k,v in query.items():
        if k.lower() in WMS_ARGUMENTS:
            lowerlist.append(k)
    for k in lowerlist:
        v = query.pop(k)
        query[k.lower()] = v
    return query

def build_layers(dataset, supported_styles):
    grids = [grid for grid in walk(dataset, GridType) if \
            gridutils.is_valid(grid, dataset)]
    # Store information for regular layers
    layers = []
    for grid in grids:
        # Style information
        standard_name = grid.attributes.get('standard_name', None)
        styles = []
        if standard_name is not None:
            styles = supported_styles.get(standard_name, [])

        # Spatial information
        lon = np.asarray(gridutils.get_lon(grid, dataset)[:])
        lat = np.asarray(gridutils.get_lat(grid, dataset)[:])
        minx, maxx = np.min(lon), np.max(lon)
        miny, maxy = np.min(lat), np.max(lat)
        bbox = [minx, miny, maxx, maxy]

        # Vertical dimension
        z = gridutils.get_vertical(grid)
        dims = grid.dimensions
        if z is not None:
            if z.name not in dims:
                z = None

        # Time information
        time = gridutils.get_time(grid)

        layer = {
            'name': grid.name,
            'title': grid.attributes.get('long_name', grid.name),
            'abstract': grid.attributes.get('history', ''),
            'styles': styles,
            'bounding_box': bbox,
            'vertical': z,
            'time': time
        }
        layers.append(layer)
              
    # Find and store information for vector layers
    for u_grid in grids:
        u_standard_name = u_grid.attributes.get('standard_name', None)
        if u_standard_name.startswith('eastward_'):
            postfix = u_standard_name.split('_', 1)[1]
            standard_name = 'northward_' + postfix
            for v_grid in grids:
                v_standard_name = v_grid.attributes.get(
                                  'standard_name', None)
                if standard_name == v_standard_name:
                    styles = supported_styles.get(postfix, [])

                    # Spatial information
                    lon = gridutils.get_lon(u_grid, dataset)
                    lat = gridutils.get_lat(u_grid, dataset)
                    minx, maxx = np.min(lon), np.max(lon)
                    miny, maxy = np.min(lat), np.max(lat)
                    bbox = [minx, miny, maxx, maxy]

                    # Vertical dimension
                    z = gridutils.get_vertical(u_grid)
                    dims = u_grid.dimensions
                    if z is not None:
                        if z.name not in dims:
                            z = None

                    # Time information
                    time = gridutils.get_time(u_grid)

                    layer = {
                        'name': ':'.join([u_grid.name, v_grid.name]),
                        'title': postfix,
                        'abstract': grid.attributes.get('history', ''),
                        'styles': styles,
                        'bounding_box': bbox,
                        'vertical': z,
                        'time': time
                    }
                    layers.append(layer)

    return layers
