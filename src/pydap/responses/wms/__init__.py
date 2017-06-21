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
from paste.request import construct_url, parse_dict_querystring
from webob import Response
from webob.dec import wsgify
from webob.exc import HTTPBadRequest, HTTPNotModified
from paste.util.converters import asbool
import numpy as np
from scipy import interpolate
from scipy.spatial import ConvexHull
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib import rcParams
from matplotlib.colors import from_levels_and_colors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.pyplot import setp
from matplotlib.patheffects import withStroke
import matplotlib.patheffects as PathEffects

rcParams['xtick.labelsize'] = 'small'
rcParams['ytick.labelsize'] = 'small'
rcParams['contour.negative_linestyle'] = 'solid'
import pyproj
from lru import LRU

from pydap.model import *
from pydap.exceptions import ServerError
from pydap.responses.lib import BaseResponse
from pydap.util.template import GenshiRenderer, StringLoader, TemplateNotFound
from pydap.lib import walk
try:
    from asteval_restricted import RestrictedInterpreter
except:
    RestrictedInterpreter = None

# Local imports
from .arrowbarbs import arrow_barbs
from . import projutils
from . import gridutils
from . import plotutils
from . import validation

# Setup caching
from dogpile.cache import make_region
from dogpile.cache.api import NO_VALUE

WMS_ARGUMENTS = ['request', 'bbox', 'cmap', 'layers', 'width', 'height', 
                 'transparent', 'time', 'level', 'elevation', 'styles',
                 'service', 'version', 'format', 'crs', 'bounds',
                 'exceptions', 'bgcolor', 'expr', 'items', 'size']
MIN_WIDTH=0
MAX_WIDTH=8192
MIN_HEIGHT=0
MAX_HEIGHT=8192
DEFAULT_CRS = 'EPSG:4326'
WMS_VERSION = '1.3.0'
SUPPORTED_CRS = ['EPSG:4326', 'EPSG:3857', 'EPSG:900913', 'EPSG:3785', 'EPSG:3395']

DEFAULT_TEMPLATE = """<?xml version='1.0' encoding="UTF-8"?>
<WMS_Capabilities version="1.3.0" xmlns="http://www.opengis.net/wms"
  xmlns:xlink="http://www.w3.org/1999/xlink"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns:py="http://genshi.edgewall.org/"
  xsi:schemaLocation="http://www.opengis.net/wms http://schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd">

<Service>
  <Name>WMS</Name>
  <Title>WMS server ${dataset.name} for ${dataset.attributes.get('long_name', dataset.name)}</Title>
  <OnlineResource xlink:href="${location + '.html'}"></OnlineResource>
  <MaxWidth>${max_width}</MaxWidth>
  <MaxHeight>${max_height}</MaxHeight>
</Service>

<Capability>
  <Request>
    <GetCapabilities>
      <Format>text/xml</Format>
      <DCPType>
        <HTTP>
          <Get><OnlineResource xlink:href="${location + '.wms'}"></OnlineResource></Get>
        </HTTP>
      </DCPType>
    </GetCapabilities>

    <GetMap>
      <Format>image/png</Format>
      <DCPType>
        <HTTP>
          <Get><OnlineResource xlink:href="${location + '.wms'}"></OnlineResource></Get>
        </HTTP>
      </DCPType>
    </GetMap>
  </Request>
  <Exception>
    <Format>XML</Format>
  </Exception>
  <Layer>
    <Title>WMS server for ${dataset.attributes.get('long_name', dataset.name)}</Title>
    <CRS py:for="crs in supported_crs">${crs}</CRS>
    <EX_GeographicBoundingBox>
      <westBoundLongitude>${lon_range[0]}</westBoundLongitude>
      <eastBoundLongitude>${lon_range[1]}</eastBoundLongitude>
      <southBoundLatitude>${lat_range[0]}</southBoundLatitude>
      <northBoundLatitude>${lat_range[1]}</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="${default_crs}" minx="${lat_range[0]}" miny="${lon_range[0]}" maxx="${lat_range[1]}" maxy="${lon_range[1]}"/>
    <Layer py:for="layer in layers">
      <?python
          import datetime
          import numpy as np

          # Unpack bounding box
          minx, miny, maxx, maxy = layer['bounding_box']

          # Time information
          time = layer['time']
          now = datetime.datetime.utcnow()
          epoch = datetime.datetime(1970, 1, 1)
          now_secs = (now - epoch).total_seconds()
          if time is not None:
              time_secs = np.array([(t-epoch).total_seconds() for t in time])
              idx_nearest = np.abs(time_secs-now_secs).argmin()

          # Vertical coordinate variable
          z = layer['vertical']
      ?>

      <Name>${layer['name']}</Name>
      <Title>${layer['title']}</Title>
      <Abstract>${layer['abstract']}</Abstract>
      <EX_GeographicBoundingBox>
        <westBoundLongitude>${minx}</westBoundLongitude>
        <eastBoundLongitude>${maxx}</eastBoundLongitude>
        <southBoundLatitude>${miny}</southBoundLatitude>
        <northBoundLatitude>${maxy}</northBoundLatitude>
      </EX_GeographicBoundingBox>
      <BoundingBox CRS="${default_crs}" minx="${minx}" miny="${miny}" maxx="${maxx}" maxy="${maxy}"/>
      <Dimension py:if="time is not None" name="time" units="ISO8601" default="${time[idx_nearest].isoformat()}" nearestValue="0">
        ${','.join([t.isoformat() for t in time])}
      </Dimension>
      <Dimension py:if="z is not None" name="elevation" units="${z.attributes.get('units', '')}" default="${np.asarray(z[:])[0]}" multipleValues="0" nearestValue="0">
        ${','.join([str(zz) for zz in np.asarray(z[:])])}
      </Dimension>
      <Style py:for="style in layer['styles']">
        <?python
            style_list = []
            for key, value in style.items():
                item = key + '%3D' + value
                style_list.append(item)
            style_str = '%3B'.join(style_list)
        ?>
        <Name>${style_str}</Name>
        <Title>Style ${style_str} for ${layer['name']}</Title>
        <LegendURL width="250" height="40">
          <Format>image/png</Format>
          <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink"
          xlink:type="simple"
          xlink:href="${location}.wms?REQUEST=GetColorbar&amp;STYLES=horizontal%2Cnolabel&amp;CMAP=${style['legend']}&amp;SIZE=250" />
        </LegendURL>
      </Style>
    </Layer>
  </Layer>
</Capability>
</WMS_Capabilities>"""


class OutsideGridException(Exception):
    pass


class WMSResponse(BaseResponse):
    __description__ = "Web Map Service 1.3.0"

    renderer = GenshiRenderer(
            options={}, loader=StringLoader( {'capabilities.xml': DEFAULT_TEMPLATE} ))

    exception_renderer = GenshiRenderer(
            options={}, loader=StringLoader( {'exceptions.xml': validation.EXCEPTION_TEMPLATE} ))

    def __init__(self, dataset):
        BaseResponse.__init__(self, dataset)
        self.headers.append( ('Content-Description', 'dods_wms') )

    #@profile
    @wsgify
    def __call__(self, req):
        environ = req.environ

        # Init colors and styles class instances on first invokation
        config = environ.get('pydap.config', {})
        self._init_colors(config)
        self._init_styles(config)
    
        # Init redis cache on first invokation
        self._init_cache(config)
    
        query = parse_dict_querystring_lower(environ)
        try:
            query_valid = validation.validate(environ, self.dataset,
                                              self.styles)
            # Get REQUEST
            type_ = query_valid.get('request')

            try:
                dap_query = ['%s=%s' % (k, query[k]) for k in query
                        if k.lower() not in WMS_ARGUMENTS]
                dap_query = [pair.rstrip('=') for pair in dap_query]
                dap_query.sort()  # sort for uniqueness
                dap_query = '&'.join(dap_query)
                self.location = construct_url(environ,
                        with_query_string=True,
                        querystring=dap_query)
                # Since we might use different aliases for this host we only use
                # the path component to identify the dataset
                self.path = urlparse(self.location).path
                    
                # Create a Beaker cache dependent on the query string for
                # pre-computed values that depend on the specific dataset
                # We exclude all WMS related arguments since they don't
                # affect the dataset.
                self.cache = self.cacheRegion
            except KeyError:
                self.cache = None
    
            # Support cross-origin resource sharing (CORS)
            self.headers.append( ('Access-Control-Allow-Origin', '*') )
    
            # Handle GetMap, GetColorbar, GetCapabilities and GetMetadata requests
            if type_ == 'GetCapabilities':
                content = self._get_capabilities(environ)
                self.headers.append( ('Content-Type', 'text/xml') )
            elif type_ == 'GetMetadata':
                content = self._get_metadata(environ)
                self.headers.append( ('Content-Type', 'application/json; charset=utf-8') )
            elif type_ == 'GetMap':
                content = self._get_map(environ)
                self.headers.append( ('Content-Type', 'image/png') )
            elif type_ == 'GetColorbar':
                content = self._get_colorbar(environ)
                self.headers.append( ('Content-Type', 'image/png') )
            else:
                raise HTTPBadRequest('Invalid REQUEST "%s"' % type_)
    
            # The GetCapabilities default time response changes with time
            # and it is therefore not cached
            if type_ != 'GetCapabilities':
                # Check if we should return a 304 Not Modified response
                # This check has been disabled since the required information
                # is not available in the environ variable in pydap 3.2.0
                #self._check_last_modified(environ)
    
                # Set response caching headers
                self._set_caching_headers(environ)

        except validation.WMSException as err:
            # Remove existing Content-Type header and insert text/xml
            for header in self.headers:
                if header[0].lower() == 'content-type':
                    self.headers.remove(header)
            self.headers.append( ('Content-Type', 'text/xml') )

            # Render XML response
            context = {
                'message': str(err),
                'error_code': err.error_code
            }
            try:
                renderer = environ['pydap.exception_renderer']
                template = renderer.loader('exceptions.xml')
            except (KeyError, TemplateNotFound):
                renderer = self.exception_renderer
                template = renderer.loader('exceptions.xml')
            output = renderer.render(template, context, output_format='text/xml')
            output = output.encode('utf-8')

            # Return WSGI response object
            return Response(body=output, status=err.status_code,
                             headerlist=self.headers)

        return Response(body=content[0], headers=self.headers)

    def _check_last_modified(self, environ):
        """Check if resource is modified since last_modified header value."""
        if 'HTTP_IF_MODIFIED_SINCE' in environ:
            cache_control = environ.get('HTTP_CACHE_CONTROL')
            if cache_control != 'no-cache':
                if_modified_since = calendar.timegm(parsedate(environ.get('HTTP_IF_MODIFIED_SINCE')))
                for header in environ['pydap.headers']:
                    if 'Last-modified' in header:
                        last_modified = calendar.timegm(parsedate(header[1]))
                        if if_modified_since <= last_modified:
                            raise HTTPNotModified

    def _set_caching_headers(self, config):
        """Set caching headers"""
        max_age = config.get('pydap.responses.wms.max_age', None)
        s_maxage = config.get('pydap.responses.wms.s_maxage', None)
        cc_str = 'public'
        if max_age is not None:
            cc_str = cc_str + ', max-age=%i' % int(max_age)
        if s_maxage is not None:
            cc_str = cc_str + ', s-maxage=%i' % int(s_maxage)
        if max_age is not None or s_maxage is not None:
            self.headers.append( ('Cache-Control', cc_str) )

    def _init_colors(self, config):
        """Set class variable "colors" on first call"""
        if not hasattr(WMSResponse, 'colors'):
            colorfile = config.get('pydap.responses.wms.colorfile', None)
            if colorfile is None:
                WMSResponse.colors = {}
            else:
                with open(colorfile, 'r') as file:
                    colors = json.load(file)
                    for cname in colors:
                        levs = colors[cname]['levels']
                        cols = colors[cname]['colors']
                        extend = colors[cname]['extend']
                        cmap, norm = from_levels_and_colors(levs, cols, extend)
                        colors[cname] = {'cmap': cmap, 'norm': norm}
                    del levs, cols, extend, cmap, norm, cname
                    WMSResponse.colors = colors
                del colorfile

    def _init_styles(self, config):
        """Set class variable "styles" on first call"""
        if not hasattr(WMSResponse, 'styles'):
            styles_file = config.get('pydap.responses.wms.styles_file', None)
            if styles_file is None:
                WMSResponse.styles = {}
            else:
                with open(styles_file, 'r') as file:
                    WMSResponse.styles = json.load(file)
                del styles_file

    def _init_cache(self, config):
        """Set class variable "cacheRegion" on first call"""
        if not hasattr(WMSResponse, 'cacheRegion'):
            redis = asbool(config.get('pydap.responses.wms.redis', 'false'))
            if not redis:
                WMSResponse.cacheRegion = None
            else:
                WMSResponse.cacheRegion = make_region().configure(
                    'dogpile.cache.redis',
                    arguments = {
                        'host': config.get('pydap.responses.wms.redis.host', 'localhost'),
                        'port': int(config.get('pydap.responses.wms.redis.port', 6379)),
                        'db': int(config.get('pydap.responses.wms.redis.db', 0)),
                        'redis_expiration_time': int(config.get('pydap.responses.wms.redis.redis_expiration_time', 604800)),
                        'distributed_lock': asbool(config.get('pydap.responses.wms.redis.distributed_lock', 'false'))
                    }
                )
        # Setup local caches for this process
        if not hasattr(WMSResponse, 'localCache'):
            WMSResponse.localCache = {}
            WMSResponse.localCache['figures'] = LRU(1000)
            WMSResponse.localCache['pyproj'] = LRU(1000)
            WMSResponse.localCache['project_data'] = LRU(5)

    def _get_colorbar(self, environ):
        # Get query string parameters
        query = parse_dict_querystring_lower(environ)

        # Get pydap configuration
        config = environ.get('pydap.config', {})

        # Get colorbar name and check whether it is valid
        cmapname = query.get('cmap', config.get('pydap.responses.wms.cmap', 'jet'))
        self._get_cmap(cmapname)

        # Scale DPI according to size
        size = int(query.get('size', 0))
        if size < 25 or size > min(MAX_WIDTH, MAX_HEIGHT):
            msg = 'Image size must be in range [25; %d]' \
                  % min(MAX_WIDTH, MAX_HEIGHT)
            raise HTTPBadRequest(msg)

        def serialize(dataset):
            # Check for cached version of colorbar
            if self.cache:
                try:
                    output = self.cache.get(('colorbar', cmapname))
                    if output is NO_VALUE:
                        raise KeyError
                    if hasattr(dataset, 'close'): dataset.close()
                    return [output]
                except KeyError:
                    pass

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

            transparent = asbool(query.get('transparent', 'true'))

            # Get WMS settings
            dpi = float(config.get('pydap.responses.wms.dpi', 80))
            paletted = asbool(config.get('pydap.responses.wms.paletted', 'false'))

            # Set color bar size depending on orientation
            w, h = 100, 300
            if orientation == 'horizontal':
                w, h = 500, 80

            if size != 0:
                wh_max = max(w, h)
                dpi = size/wh_max*dpi
                w = int(size/wh_max*w)
                h = int(size/wh_max*h)

            gridutils.fix_map_attributes(dataset)

            # Plot requested grids.
            layers = [layer for layer in query.get('layers', '').split(',')
                    if layer] or [var.id for var in walk(dataset, GridType)]
            layer = layers[0]
            names = [dataset] + layer.split('.')
            grid = reduce(operator.getitem, names)

            norm, cmap, extend = self._get_colors(cmapname, grid)

            output = plotutils.make_colorbar(w, h, dpi, grid, orientation, 
                     transparent, norm, cmap, extend, paletted, add_ticks,
                     center_labels)

            if hasattr(dataset, 'close'): dataset.close()
            output = output.getvalue()
            # FIXME: We should use all parameters for caching
            if self.cache:
                key = ('colorbar', cmapname)
                #self.cache.set(key, output)

            return [output]

        return serialize(self.dataset)

    def _get_colors(self, cmapname, grid):
        """Returns Normalization, Colormap and extend information."""
        if cmapname in self.colors:
            norm = self.colors[cmapname]['norm']
            cmap = self.colors[cmapname]['cmap']
            extend = cmap.colorbar_extend 
        else:
            actual_range = self._get_actual_range(grid)
            norm = Normalize(vmin=actual_range[0], vmax=actual_range[1])
            cmap = self._get_cmap(cmapname)
            extend = 'neither'
        return norm, cmap, extend

    def _get_actual_range(self, grid):
        try:
            # TODO: Use file time stamp to invalidate cache
            actual_range = self.cache.get((grid.id, 'actual_range'))
            if actual_range is NO_VALUE:
                raise KeyError
        except (KeyError, AttributeError):
            try:
                actual_range = grid.attributes['actual_range']
            except KeyError:
                data = gridutils.fix_data(np.asarray(grid.array[:]), grid.attributes)
                actual_range = np.min(data), np.max(data)
            if self.cache:
                key = (grid.id, 'actual_range')
                self.cache.set(key, actual_range)
        return actual_range

    def _get_cmap(self, name):
        """Tries to get local colormap first and then a mpl colormap."""
        try:
            cmap = self.colors[name]
        except KeyError:
            try:
                cmap = get_cmap(name)
            except ValueError as e:
                raise HTTPBadRequest('Colormap Error: %s' % e.message)
        return cmap

    #@profile
    def _get_map(self, environ):
        def prep_map(environ):
            query = parse_dict_querystring_lower(environ)

            # Get pydap configuration
            config = environ.get('pydap.config', {})

            # TODO: Validate requested media type is image/png; throw InvalidFormat

            # Allow basic mathematical operations on layers
            allow_eval = asbool(config.get('pydap.responses.wms.allow_eval', 'false'))

            # Image resolution and size
            dpi = float(config.get('pydap.responses.wms.dpi', 80))
            w = float(query.get('width', 256))
            h = float(query.get('height', 256))
            if w > MAX_WIDTH or h > MAX_HEIGHT:
                msg = 'Image (width, height) must be less than or equal to: (%d, %d)' \
                      % (MAX_WIDTH, MAX_HEIGHT)
                raise HTTPBadRequest(msg)
            figsize = w/dpi, h/dpi

            # Time; if time is None we will use the nearest timestep available
            time = query.get('time', None)
            if time == 'current': time = None

            # Vertical level
            elevation = query.get('elevation', None)
            if elevation is not None:
                elevation = float(elevation)

            # We support integer levels a little while longer
            level = int(query.get('level', 0))

            # Bounding box in projection coordinates
            bbox = query.get('bbox', None)
            if bbox is not None:
                bbox = [float(v) for v in bbox.split(',')]

            # Projection
            crs = query.get('crs', DEFAULT_CRS)
            if crs == 'EPSG:900913': crs = 'EPSG:3785'

            # Reorder bounding box for EPSG:4326 which has lat/lon ordering
            if crs == 'EPSG:4326':
                bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]

            # Default vector settings
            vector_spacing = int(\
                config.get('pydap.responses.wms.pixels_between_vectors', 40))
            vector_offset = 0
            vector_color = 'k' 

            # Construct layer list
            layers = [layer for layer in query.get('layers', '').split(',')
                    if layer] or [var.id for var in walk(self.dataset, GridType)]

            # Check if layers are present in file and that the grid is valid
            for layer in layers:
                hlayers = layer.split('.')
                vlayers = hlayers[-1].split(':')
                for vlayer in vlayers:
                    names = [self.dataset] + hlayers[:-2] + [vlayer]
                    try:
                        grid = reduce(operator.getitem, names)
                    except KeyError as error:
                        raise HTTPBadRequest('Nonexistent LAYER "%s"' % error.message)
                    if not gridutils.is_valid(grid, self.dataset):
                        raise HTTPBadRequest('Invalid LAYERS "%s"' % layers)

            # Process styles element
            styles = query.get('styles')
            styles = unquote(styles)
            if len(styles) > 0:
                styles = styles.split(',')
            else:
                styles = len(layers)*['']

            # Defaults
            legend = config.get('pydap.responses.wms.cmap', 'jet')
            plot_method = 'contourf'

            # FIXME: We currently only support one legend and other style params.
            # Change to dict of layer items instead
            for style in styles:
                found_legend = False
                if len(style) > 0:
                    style_items = style.split(';')
                    for style_item in style_items:
                        key, value = style_item.split('=')
                        if key == 'plot_method':
                            if value in ['contour', 'contourf', 
                                         'pcolor', 'pcolormesh',
                                         'pcolorfast',
                                         'black_vector', 'black_quiver', 
                                         'black_barbs', 'black_arrowbarbs',
                                         'color_quiver1', 'color_quiver2',
                                         'color_quiver3', 'color_quiver4']:
                                plot_method = value
                        elif key == 'vector_color':
                            vector_color = value
                        elif key == 'vector_spacing':
                            vector_spacing = int(value)
                        elif key == 'vector_offset':
                            vector_offset = int(value)
                        elif key == 'legend':
                            #legend = unicode(value, "utf-8")
                            legend = value.strip()
                            found_legend = True
                if not found_legend:
                    # Use defaults
                    # FIXME: Change to use dict of layers
                    attrs = grid.attributes
                    if 'standard_name' in attrs and \
                        attrs['standard_name'] in self.styles:
                        legend = self.styles[attrs['standard_name']][0]['legend']

            # Check whether legend is valid by getting it
            self._get_cmap(legend)

            if len(styles) != len(layers):
                msg = 'STYLES and LAYERS must contain the same number of ' \
                      'elements (STYLE can also be empty)'
                raise HTTPBadRequest(msg)
             
            return query, dpi, plot_method, vector_color, time, \
                   elevation, level, figsize, bbox, legend, crs, styles, w, \
                   h, vector_spacing, vector_offset, allow_eval, \
                   layers

        query, dpi, plot_method, vector_color, time, elevation, \
            level, figsize, bbox, legend, crs, styles, w, h, \
            vector_spacing, vector_offset, allow_eval, layers \
            = prep_map(environ)

        #@profile
        def serialize(dataset):
            gridutils.fix_map_attributes(dataset)
            # It is apparently expensive to add an axes in matplotlib - so we cache the
            # axes (it is pickled so it is a threadsafe copy). It is safe to store
            # this in the cache forever. We maintain two cache levels. The fastests
            # is local to this process and is the "figures" dict attribute on
            # the WMSResponse class while the second level is in the redis cache
            
            if 'figures' in self.localCache and figsize in self.localCache['figures'].keys():
                fig = pickle.loads(self.localCache['figures'][figsize])
                ax = fig.get_axes()[0]
            else:
                try:
                    if not self.cache:
                        raise KeyError
                    fig = self.cache.get((figsize, 'figure'))
                    if fig is NO_VALUE:
                        raise KeyError
                    ax = fig.get_axes()[0]
                except KeyError:
                    fig = Figure(figsize=figsize, dpi=dpi)
                    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
                    if self.cache:
                        key = (figsize, 'figure')
                        self.cache.set(key, fig)
            if figsize not in self.localCache['figures'].keys():
                self.localCache['figures'][figsize] = pickle.dumps(fig, -1)

            # Set transparent background; found through http://sparkplot.org/browser/sparkplot.py.
            if asbool(query.get('transparent', 'true')):
                fig.figurePatch.set_alpha(0.0)
                ax.axesPatch.set_alpha(0.0)
            
            # Per default we only quantize fields if they have less than 
            # 255 colors.
            ncolors = None

            if allow_eval:
                expr = query.get('expr', None)
            else:
                expr = None

            for layer in layers:
                hlayers = layer.split('.')
                vlayers = hlayers[-1].split(':')
                if expr is not None:
                    # Plot expression
                    grids = []
                    for vlayer in vlayers:
                        names = [dataset] + hlayers[:-2] + [vlayer]
                        grid = reduce(operator.getitem, names)
                        if not gridutils.is_valid(grid, dataset):
                            raise HTTPBadRequest('Invalid LAYERS "%s"' % layers)
                        if bbox is None:
                            lon = gridutils.get_lon(grid, dataset)
                            lat = gridutils.get_lat(grid, dataset)
                            bbox_local = [np.min(lon), np.min(lat), np.max(lon), np.max(lat)]
                        else:
                            bbox_local = bbox[:]
                        grids.append(grid)
                    self._plot_grid(dataset, grids, time, elevation, level, bbox_local, 
                                    (w, h), ax, crs, plot_method, 
                                    legend, expr=expr, layers=vlayers)
                elif len(vlayers) == 1:
                    # Plot scalar field
                    names = [dataset] + hlayers
                    grid = reduce(operator.getitem, names)
                    if gridutils.is_valid(grid, dataset):
                        if bbox is None:
                            lon = gridutils.get_lon(grid, dataset)
                            lat = gridutils.get_lat(grid, dataset)
                            bbox_local = [np.min(lon), np.min(lat), np.max(lon), np.max(lat)]
                        else:
                            bbox_local = bbox[:]
                        self._plot_grid(dataset, grid, time, elevation, level,
                                        bbox_local, (w, h), ax, crs,
                                        plot_method, legend)
                elif len(vlayers) == 2:
                    # Plot vector field
                    grids = []
                    for vlayer in vlayers:
                        names = [dataset] + hlayers[:-2] + [vlayer]
                        grid = reduce(operator.getitem, names)
                        if not gridutils.is_valid(grid, dataset):
                            raise HTTPBadRequest('Invalid LAYERS "%s"' % layers)
                        if bbox is None:
                            lon = gridutils.get_lon(grid, dataset)
                            lat = gridutils.get_lat(grid, dataset)
                            bbox_local = [np.min(lon), np.min(lat), np.max(lon), np.max(lat)]
                        else:
                            bbox_local = bbox[:]
                        grids.append(grid)
                    self._plot_vector_grids(dataset, grids, time, elevation,
                        level, bbox_local, (w, h), ax, crs, plot_method, 
                        vector_color, legend, vector_spacing, vector_offset)
                    # Force paletting of black vector plots to max 7 colors
                    # and 127 for color vectors (disabled - antialiasing is
                    # disabled for color vectors for now)
                    if plot_method in ['black_vector', 'black_quiver', 
                                       'black_barbs', 'black_arrowbarbs']:
                        ncolors = 7
                    #else:
                    #    ncolors = 127

            # Save to buffer.
            if bbox is not None:
                ax.axis( [bbox[0], bbox[2], bbox[1], bbox[3]] )
            ax.axis('off')
            canvas = FigureCanvas(fig)
            # Optionally convert to paletted png
            config = environ.get('pydap.config', {})
            paletted = asbool(config.get('pydap.responses.wms.paletted', 'false'))
            if paletted:
                output = plotutils.convert_paletted(canvas, ncolors=ncolors)
            else:
                output = BytesIO() 
                canvas.print_png(output)
            if hasattr(dataset, 'close'): dataset.close()
            return [ output.getvalue() ]
        return serialize(self.dataset)

    #@profile
    def _get_proj(self, crs):
        """\
        Returns projection objects.
        """
        # We assume that lon/lat arrays are centered on data points
        base_crs = 'EPSG:4326'
        do_proj = crs != base_crs

        if do_proj:
            # We use cached versions of the projection object if possible.
            # It is safe to store these forever.
            key = (base_crs, 'pyproj')
            if 'pyproj' in self.localCache and key in self.localCache['pyproj'].keys():
                p_base = self.localCache['pyproj'][key]
            else:
                try:
                    if not self.cache:
                        raise KeyError
                    p_base = self.cache.get(key)
                    if p_base is NO_VALUE:
                        raise KeyError
                except KeyError:
                    p_base = pyproj.Proj(init=base_crs)
                    if self.cache:
                        self.cache.set(key, p_base)
            if key not in self.localCache['pyproj'].keys():
                self.localCache['pyproj'][key] = p_base
             
            key = (crs, 'pyproj')
            if 'pyproj' in self.localCache and key in self.localCache['pyproj'].keys():
                p_query = self.localCache['pyproj'][key]
            else:
                try:
                    if not self.cache:
                        raise KeyError
                    p_query = self.cache.get(key)
                    if p_query is NO_VALUE:
                        raise KeyError
                except KeyError:
                    p_query = pyproj.Proj(init=crs)
                    if self.cache:
                        self.cache.set(key, p_query)
            if key not in self.localCache['pyproj'].keys():
                self.localCache['pyproj'][key] = p_query
        else:
            p_base = None
            p_query = None
        return do_proj, p_base, p_query

    #@profile
    def _prepare_grid(self, crs, bbox, grid, dataset, return_proj):
        """\
        Calculates various grid parameters.
        """
        got_proj = False
        # Plot the data over all the extension of the bbox.
        # First we "rewind" the data window to the begining of the bbox:
        lonGrid = gridutils.get_lon(grid, dataset)
        latGrid = gridutils.get_lat(grid, dataset)
        cyclic = hasattr(lonGrid, 'modulo')

        # This operation is expensive - cache results both locally and using
        # redis
        key = (self.path, grid.id, crs, 'project_data')
        try:
            if not ('project_data' in self.localCache and
                    key in self.localCache['project_data'].keys()):
                raise KeyError
            lon, lat, dlon, do_proj = pickle.loads(self.localCache['project_data'][key])
            # Check that the dimensions match - otherwise discard
            # cached value and recalculate
            # TODO: Check first and last values as well
            if len(lonGrid.shape) == 1:
                shapeGrid = (latGrid.shape[0], lonGrid.shape[0])
            else:
                shapeGrid = lonGrid.shape
            if shapeGrid != lon.shape or shapeGrid != lat.shape:
                del self.localCache['project_data'][key]
                raise KeyError
        except KeyError:
            try:
                if not self.cache:
                    raise KeyError
                value = self.cache.get(key)
                if value is NO_VALUE:
                    raise KeyError
                lon, lat, dlon, do_proj = value
                # Check that the dimensions match - otherwise discard
                # cached value and recalculate
                # TODO: Check first and last values as well
                if len(lonGrid.shape) == 1:
                    shapeGrid = (latGrid.shape[0], lonGrid.shape[0])
                else:
                    shapeGrid = lonGrid.shape
                if shapeGrid != lon.shape or shapeGrid != lat.shape:
                    self.cache.invalidate()
                    raise KeyError
            except KeyError:
                # Project data
                lon = np.asarray(lonGrid[:])
                lat = np.asarray(latGrid[:])
                got_proj = True
                do_proj, p_base, p_query = self._get_proj(crs)
                if not do_proj:
                    dlon = 360.0
                else:
                    # Fetch projected data from disk if possible
                    proj_attr = 'coordinates_' + crs.replace(':', '_')
                    if proj_attr in grid.attributes:
                        proj_coords = grid.attributes[proj_attr].split()
                        yname, xname = proj_coords
                        lat = np.asarray(dataset[yname])
                        lon = np.asarray(dataset[xname])
                        dlon = dataset[xname].attributes['modulo']
                    else:
                        lon, lat, dlon, do_proj = projutils.project_data(p_base, p_query,
                                          bbox, lon, lat, cyclic)
                    if self.cache:
                        self.cache.set(key, (lon, lat, dlon, do_proj))
        if key not in self.localCache['project_data'].keys():
            self.localCache['project_data'][key] = pickle.dumps((lon, lat, dlon, do_proj), -1)

        if return_proj:
            if not got_proj:
                do_proj, p_base, p_query = self._get_proj(crs)
            return lon, lat, dlon, cyclic, do_proj, p_base, p_query
        else:
            return lon, lat, dlon, cyclic

    #@profile
    def _find_slices(self, lon, lat, dlon, bbox, cyclic, size):
        """Returns slicing information for a given input."""
        # Retrieve only the data for the request bbox, and at the 
        # optimal resolution (avoiding oversampling).
        w, h = size
        if len(lon.shape) == 1:
            I, J = np.arange(lon.shape[0]), np.arange(lat.shape[0])

            # Find points within bounding box
            xcond = (lon >= bbox[0]) & (lon <= bbox[2])
            ycond = (lat >= bbox[1]) & (lat <= bbox[3])
            # Check whether any of the points are in the bounding box
            if (not xcond.any() or not ycond.any()):
                # If the bounding box is smaller than the grid cell size
                # it can "fall between" grid cell centers. In this situation
                # xcond and ycond are all false but we do want to return
                # the surrounding points so we perform an additional check
                # for this before raising an exception.
                if not xcond.any():
                    xcond2 = ((lon[:-1] <= bbox[0]) & (lon[1:] >= bbox[2]))
                    xcond[:-1] = xcond2
                if not ycond.any():
                    ycond2 = ((lat[:-1] <= bbox[1]) & (lat[1:] >= bbox[3]))
                    ycond[:-1] = ycond2
                if (not xcond.any() or not ycond.any()):
                    raise OutsideGridException

            istep = max(1, int(np.floor( (lon.shape[0] * (bbox[2]-bbox[0])) / (w * abs(np.amax(lon)-np.amin(lon))) )))
            jstep = max(1, int(np.floor( (lat.shape[0] * (bbox[3]-bbox[1])) / (h * abs(np.amax(lat)-np.amin(lat))) )))
            i0 = 0
            j0 = 0
            lon1 = lon[i0::istep]
            lat1 = lat[j0::jstep]
            # Find containing bound in reduced indices
            I, J = np.arange(lon1.shape[0]), np.arange(lat1.shape[0])
            #I, J = np.meshgrid(i, j)
            xcond = (lon1 >= bbox[0]) & (lon1 <= bbox[2])
            ycond = (lat1 >= bbox[1]) & (lat1 <= bbox[3])
            if not xcond.any() or not ycond.any():
                return
            i0r = np.min(I[xcond])-1
            i1r = np.max(I[xcond])+2
            j0r = np.min(J[ycond])-1
            j1r = np.max(J[ycond])+2
            # Convert reduced indices to global indices
            i0 = max(istep*i0r, 0)
            i1 = min(istep*i1r, lon.shape[0]+1)
            j0 = max(jstep*j0r, 0)
            j1 = min(jstep*j1r, lat.shape[0]+1)
            #i0, i1 = gridutils.find_containing_bounds(lon, bbox[0], bbox[2])
            #j0, j1 = gridutils.find_containing_bounds(lat, bbox[1], bbox[3])

        elif len(lon.shape) == 2:
            i, j = np.arange(lon.shape[1]), np.arange(lon.shape[0])
            I, J = np.meshgrid(i, j, copy=False) # No copy needed

            # Find points within bounding box
            xcond = (lon >= bbox[0]) & (lon <= bbox[2])
            ycond = (lat >= bbox[1]) & (lat <= bbox[3])
            cond = xcond & ycond

            if not cond.any():
                # When bbox "falls between" grid cells xcond and ycond are
                # all false. So we have an additional check for that
                if not xcond.any():
                    xcond2 = ((lon[:,:-1] <= bbox[0]) & (lon[:,1:] >= bbox[2]))
                    xcond[:,:-1] = xcond2
                if not ycond.any():
                    if lat[0,0] < lat[-1,0]:
                        ycond2 = ((lat[:-1,:] <= bbox[1]) & (lat[1:,:] >= bbox[3]))
                    else:
                        ycond2 = ((lat[:-1,:] >= bbox[3]) & (lat[1:,:] <= bbox[1]))
                    ycond[:-1,:] = ycond2
                cond = xcond & ycond
                if not cond.any():
                    raise OutsideGridException

            # Avoid oversampling
            istep = max(1, int(np.floor( (lon.shape[1] * (bbox[2]-bbox[0])) / (w * abs(np.amax(lon)-np.amin(lon))) )))
            jstep = max(1, int(np.floor( (lon.shape[0] * (bbox[3]-bbox[1])) / (h * abs(np.amax(lat)-np.amin(lat))) )))

            # Find containing bounds
            # TODO: Performance can be improved by avoiding 2d I and J's
            Icond = I[cond]
            Jcond = J[cond]
            i0 = max(0, np.min(Icond)-1*istep)
            i1 = min(np.max(Icond)+2*istep, lon.shape[1])
            j0 = max(0, np.min(Jcond)-1*jstep)
            j1 = min(np.max(Jcond)+2*jstep, lon.shape[0])

            # Set common origin for grid indices
            i0 = (i0 // istep)*istep
            i1 = min((i1 // istep + 1)*istep, lon.shape[1])
            j0 = (j0 // jstep)*jstep
            j1 = min((j1 // jstep + 1)*jstep, lon.shape[0])
        return (j0, j1, jstep), (i0, i1, istep)

    #@profile
    def _extract_data(self, l, level, j0, j1, jstep, i0, i1, istep, 
                      lat, lon, dlon, cyclic, grids):
        """Returns coordinate and data arrays given slicing information."""
        # TODO: Currently we just select the first lon value for cyclic
        # wrapping for the 1d lat/lon case. It should be "equally spaced".
        if len(lon.shape) == 1:
            X = lon[i0:i1:istep]
            Y = lat[j0:j1:jstep]
            # Fix cyclic data.
            if cyclic:
                X = np.ma.concatenate((X, X[0:1] + dlon), 0)
            X, Y = np.meshgrid(X, Y)
        else:
            X = lon[j0:j1:jstep,i0:i1:istep]
            Y = lat[j0:j1:jstep,i0:i1:istep]
            # Fix cyclic data.
            if cyclic:
                if i0 < 0:
                    # The leftmost bound is to the left of the grid origin
                    X = np.concatenate((X[:,i0:-1:istep]-dlon, X), axis=1)
                    Y = np.concatenate((Y[:,i0:-1:istep], Y), axis=1)
                if i1 > lon.shape[1]:
                    # The rightmost bound is to the right of the grid extent
                    di = i1 - lon.shape[1]
                    X = np.concatenate((X, X[:,0:di:istep]+dlon), axis=1)
                    Y = np.concatenate((Y, Y[:,0:di:istep]), axis=1)
        data = []
        for grid in grids:
            # The line below is a performance bottleneck.
            if gridutils.get_vertical(grid) is None:
                gdata = np.asarray(grid.array[l,j0:j1:jstep,i0:i1:istep])
                # FIXME: Does not work for NCA's. The line below does but is inefficient
                #gdata = np.asarray(grid.array)[l,j0:j1:jstep,i0:i1:istep]
                if cyclic:
                    if len(lon.shape) == 1:
                        gdata = np.ma.concatenate((
                            gdata, np.asarray(grid.array[l,j0:j1:jstep,0:1])), -1)
                    else:
                        if i0 < 0:
                            # The leftmost bound is to the left of the grid origin
                            ii = lon.shape[1] + i0
                            gdata = np.ma.concatenate((grid.array[l,j0:j1:jstep,ii:-1:istep], 
                                                       gdata), axis=-1)
                        if i1 > lon.shape[1]:
                            # The rightmost bound is to the right of the grid extent
                            di = i1 - lon.shape[1]
                            gdata = np.ma.concatenate((gdata,
                                                       grid.array[l,j0:j1:jstep,0:di:istep]), axis=-1)
            else:
                gdata = np.asarray(grid.array[l,level,j0:j1:jstep,i0:i1:istep])
                if cyclic:
                    if len(lon.shape) == 1:
                        gdata = np.ma.concatenate((
                            gdata, np.asarray(grid.array[l,level,j0:j1:jstep,0:1])), -1)
                    else:
                        if i0 < 0:
                            # The leftmost bound is to the left of the grid origin
                            ii = lon.shape[1] + i0
                            gdata = np.ma.concatenate((grid.array[l,level,j0:j1:jstep,ii:-1:istep], 
                                                       gdata), axis=-1)
                        if i1 > lon.shape[1]:
                            # The rightmost bound is to the right of the grid extent
                            di = i1 - lon.shape[1]
                            gdata = np.ma.concatenate((gdata,
                                                       grid.array[l,level,j0:j1:jstep,0:di:istep]), axis=-1)

            data.append(gdata)

        # Postcondition
        assert X.shape == Y.shape, 'shapes: %s %s' % (X.shape, Y.shape)

        return X, Y, data


    #@profile
    def _plot_vector_grids(self, dataset, grids, tm, elevation, level, bbox,
                           size, ax, crs, vector_method, vector_color, 
                           cmapname, vector_spacing, vector_offset):
        try:
            # Slice according to time request (WMS-T).
            l = gridutils.time_slice(tm, grids[0], dataset)
        except IndexError:
            # Return empty image for out of time range requests
            return
 
        vertical = gridutils.get_vertical(grids[0])
        if elevation is not None:
            vert = np.asarray(vertical[:])
            vert_bool = np.isclose(vert, elevation)
            if not vert_bool.any():
                # Return empty image for requests with no corresponding
                # vertical level
                return
            level = np.where(vert_bool)[0][0]

        # Return empty image for vertical indices out of range
        if vertical is not None and vertical.shape[0] <= level:
            return

        size_ave = np.mean(size)
        nnice = int(size_ave/vector_spacing)
        offset = vector_offset/size_ave
        X1d, Y1d = nice_points(bbox, nnice, offset)
        X, Y = np.meshgrid(X1d, Y1d)
        points_intp = np.vstack((X.flatten(), Y.flatten())).T

        # Extract projected grid information
        lon, lat, dlon, cyclic, do_proj, p_base, p_query = \
                self._prepare_grid(crs, bbox, grids[0], dataset,
                                   return_proj=True)

        # If lon/lat arrays are 1d make them 2d
        if len(lon.shape) == 1 and len(lat.shape) == 1:
            lon, lat = np.meshgrid(lon, lat)

        # Now we plot the data window until the end of the bbox:
        cnt_window = 0
        while np.min(lon) < bbox[2]:
            lon_save = lon[:]
            lat_save = lat[:]

            # We do not really need the slicing information but we want to
            # see if an OutsideGridException is thrown
            try:
                (j0, j1, jstep), (i0, i1, istep) = \
                self._find_slices(lon, lat, dlon, bbox, cyclic, size)
            except OutsideGridException:
                lon += dlon
                cnt_window += 1
                continue

            lonf = lon[j0:j1:jstep,i0:i1:istep].flatten()
            latf = lat[j0:j1:jstep,i0:i1:istep].flatten()
            points = np.vstack((lonf, latf)).T

            # Get convex hull mask and cache it
            try:
                if not self.cache:
                    raise KeyError
                key = (self.path, grids[0].id, tuple(bbox), crs, cnt_window,
                       vector_spacing, vector_offset, 'in_hull')
                in_hull = self.cache.get(key)
                if in_hull is NO_VALUE:
                    raise KeyError
            except KeyError:
                hull = ConvexHull(points)
                vertices = np.hstack([hull.vertices, hull.vertices[0]])
                in_hull = np.ones(X.shape, dtype=bool)
                for i in range(len(vertices)-1):
                    v1 = vertices[i]
                    v2 = vertices[i+1]
                    x1, y1 = points[v1,:]
                    x2, y2 = points[v2,:]
                    left_simplex = np.sign((x2-x1)*(Y-y1) - (y2-y1)*(X-x1)) > -1
                    in_hull = in_hull & left_simplex
                in_hull = np.invert(in_hull)
                if self.cache:
                    in_hull_str = BytesIO()
                    key = (self.path, grids[0].id, tuple(bbox), crs, 
                           cnt_window, vector_spacing, vector_offset, 'in_hull')
                    self.cache.set(key, in_hull)
                    
            data = []
            ny, nx = X.shape
            data = []
            for grid in grids:
                attrs = grid.attributes
                if gridutils.get_vertical(grid) is None:
                    values = np.asarray(grid.array[l,j0:j1:jstep,i0:i1:istep]).squeeze().flatten()
                else:
                    values = np.asarray(grid.array[l,level,j0:j1:jstep,i0:i1:istep]).squeeze().flatten()
                if 'missing_value' in attrs:
                    missing_value = attrs['missing_value']
                    values = np.ma.masked_equal(values, missing_value)
                elif '_FillValue' in attrs:
                    missing_value = attrs['_FillValue']
                    values = np.ma.masked_equal(values, missing_value)
                #assert points.shape == values.shape, (points.shape, values.shape, l, level, j0, j1, jstep, i0, i1, istep)
                f = interpolate.NearestNDInterpolator(points, values)
                d = np.ma.masked_equal(f(points_intp).reshape((ny, nx)), missing_value)
                d = np.ma.masked_where(in_hull, d)
                data.append(d)

            # Plot data.
            if data[0].shape: 
                """
                # apply time slices
                if l is not None:
                    data = [np.asarray(data[0])[l],
                            np.asarray(data[1])[l]]
                """

                # reduce dimensions and mask missing_values
                data[0] = gridutils.fix_data(data[0], grids[0].attributes)
                data[1] = gridutils.fix_data(data[1], grids[1].attributes)

                # plot
                if np.ma.count(data[0]) > 0:
                    if do_proj:
                        # Transform back to lat/lon (can be optimized)
                        lons, lats = pyproj.transform(p_query, p_base, X, Y)
                        u,v = projutils.rotate_vector(p_base, p_query, data[0], 
                                data[1], lons, lats, returnxy=False)
                    # Custom increments since our data are in m/s
                    barb_incs = {'half': 2.57222,
                                 'full': 5.14444,
                                 'flag': 25.7222}

                    # Colormap information
                    if cmapname in self.colors:
                        norm = self.colors[cmapname]['norm']
                        cmap = self.colors[cmapname]['cmap']
                        levels = norm.boundaries
                    else:
                        # Get actual data range for levels.
                        actual_range = self._get_actual_range(grid)
                        norm = Normalize(vmin=actual_range[0], vmax=actual_range[1])
                        ncolors = 15
                        dlev = (actual_range[1] - actual_range[0])/ncolors
                        levels = np.arange(actual_range[0], actual_range[1]+dlev, dlev)
                        cmap = self._get_cmap(cmapname)
                    newlevels = levels[:]
                    dtype = newlevels.dtype
                    if cmap.colorbar_extend in ['min', 'both']:
                        newlevels = np.insert(newlevels, 0, np.finfo(dtype).min)
                    if cmap.colorbar_extend in ['max', 'both']:
                        newlevels = np.append(newlevels, np.finfo(dtype).max)

                    # Vector magnitude
                    d = np.ma.sqrt(data[0]**2 + data[1]**2)

                    # Construct limiter
                    dmin = newlevels[1]
                    di = np.where(d<0.5*dmin, 0, 1/d)

                    if vector_method == 'black_barbs':
                        sizes = {'spacing': 0.2,
                                 'width': 0.2,
                                 'headwidth': 1.0,
                                 'headlength': 1.0,
                                 'headaxislength': 1.0,
                                 'emptybarb': 0.2}
                        arrow_barbs(ax, X, Y, data[0], data[1], pivot='middle', 
                                    length=5.5, linewidth=1, color=vector_color,
                                    edgecolor='w', antialiased=True, fill_empty=True,
                                    sizes=sizes, barb_increments=barb_incs)
                    elif vector_method == 'black_arrowbarbs':
                        sizes = {'spacing': 0.2,
                                 'width': 0.2,
                                 'headwidth': 2.0,
                                 'headlength': 1.0,
                                 'headaxislength': 1.25,
                                 'emptybarb': 0.2}
                        arrow_barbs(ax, X, Y, data[0], data[1], pivot='middle', 
                                    length=5.5, linewidth=1, color=vector_color,
                                    edgecolor='w', antialiased=True, fill_empty=True,
                                    sizes=sizes, barb_increments=barb_incs)
                    elif vector_method == 'color_quiver1':
                        ax.quiver(X, Y, data[0]*di, data[1]*di, d, pivot='middle',
                        units='inches', scale=3.0, scale_units='inches',
                        width=0.11, linewidths=0.5, headwidth=2,
                        headlength=2, headaxislength=2,
                        edgecolors=('k'), antialiased=False,
                        norm=norm, cmap=cmap)
                    elif vector_method == 'color_quiver2':
                        ax.quiver(X, Y, data[0]*di, data[1]*di, d, pivot='middle',
                        units='inches', scale=4.0, scale_units='inches',
                        width=0.0825, linewidths=0.5, headwidth=2,
                        headlength=1, headaxislength=1,
                        edgecolors=('k'), antialiased=False,
                        norm=norm, cmap=cmap)
                    elif vector_method == 'color_quiver3':
                        ax.quiver(X, Y, data[0]*di, data[1]*di, d, pivot='middle',
                        units='inches', scale=4.0, scale_units='inches',
                        width=0.05, linewidths=0.5, headwidth=3,
                        headlength=1.5, headaxislength=1.5,
                        edgecolors=('k'), antialiased=False,
                        norm=norm, cmap=cmap)
                    elif vector_method == 'color_quiver4':
                        ax.quiver(X, Y, data[0]*di, data[1]*di, d, pivot='middle',
                        units='inches', scale=4.0, scale_units='inches',
                        width=0.1875, linewidths=0.5, headwidth=1,
                        headlength=0.5, headaxislength=0.5,
                        edgecolors=('k'), antialiased=False,
                        norm=norm, cmap=cmap)
                    else:
                        #if vector_method == 'black_quiver':
                        ax.quiver(X, Y, data[0]*di, data[1]*di, pivot='middle',
                                  units='inches', scale=4.0, scale_units='inches',
                                  width=0.04, color=vector_color, linewidths=1,
                                  headlength=4, headaxislength=3.5,
                                  edgecolors=('w'), antialiased=True)
                        
            lon = lon_save
            lat = lat_save
            lon += dlon
            cnt_window += 1


    #@profile
    def _plot_grid(self, dataset, grid, time, elevation, level, bbox, size, ax,
                   crs, fill_method, cmapname='jet', expr=None,
                   layers=None):
        if expr is not None:
            aeval = RestrictedInterpreter()
            grids = grid
            grid = grids[0]
        # We currently only support contour and contourf
        assert fill_method in ['contour', 'contourf']

        try:
            # Slice according to time request (WMS-T).
            l = gridutils.time_slice(time, grid, dataset)
        except IndexError:
            # Return empty image for out of time range requests
            return
 
        vertical = gridutils.get_vertical(grid)
        if elevation is not None:
            vert = np.asarray(vertical[:])
            vert_bool = np.isclose(vert, elevation)
            if not vert_bool.any():
                # Return empty image for requests with no corresponding
                # vertical level
                return
            level = np.where(vert_bool)[0][0]

        # Return empty image for vertical indices out of range
        if vertical is not None and vertical.shape[0] <= level:
            return

        # Extract projected grid information
        lon, lat, dlon, cyclic = \
                  self._prepare_grid(crs, bbox, grid, dataset,
                                     return_proj=False)

        # Now we plot the data window until the end of the bbox:
        while np.min(lon) < bbox[2]:
            lon_save = lon[:]
            lat_save = lat[:]

            try:
                (j0, j1, jstep), (i0, i1, istep) = \
                self._find_slices(lon, lat, dlon, bbox, cyclic, size)
            except OutsideGridException:
                lon += dlon
                continue
            
            if expr is None:
                X, Y, data = self._extract_data(l, level, j0, j1, jstep,
                             i0, i1, istep, lat, lon, dlon, cyclic, [grid])
                data = data[0]
            else:
                X, Y, datas = self._extract_data(l, level, j0, j1, jstep,
                              i0, i1, istep, lat, lon, dlon, cyclic, grids)
                data = datas[0]

            # Plot data.
            if data.shape: 
                # FIXME: Is this section really necessary - see vector method
                # apply time slices
                if l is not None:
                    data = np.asarray(data)
                else:
                    # FIXME: Do the indexing here if we decide to go for just the first timestep
                    if len(data.shape) > 2:
                        # The array proxy object cannot be sliced prior to conversion
                        data = np.asarray(data)[0]
                    else:
                        data = np.asarray(data)

                if expr is None:
                    # reduce dimensions and mask missing_values
                    data = gridutils.fix_data(data, grid.attributes)
                else:
                    # reduce dimensions and mask missing_values
                    for i in range(len(datas)):
                        datas[i] = gridutils.fix_data(datas[i], grids[i].attributes)
                        aeval.symtable[layers[i]] = datas[i]
                    data = aeval(expr)
                    if len(aeval.error) > 0:
                        errs = '\n'.join([str(e.get_error()) for e in aeval.error])
                        #raise HTTPBadRequest('Error evaluating expression: "%s"'
                        #                     'Error messages:\n%s' % (expr, errs))
                        #raise ValueError('Error evaluating expression: %s\n'
                        #                 'Error messages:\n%s' % (expr, errs))
                        raise ValueError('Error evaluating expression: %s\n' % (expr))

                # plot
                #if data.shape and data.any():
                if data.shape and np.ma.count(data) > 0:
                    plot_method = getattr(ax, fill_method)
                    if cmapname in self.colors:
                        norm = self.colors[cmapname]['norm']
                        cmap = self.colors[cmapname]['cmap']
                        levels = norm.boundaries
                    else:
                        # Get actual data range for levels.
                        actual_range = self._get_actual_range(grid)
                        norm = Normalize(vmin=actual_range[0], vmax=actual_range[1])
                        ncolors = 15
                        dlev = (actual_range[1] - actual_range[0])/ncolors
                        levels = np.arange(actual_range[0], actual_range[1]+dlev, dlev)
                        cmap = self._get_cmap(cmapname)
                    if fill_method == 'contourf':
                        newlevels = levels[:]
                        dtype = newlevels.dtype
                        if cmap.colorbar_extend in ['min', 'both']:
                            newlevels = np.insert(newlevels, 0, np.finfo(dtype).min)
                        if cmap.colorbar_extend in ['max', 'both']:
                            newlevels = np.append(newlevels, np.finfo(dtype).max)
                        plot_method(X, Y, data, norm=norm, cmap=cmap, 
                                    levels=newlevels, antialiased=False)
                        #newlevels = plotutils.modify_contour_levels(levels, cmap.colorbar_extend)
                        #cs = ax.contour(X, Y, data, colors='w', levels=newlevels, 
                        #            antialiased=False)
                        #ax.clabel(cs, inline=1, fontsize=10)
                    elif fill_method == 'contour':
                        newlevels = plotutils.modify_contour_levels(levels, 
                                    cmap.colorbar_extend)
                        fltfmt = FormatStrFormatter('%d')
                        cs = plot_method(X, Y, data, colors='black', levels=newlevels, 
                                    antialiased=False)
                        cs.levels = [fltfmt(val) for val in cs.levels]
                        clbls = ax.clabel(cs, inline=1, fontsize=11, use_clabeltext=True)
                        setp(clbls, path_effects=[withStroke(linewidth=2, foreground="white")])
                    else:
                        plot_method(X, Y, data, norm=norm, cmap=cmap, antialiased=False)
            lon = lon_save
            lat = lat_save
            lon += dlon

        # Check if we should plot annotations on the map
        if 'annotations' in grid.attributes:
            annotations = grid.attributes['annotations']
            avar = dataset[annotations]
            adims = avar.dimensions
            acoords = avar.coordinates.split()
            assert len(acoords) == 2
            nt, ny, nx = avar.array.shape
            adata = np.asarray(avar.array.data)[l,0:ny:1,0:nx:1]
            nanno, nstr = adata.shape
            aadata = np.empty((nanno,), dtype=np.object)
            for inanno in range(nanno):
                ads = [bs.decode('utf-8') for bs in adata[inanno,:]]
                aadata[inanno] = ''.join(ads).strip()

            alat = np.asarray(dataset[acoords[0]].array.data)[l,:]
            alon = np.asarray(dataset[acoords[1]].array.data)[l,:]
            alat = alat.reshape((nanno))
            alon = alon.reshape((nanno))
            do_proj, p_base, p_query = self._get_proj(crs)
            x, y = pyproj.transform(p_base, p_query, alon, alat)
            _plot_annotations(aadata, y, x, ax)


    def _get_capabilities(self, environ):
        def serialize(dataset):
            """
            # Check for cached version of XML document
            last_modified = None
            for header in environ['pydap.headers']:
                if 'Last-modified' in header:
                    last_modified = header[1]
            if last_modified is not None and self.cache:
                try:
                    key = ('capabilities', self.path, last_modified)
                    output = self.cache.get(key, expiration_time=86400)
                    if output is NO_VALUE:
                        raise KeyError
                    output = output[:]
                    if hasattr(dataset, 'close'): dataset.close()
                    return [output]
                except KeyError:
                    pass
            """

            gridutils.fix_map_attributes(dataset)
            grids = [grid for grid in walk(dataset, GridType) if gridutils.is_valid(grid, dataset)]

            # Set global lon/lat ranges.
            try:
                lon_range = self.cache.get((self.path, 'lon_range'))
                if lon_range is NO_VALUE:
                    raise KeyError
            except (KeyError, AttributeError):
                try:
                    lon_range = dataset.attributes['NC_GLOBAL']['lon_range']
                except KeyError:
                    lon_range = [np.inf, -np.inf]
                    for grid in grids:
                        lon = np.asarray(gridutils.get_lon(grid, dataset)[:])
                        lon_range[0] = min(lon_range[0], np.min(lon))
                        lon_range[1] = max(lon_range[1], np.max(lon))
                if self.cache:
                    key = (self.path, 'lon_range')
                    self.cache.set(key, lon_range)
            try:
                lat_range = self.cache.get((self.path, 'lat_range'))
                if lat_range is NO_VALUE:
                    raise KeyError
            except (KeyError, AttributeError):
                try:
                    lat_range = dataset.attributes['NC_GLOBAL']['lat_range']
                except KeyError:
                    lat_range = [np.inf, -np.inf]
                    for grid in grids:
                        lat = np.asarray(gridutils.get_lat(grid, dataset)[:])
                        lat_range[0] = min(lat_range[0], np.min(lat))
                        lat_range[1] = max(lat_range[1], np.max(lat))
                if self.cache:
                    key = (self.path, 'lat_range')
                    self.cache.set(key, lat_range)

            # Remove ``REQUEST=GetCapabilites`` from query string.
            location = construct_url(environ, with_query_string=True)

            #base = location.split('REQUEST=')[0].rstrip('?&')
            base = location.split('?')[0].rstrip('.wms')

            layers = build_layers(dataset, self.styles)

            context = {
                    'dataset': dataset,
                    'location': base,
                    'layers': layers,
                    'lon_range': lon_range,
                    'lat_range': lat_range,
                    'max_width': MAX_WIDTH,
                    'max_height': MAX_HEIGHT,
                    'default_crs': DEFAULT_CRS,
                    'default_styles': self.styles,
                    'supported_crs': SUPPORTED_CRS
            }
            # Load the template using the specified renderer, or fallback to the 
            # default template since most of the people won't bother installing
            # and/or creating a capabilities template -- this guarantees that the
            # response will work out of the box.
            try:
                renderer = environ['pydap.renderer']
                template = renderer.loader('capabilities.xml')
            except (KeyError, TemplateNotFound):
                renderer = self.renderer
                template = renderer.loader('capabilities.xml')

            output = renderer.render(template, context, output_format='text/xml')
            if hasattr(dataset, 'close'): dataset.close()
            output = output.encode('utf-8')
            """
            if last_modified is not None and self.cache:
                key = ('capabilities', self.path, last_modified)
                self.cache.set(key, output[:])
            """
            return [output]
        return serialize(self.dataset)

    def _get_metadata(self, environ):
        #@profile
        def serialize(dataset):
            # Figure out when data file was last modified
            last_modified = None
            #for header in environ['pydap.headers']:
            #    if 'Last-modified' in header:
            #        last_modified = header[1]

            # Remove ``REQUEST=GetMetadata`` from query string.
            location = construct_url(environ, with_query_string=True)
            base = location.split('?')[0].rstrip('.wms')
            query = parse_dict_querystring_lower(environ)

            # Construct list of requested layers
            layers = sorted([layer for layer in query.get('layers', '').split(',')
                      if layer])

            # Construct list of requested information items
            items = sorted(query.get('items', 'epoch').split(','))

            # Decide how long an expiration time we will use. We default to
            # one day (86400 seconds)
            expiretime = 86400
            global_attrs = dataset.attributes['NC_GLOBAL']
            # Reduce expiration time when requesting epoch, last_modified
            # or time since these values are updated when new forecasts
            # are produced
            if 'epoch' in items and 'epoch' in global_attrs:
                expiretime = 60
            elif 'last_modified' in items and last_modified is not None:
                expiretime = 60
            else:
                for layer in layers:
                    if 'time' in items:
                        tm = gridutils.get_time(dataset[layer])
                        if tm is not None:
                            expiretime = 60

            # Check for cached version of JSON document
            if last_modified is not None and self.cache:
                try:
                    key = ('_get_metadata+all', self.path, 
                           'layers=' + '+'.join(layers), 
                           'items=' + '+'.join(items))
                    output = self.cache.get(key, expiration_time=expiretime)
                    if output is NO_VALUE:
                        raise KeyError
                    # Only use cached version if last_modified attribute is
                    # the same as that of the current file
                    if 'last_modified' in items:
                        last_modified_file = datetime.utcfromtimestamp(
                              calendar.timegm(parsedate(last_modified))). \
                              strftime('%Y-%m-%dT%H:%M:%SZ')
                        if output['last_modified'] != last_modified_file:
                            raise KeyError
                    if hasattr(dataset, 'close'): dataset.close()
                    output = json.dumps(output)
                    return [output]
                except KeyError:
                    pass

            gridutils.fix_map_attributes(dataset)
            grids = [grid for grid in walk(dataset, GridType) if gridutils.is_valid(grid, dataset)]

            output = {}

            # We often have the same time and bounds for different layers so we
            # try to reuse the information from the first layer
            # We do this caching only using the entire dimensions attribute
            # even though we could optimize it further
            bounds_cache = {}
            time_cache = {}
            for layer in layers:
                output[layer] = {}
                attrs = dataset[layer].attributes
                # Only cache epoch, last_modified and time requests for 60 seconds
                if 'units' in items and 'units' in attrs:
                    output[layer]['units'] = attrs['units']
                if 'long_name' in items and 'long_name' in attrs:
                    output[layer]['long_name'] = attrs['long_name']
                if 'styles' in items and 'standard_name' in attrs and \
                   attrs['standard_name'] in self.styles:
                    output[layer]['styles'] = self.styles[attrs['standard_name']]
                if 'bounds' in items:
                    try:
                        if not self.cache:
                            raise KeyError
                        key = ('metadata_bounds', self.path, layer)
                        output[layer]['bounds'] = self.cache.get(key,
                                                  expiration_time=864000)
                        if output[layer]['bounds'] is NO_VALUE:
                            raise KeyError
                        output[layer]['bounds'] = output[layer]['bounds'][:]
                    except KeyError:
                        dims = dataset[layer].dimensions
                        if dims not in bounds_cache:
                            lon = np.asarray(gridutils.get_lon(dataset[layer], dataset))
                            lat = np.asarray(gridutils.get_lat(dataset[layer], dataset))
                            minx, maxx = float(np.amin(lon)), float(np.amax(lon))
                            miny, maxy = float(np.amin(lat)), float(np.amax(lat))
                            bounds_cache[dims] = [minx, maxx, miny, maxy]
                        else:
                            minx, maxx, miny, maxy = bounds_cache[dims]
                        output[layer]['bounds'] = [minx, miny, maxx, maxy]
                        key = ('metadata_bounds', self.path, layer)
                        if self.cache:
                            self.cache.set(key, output[layer]['bounds'])
                if 'time' in items:
                    dims = dataset[layer].dimensions
                    if dims not in time_cache:
                        tm = gridutils.get_time(dataset[layer])
                        time_cache[dims] = tm
                    else:
                        tm = time_cache[dims]
                    if tm is not None:
                        output[layer]['time'] = [t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in tm]
                levels = gridutils.get_vertical(dataset[layer])
                if levels is not None and 'levels' in items:
                    output[layer]['levels'] = {}
                    output[layer]['levels']['values'] = np.asarray(levels[:]).tolist()
                    if 'units' in levels.attributes:
                        output[layer]['levels']['units'] = levels.attributes['units']
                    if 'long_name' in levels.attributes:
                        output[layer]['levels']['long_name'] = levels.attributes['long_name']
                    if 'positive' in levels.attributes:
                        output[layer]['levels']['positive'] = levels.attributes['positive']

                if not output[layer]:
                    del output[layer]
            if 'epoch' in items and 'epoch' in global_attrs:
                output['epoch'] = global_attrs['epoch']
            if 'last_modified' in items and last_modified is not None:
                last_modified = datetime.utcfromtimestamp(
                      calendar.timegm(parsedate(last_modified))). \
                      strftime('%Y-%m-%dT%H:%M:%SZ')
                output['last_modified'] = last_modified

            if hasattr(dataset, 'close'): dataset.close()
            if last_modified is not None and self.cache:
                key = ('_get_metadata+all', self.path, 
                       'layers=' + '+'.join(layers), 
                       'items=' + '+'.join(items))
                self.cache.set(key, output)
            output = json.dumps(output)
            return [output]
        return serialize(self.dataset)

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

def nice_points(bbox, npoints, offset):
    """\
    Returns nice coordinate points for a bbox with a given
    space between. The coordinate points are padded so that
    the bbox is always contained in the points.

    Offset is specified as a fraction of the bbox.
    """
    xmin, ymin, xmax, ymax = bbox
    dx = (xmax-xmin)/float(npoints)
    dy = (ymax-ymin)/float(npoints)
    x0 = (xmax-xmin)*offset
    y0 = (ymax-ymin)*offset
    if abs(x0) > abs(dx) or abs(y0) > abs(dy):
        raise HTTPBadRequest('Offset too large "%s"' % offset)
         
    # Calculate start and end of slices
    x1 = int(np.floor(xmin/dx))*dx - dx + x0
    x2 = int(np.ceil(xmax/dx))*dx + dx + x0
    y1 = int(np.floor(ymin/dy))*dy - dy + y0
    y2 = int(np.ceil(ymax/dy))*dy + dy + y0
    xn = np.round(np.arange(x1, x2+dx, dx), decimals=6)
    yn = np.round(np.arange(y1, y2+dy, dy), decimals=6)
    return xn, yn

def _plot_annotations(data, y, x, ax):
    """Plot string variable annotations."""
    # Loop over extrema
    for xe, ye, te in zip(x, y, data):
       # Extract string and color
       pltstr, color = plotutils.mslext2text(te)
       if pltstr == '': # last one reached
           break
       ax.text(xe, ye, pltstr, color=color, ha='center', va='center',
               fontsize=24, path_effects=[PathEffects.withStroke(linewidth=1.0,
               foreground='k')], zorder=100)

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
        u_standard_name = u_grid.attributes.get('standard_name', '')
        if u_standard_name.startswith('eastward_'):
            postfix = u_standard_name.split('_', 1)[1]
            standard_name = 'northward_' + postfix
            for v_grid in grids:
                v_standard_name = v_grid.attributes.get(
                                  'standard_name', '')
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
