from __future__ import division

from cStringIO import StringIO
import re
import operator
import json
import cPickle
import time
from rfc822 import parsedate
from datetime import datetime

from paste.request import construct_url, parse_dict_querystring
from paste.httpexceptions import HTTPBadRequest, HTTPNotModified
from paste.util.converters import asbool
import numpy as np
from scipy import interpolate
from scipy.spatial import Delaunay
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib import rcParams
from matplotlib.colors import from_levels_and_colors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.pyplot import setp
from matplotlib.patheffects import withStroke

rcParams['xtick.labelsize'] = 'small'
rcParams['ytick.labelsize'] = 'small'
rcParams['contour.negative_linestyle'] = 'solid'
import pyproj

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
from arrowbarbs import arrow_barbs
import projutils
import gridutils
import plotutils

WMS_ARGUMENTS = ['request', 'bbox', 'cmap', 'layers', 'width', 'height', 'transparent', 'time',
                 'styles', 'service', 'version', 'format', 'crs', 'bounds', 'srs', 'expr',
                 'items']

DEFAULT_TEMPLATE = """<?xml version='1.0' encoding="UTF-8" standalone="no" ?>
<!DOCTYPE WMT_MS_Capabilities SYSTEM "http://schemas.opengis.net/wms/1.1.1/WMS_MS_Capabilities.dtd"
 [
 <!ELEMENT VendorSpecificCapabilities EMPTY>
 ]>

<WMT_MS_Capabilities version="1.1.1"
        xmlns:wms="http://www.opengis.net/wms" 
        xmlns:py="http://genshi.edgewall.org/"
        xmlns:xlink="http://www.w3.org/1999/xlink">

<Service>
  <Name>${dataset.name}</Name>
  <Title>WMS server for ${dataset.attributes.get('long_name', dataset.name)}</Title>
  <OnlineResource xlink:href="${location + '.html'}"></OnlineResource>
</Service>

<Capability>
  <Request>
    <GetCapabilities>
      <Format>application/vnd.ogc.wms_xml</Format>
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
    <Format>application/vnd.ogc.se_blank</Format>
  </Exception>
  <VendorSpecificCapabilities></VendorSpecificCapabilities>
  <UserDefinedSymbolization SupportSLD="1" UserLayer="0" UserStyle="1" RemoteWFS="0"/>
  <Layer>
    <Title>WMS server for ${dataset.attributes.get('long_name', dataset.name)}</Title>
    <SRS>EPSG:4326</SRS>
    <SRS>EPSG:3857</SRS>
    <SRS>EPSG:900913</SRS>
    <SRS>EPSG:3785</SRS>
    <SRS>EPSG:3395</SRS>
    <LatLonBoundingBox minx="${lon_range[0]}" miny="${lat_range[0]}" maxx="${lon_range[1]}" maxy="${lat_range[1]}"></LatLonBoundingBox>
    <BoundingBox CRS="EPSG:4326" minx="${lon_range[0]}" miny="${lat_range[0]}" maxx="${lon_range[1]}" maxy="${lat_range[1]}"/>
    <Layer py:for="grid in layers">
      <Name>${grid.name}</Name>
      <Title>${grid.attributes.get('long_name', grid.name)}</Title>
      <Abstract>${grid.attributes.get('history', '')}</Abstract>
      <?python
          import numpy as np
          from pydap.responses.wms.gridutils import get_lon, get_lat, get_time
          lon = get_lon(grid, dataset)
          lat = get_lat(grid, dataset)
          time = get_time(grid, dataset)
          minx, maxx = np.min(lon), np.max(lon)
          miny, maxy = np.min(lat), np.max(lat)
      ?>
      <LatLonBoundingBox minx="${minx}" miny="${miny}" maxx="${maxx}" maxy="${maxy}"></LatLonBoundingBox>
      <BoundingBox CRS="EPSG:4326" minx="${minx}" miny="${miny}" maxx="${maxx}" maxy="${maxy}"/>
      <Dimension py:if="time is not None" name="time" units="ISO8601"/>
      <Extent py:if="time is not None" name="time" default="${time[0].isoformat()}" nearestValue="0">${','.join([t.isoformat() for t in time])}</Extent>
    </Layer>
  </Layer>
</Capability>
</WMT_MS_Capabilities>"""

class OutsideGridException(Exception):
    pass

class WMSResponse(BaseResponse):

    __description__ = "Web Map Service image"

    renderer = GenshiRenderer(
            options={}, loader=StringLoader( {'capabilities.xml': DEFAULT_TEMPLATE} ))

    def __init__(self, dataset):
        BaseResponse.__init__(self, dataset)
        self.headers.append( ('Content-description', 'dods_wms') )

    #@profile
    def __call__(self, environ, start_response):
        query = parse_dict_querystring_lower(environ)

        # Init colors class instance on first invokation
        self._init_colors(environ)

        try:
            dap_query = ['%s=%s' % (k, query[k]) for k in query
                    if k.lower() not in WMS_ARGUMENTS]
            dap_query = [pair.rstrip('=') for pair in dap_query]
            dap_query.sort()  # sort for uniqueness
            dap_query = '&'.join(dap_query)
            location = construct_url(environ,
                    with_query_string=True,
                    querystring=dap_query)
            # Create a Beaker cache dependent on the query string for
            # pre-computed values that depend on the specific dataset
            # We exclude all WMS related arguments since they don't
            # affect the dataset.
            self.cache = environ['beaker.cache'].get_cache(
                    'pydap.responses.wms+' + location)
            # We also create a global cache for pre-computed values
            # that can be shared across datasets
            self.global_cache = environ['beaker.cache'].get_cache(
                    'pydap.responses.wms+global')
            #print self.cache.namespace.keys()
        except KeyError:
            self.cache = None
            self.global_cache = None

        # Support cross-origin resource sharing (CORS)
        self.headers.append( ('Access-Control-Allow-Origin', '*') )

        # Handle GetMap, GetColorbar, GetCapabilities and GetMetadata requests
        type_ = query.get('request', 'GetMap')
        if type_ == 'GetCapabilities':
            self.serialize = self._get_capabilities(environ)
            self.headers.append( ('Content-type', 'text/xml') )
        elif type_ == 'GetMetadata':
            self.serialize = self._get_metadata(environ)
            self.headers.append( ('Content-type', 'application/json; charset=utf-8') )
        elif type_ == 'GetMap':
            self.serialize = self._get_map(environ)
            self.headers.append( ('Content-type', 'image/png') )
        elif type_ == 'GetColorbar':
            self.serialize = self._get_colorbar(environ)
            self.headers.append( ('Content-type', 'image/png') )
        else:
            raise HTTPBadRequest('Invalid REQUEST "%s"' % type_)

        # Check if we should return a 304 Not Modified response
        self._check_last_modified(environ)

        # Set response caching headers
        self._set_caching_headers(environ)

        return BaseResponse.__call__(self, environ, start_response)

    def _check_last_modified(self, environ):
        """Check if resource is modified since last_modified header value."""
        if 'HTTP_IF_MODIFIED_SINCE' in environ:
            cache_control = environ.get('HTTP_CACHE_CONTROL')
            if cache_control != 'no-cache':
                if_modified_since = time.mktime(parsedate(environ.get('HTTP_IF_MODIFIED_SINCE')))
                for header in environ['pydap.headers']:
                    if 'Last-modified' in header:
                        last_modified = time.mktime(parsedate(header[1]))
                        if if_modified_since <= last_modified:
                            raise HTTPNotModified

    def _set_caching_headers(self, environ):
        """Set caching headers"""
        max_age = environ.get('pydap.responses.wms.max_age', None)
        s_maxage = environ.get('pydap.responses.wms.s_maxage', None)
        cc_str = 'public'
        if max_age is not None:
            cc_str = cc_str + ', max-age=%i' % int(max_age)
        if s_maxage is not None:
            cc_str = cc_str + ', s-maxage=%i' % int(s_maxage)
        if max_age is not None or s_maxage is not None:
            self.headers.append( ('Cache-Control', cc_str) )

    def _init_colors(self, environ):
        """Set class variable "colors" on first call"""
        if not hasattr(WMSResponse, 'colors'):
            colorfile = environ.get('pydap.responses.wms.colorfile', None)
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

    def _get_colorbar(self, environ):
        def serialize(dataset):
            # Get query string parameters
            query = parse_dict_querystring_lower(environ)
            cmapname = query.get('cmap', environ.get('pydap.responses.wms.cmap', 'jet'))

            # Check for cached version of colorbar
            if self.global_cache:
                try:
                    output = self.global_cache.get_value(('colorbar', cmapname))
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
            dpi = float(environ.get('pydap.responses.wms.dpi', 80))
            paletted = asbool(environ.get('pydap.responses.wms.paletted', 'false'))

            # Set color bar size depending on orientation
            w, h = 100, 300
            if orientation == 'horizontal':
                w, h = 500, 80

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
            if self.global_cache:
                key = ('colorbar', cmapname)
                self.global_cache.set_value(key, output)

            return [output]

        return serialize

    def _get_colors(self, cmapname, grid):
        """Returns Normalization, Colormap and extend information."""
        if cmapname in self.colors:
            norm = self.colors[cmapname]['norm']
            cmap = self.colors[cmapname]['cmap']
            extend = cmap.colorbar_extend 
        else:
            actual_range = self._get_actual_range(grid)
            norm = Normalize(vmin=actual_range[0], vmax=actual_range[1])
            cmap = get_cmap(cmapname)
            extend = 'neither'
        return norm, cmap, extend

    def _get_actual_range(self, grid):
        try:
            actual_range = self.cache.get_value((grid.id, 'actual_range'))
        except (KeyError, AttributeError):
            try:
                actual_range = grid.attributes['actual_range']
            except KeyError:
                data = gridutils.fix_data(np.asarray(grid.array[:]), grid.attributes)
                actual_range = np.min(data), np.max(data)
            if self.cache:
                key = (grid.id, 'actual_range')
                self.cache.set_value(key, actual_range)
        return actual_range

    #@profile
    def _get_map(self, environ):
        def prep_map(environ):
            # Calculate appropriate figure size.
            query = parse_dict_querystring_lower(environ)
            dpi = float(environ.get('pydap.responses.wms.dpi', 80))
            fill_method = environ.get('pydap.responses.wms.fill_method', 'contourf')
            allow_eval = asbool(environ.get('pydap.responses.wms.allow_eval', 'false'))
            vector_method = environ.get('pydap.responses.wms.vector_method', 'black_vector')
            assert fill_method in ['contour', 'contourf', 'pcolor', 'pcolormesh', 
                                   'pcolorfast']
            w = float(query.get('width', 256))
            h = float(query.get('height', 256))
            time = query.get('time')
            if time == 'current': time = None
            figsize = w/dpi, h/dpi
            bbox = query.get('bbox', None)
            if bbox is not None:
                bbox = [float(v) for v in bbox.split(',')]
            cmapname = query.get('cmap', environ.get('pydap.responses.wms.cmap', 'jet'))
            srs = query.get('srs', 'EPSG:4326')
            if srs == 'EPSG:900913': srs = 'EPSG:3785'
            # Override fill_method by user requested style
            styles = query.get('styles', fill_method)
            if styles in ['contour', 'contourf', 'pcolor', 'pcolormesh', 'pcolorfast']:
                fill_method = styles
            style_elems = styles.split(',')
            vector_color = 'k' 
            if style_elems[0] in ['black_vector', 'black_quiver', 
                                  'black_barbs', 'black_arrowbarbs']:
                vector_method = style_elems[0]
                if len(style_elems) > 1:
                    vector_color = style_elems[1]
            nthin_fill = map(int, 
                environ.get('pydap.responses.wms.fill_thinning', "12,12") \
                .split(','))
            npixels_vector = int(\
                environ.get('pydap.responses.wms.pixels_between_vectors', 40))
            return query, dpi, fill_method, vector_method, vector_color, time, \
                   figsize, bbox, cmapname, srs, styles, w, h, nthin_fill, \
                   npixels_vector, allow_eval
        query, dpi, fill_method, vector_method, vector_color, time, figsize, \
            bbox, cmapname, srs, styles, w, h, nthin_fill, npixels_vector, \
            allow_eval = prep_map(environ)

        #@profile
        def serialize(dataset):
            gridutils.fix_map_attributes(dataset)
            # It is apparently expensive to add an axes in matplotlib - so we cache the
            # axes (it is pickled so it is a threadsafe copy)
            try:
                if not self.global_cache:
                    raise KeyError
                fig = cPickle.loads(self.global_cache.get_value((figsize, 'figure')))
                ax = fig.get_axes()[0]
            except KeyError:
                fig = Figure(figsize=figsize, dpi=dpi)
                ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
                if self.global_cache:
                    key = (figsize, 'figure')
                    self.global_cache.set_value(key, cPickle.dumps(fig))

            # Set transparent background; found through http://sparkplot.org/browser/sparkplot.py.
            if asbool(query.get('transparent', 'true')):
                fig.figurePatch.set_alpha(0.0)
                ax.axesPatch.set_alpha(0.0)
            
            # Per default we only quantize fields if they have less than 
            # 255 colors.
            ncolors = None

            # Plot requested grids (or all if none requested).
            layers = [layer for layer in query.get('layers', '').split(',')
                    if layer] or [var.id for var in walk(dataset, GridType)]
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
                    self._plot_grid(dataset, grids, time, bbox_local, (w, h), ax,
                                    srs, fill_method, nthin_fill, cmapname,
                                    expr=expr, layers=vlayers)
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
                        self._plot_grid(dataset, grid, time, bbox_local, (w, h), ax,
                                        srs, fill_method, nthin_fill, cmapname)
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
                    self._plot_vector_grids(dataset, grids, time, bbox_local, 
                        (w, h), ax, srs, vector_method, vector_color, 
                        npixels_vector)
                    if vector_method not in ['black_barbs', 'black_arrowbarbs']:
                        ncolors = 7

            # Save to buffer.
            if bbox is not None:
                ax.axis( [bbox[0], bbox[2], bbox[1], bbox[3]] )
            ax.axis('off')
            canvas = FigureCanvas(fig)
            # Optionally convert to paletted png
            paletted = asbool(environ.get('pydap.responses.wms.paletted', 'false'))
            if paletted:
                output = plotutils.convert_paletted(canvas, ncolors=ncolors)
            else:
                output = StringIO() 
                canvas.print_png(output)
            if hasattr(dataset, 'close'): dataset.close()
            return [ output.getvalue() ]
        return serialize

    def _prepare_grid(self, srs, bbox, grid, dataset):
        """\
        Calculates various grid parameters.
        """
        # Plot the data over all the extension of the bbox.
        # First we "rewind" the data window to the begining of the bbox:
        lon = gridutils.get_lon(grid, dataset)
        cyclic = hasattr(lon, 'modulo')

        # We assume that lon/lat arrays are centered on data points
        base_srs = 'EPSG:4326'
        do_proj = srs != base_srs
        if do_proj:
            try:
                if not self.global_cache:
                    raise KeyError
                p_base = self.global_cache.get_value((base_srs, 'pyproj'))
            except KeyError:
                p_base = pyproj.Proj(init=base_srs)
                if self.global_cache:
                    key = (base_srs, 'pyproj')
                    self.global_cache.set_value(key, p_base)
            try:
                if not self.global_cache:
                    raise KeyError
                p_query = self.global_cache.get_value((srs, 'pyproj'))
            except KeyError:
                p_query = pyproj.Proj(init=srs)
                if self.global_cache:
                    key = (srs, 'pyproj')
                    self.global_cache.set_value(key, p_query)
        else:
            p_base = None
            p_query = None

        # This operation is expensive - cache results using beaker
        try:
            if not self.cache:
                raise KeyError
            lon_str, lat_str, dlon, do_proj = self.cache.get_value(
                  (grid.id, srs, 'project_data'))
            lon = np.load(StringIO(lon_str))
            lat = np.load(StringIO(lat_str))
        except KeyError:
            # Project data
            lon = np.asarray(lon[:])
            lat = np.asarray(gridutils.get_lat(grid, dataset)[:])
            if not do_proj:
                dlon = 360.0
            else:
                # Fetch projected data from disk if possible
                proj_attr = 'coordinates_' + srs.replace(':', '_')
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
                    lon_str = StringIO()
                    np.save(lon_str, lon)
                    lat_str = StringIO()
                    np.save(lat_str, lat)
                    key = (grid.id, srs, 'project_data')
                    self.cache.set_value(key,
                         (lon_str.getvalue(), lat_str.getvalue(), dlon, do_proj))

        return lon, lat, dlon, cyclic, do_proj, p_base, p_query

    def _find_slices(self, lon, lat, dlon, bbox, cyclic, nthin, size):
        """Returns slicing information for a given input."""
        # Retrieve only the data for the request bbox, and at the 
        # optimal resolution (avoiding oversampling).
        w, h = size
        nthin_lat, nthin_lon = nthin
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

            istep = max(1, int(np.floor( (nthin_lon * lon.shape[0] * (bbox[2]-bbox[0])) / (w * abs(np.amax(lon)-np.amin(lon))) )))
            jstep = max(1, int(np.floor( (nthin_lat * lat.shape[0] * (bbox[3]-bbox[1])) / (h * abs(np.amax(lat)-np.amin(lat))) )))
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
            I, J = np.meshgrid(i, j)

            xcond = (lon >= bbox[0]) & (lon <= bbox[2])
            ycond = (lat >= bbox[1]) & (lat <= bbox[3])
            if (not xcond.any() or not ycond.any()):
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
                if (not xcond.any() or not ycond.any()):
                    raise OutsideGridException

            istep = max(1, int(np.floor( (nthin_lon * lon.shape[1] * (bbox[2]-bbox[0])) / (w * abs(np.amax(lon)-np.amin(lon))) )))
            jstep = max(1, int(np.floor( (nthin_lat * lon.shape[0] * (bbox[3]-bbox[1])) / (h * abs(np.amax(lat)-np.amin(lat))) )))
            i0 = 0
            j0 = 0
            lon1 = lon[j0::jstep,i0::istep]
            lat1 = lat[j0::jstep,i0::istep]
            # Find containing bound in reduced indices
            i, j = np.arange(lon1.shape[1]), np.arange(lon1.shape[0])
            I, J = np.meshgrid(i, j)
            xcond = (lon1 >= bbox[0]) & (lon1 <= bbox[2])
            ycond = (lat1 >= bbox[1]) & (lat1 <= bbox[3])
            if not xcond.any() or not ycond.any():
                raise OutsideGridException
            i0r = np.min(I[xcond])-1
            i1r = np.max(I[xcond])+2
            j0r = np.min(J[ycond])-1
            j1r = np.max(J[ycond])+2
            # Convert reduced indices to global indices
            i0 = max(istep*i0r, 0)
            i1 = min(istep*i1r, lon.shape[1]+1)
            j0 = max(jstep*j0r, 0)
            j1 = min(jstep*j1r, lon.shape[0]+1)

        return (j0, j1, jstep), (i0, i1, istep)

    #@profile
    def _extract_data(self, l, (j0, j1, jstep), (i0, i1, istep), 
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
            gdata = np.asarray(grid.array[l,j0:j1:jstep,i0:i1:istep])
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
            data.append(gdata)

        # Postcondition
        assert X.shape == Y.shape, 'shapes: %s %s' % (X.shape, Y.shape)

        return X, Y, data

    #@profile
    def _plot_vector_grids(self, dataset, grids, tm, bbox, size, ax, srs,
                           vector_method, vector_color, npixels_vector):
        try:
            # Slice according to time request (WMS-T).
            l = gridutils.time_slice(tm, grids[0], dataset)
        except IndexError:
            # Return empty image for out of time range requests
            return
 
        size_ave = np.mean(size)
        nnice = int(size_ave/npixels_vector)
        X1d, Y1d = nice_points(bbox, nnice)
        X, Y = np.meshgrid(X1d, Y1d)
        points_intp = np.vstack((X.flatten(), Y.flatten())).T


        # Extract projected grid information
        lon, lat, dlon, cyclic, do_proj, p_base, p_query = \
                self._prepare_grid(srs, bbox, grids[0], dataset)

        # Now we plot the data window until the end of the bbox:
        while np.min(lon) < bbox[2]:
            lon_save = lon[:]
            lat_save = lat[:]

            # We do not really need the slicing information but we want to
            # see if an OutsideGridException is thrown
            try:
                (j0, j1, jstep), (i0, i1, istep) = \
                self._find_slices(lon, lat, dlon, bbox, cyclic, (1, 1), size)
            except OutsideGridException:
                lon += dlon
                continue
            
            # Interpolate to these points
            lonf = lon.flatten()
            latf = lat.flatten()
            points = np.vstack((lonf, latf)).T
            data = []
            ny, nx = X.shape
            d1 = np.ma.empty((ny, nx))
            d2 = np.ma.empty((ny, nx))
            f = interpolate.NearestNDInterpolator(points, np.empty(lonf.shape))
            f.values = np.asarray(grids[0].array[l,:,:]).squeeze().flatten()
            d1 = f(points_intp).reshape((ny, nx))
            f.values = np.asarray(grids[1].array[l,:,:]).squeeze().flatten()
            d2 = f(points_intp).reshape((ny, nx))
            data = [d1, d2]

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
                    if vector_method == 'black_barbs':
                        ax.barbs(X, Y, data[0], data[1], pivot='middle',
                                 color=vector_color, barb_increments=barb_incs,
                                 linewidth=1, length=7, antialiased=False)
                    elif vector_method == 'black_arrowbarbs':
                        arrow_barbs(ax, X, Y, data[0], data[1], pivot='middle',
                                 color=vector_color, barb_increments=barb_incs,
                                 linewidth=1, length=7, antialiased=False)
                    else:
                        #if vector_method == 'black_quiver':
                        d = np.ma.sqrt(data[0]**2 + data[1]**2)
                        ax.quiver(X, Y, data[0]/d, data[1]/d, pivot='middle',
                                  units='inches', scale=4.0, scale_units='inches',
                                  width=0.04, color=vector_color, linewidths=1,
                                  headlength=4, headaxislength=3.5,
                                  edgecolors=('w'), antialiased=True)
                        
            lon = lon_save
            lat = lat_save
            lon += dlon


    #@profile
    def _plot_grid(self, dataset, grid, time, bbox, size, ax, srs, fill_method,
                   nthin=(5, 5), cmapname='jet', expr=None, layers=None):
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
 
        # Extract projected grid information
        lon, lat, dlon, cyclic, do_proj, p_base, p_query = \
                  self._prepare_grid(srs, bbox, grid, dataset)

        # Now we plot the data window until the end of the bbox:
        while np.min(lon) < bbox[2]:
            lon_save = lon[:]
            lat_save = lat[:]

            try:
                (j0, j1, jstep), (i0, i1, istep) = \
                self._find_slices(lon, lat, dlon, bbox, cyclic, nthin, size)
            except OutsideGridException:
                lon += dlon
                continue
            
            if expr is None:
                X, Y, data = self._extract_data(l, (j0, j1, jstep),
                             (i0, i1, istep), lat, lon, dlon, cyclic, [grid])
                data = data[0]
            else:
                X, Y, datas = self._extract_data(l, (j0, j1, jstep),
                              (i0, i1, istep), lat, lon, dlon, cyclic, grids)
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
                        cmap = get_cmap(cmapname)
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

    def _get_capabilities(self, environ):
        def serialize(dataset):
            # Check for cached version of XML document
            last_modified = None
            for header in environ['pydap.headers']:
                if 'Last-modified' in header:
                    last_modified = header[1]
            if last_modified is not None and self.cache:
                try:
                    key = ('capabilities', last_modified)
                    output = self.cache.get_value(key)[:]
                    if hasattr(dataset, 'close'): dataset.close()
                    return [output]
                except KeyError:
                    pass

            gridutils.fix_map_attributes(dataset)
            grids = [grid for grid in walk(dataset, GridType) if gridutils.is_valid(grid, dataset)]

            # Set global lon/lat ranges.
            try:
                lon_range = self.cache.get_value('lon_range')
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
                    key = 'lon_range'
                    self.cache.set_value(key, lon_range)
            try:
                lat_range = self.cache.get_value('lat_range')
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
                    key = 'lat_range'
                    self.cache.set_value(key, lat_range)

            # Remove ``REQUEST=GetCapabilites`` from query string.
            location = construct_url(environ, with_query_string=True)
            #base = location.split('REQUEST=')[0].rstrip('?&')
            base = location.split('?')[0].rstrip('.wms')

            context = {
                    'dataset': dataset,
                    'location': base,
                    'layers': grids,
                    'lon_range': lon_range,
                    'lat_range': lat_range,
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
            if last_modified is not None and self.cache:
                key = ('capabilities', last_modified)
                self.cache.set_value(key, output[:], expiretime=86400)
            return [output]
        return serialize

    def _get_metadata(self, environ):
        def serialize(dataset):
            # Check for cached version of JSON document
            last_modified = None
            for header in environ['pydap.headers']:
                if 'Last-modified' in header:
                    last_modified = header[1]
            if last_modified is not None and self.cache:
                try:
                    key = ('metadata_all', last_modified)
                    output = self.cache.get_value(key)[:]
                    if hasattr(dataset, 'close'): dataset.close()
                    return [output]
                except KeyError:
                    pass

            gridutils.fix_map_attributes(dataset)
            grids = [grid for grid in walk(dataset, GridType) if gridutils.is_valid(grid, dataset)]

            # Remove ``REQUEST=GetMetadata`` from query string.
            location = construct_url(environ, with_query_string=True)
            base = location.split('?')[0].rstrip('.wms')
            query = parse_dict_querystring_lower(environ)
            layers = [layer for layer in query.get('layers', '').split(',')
                    if layer] # or [var.id for var in walk(dataset, GridType)]
            items = query.get('items', 'epoch').split(',')

            output = {}
            expiretime = 86400
            for layer in layers:
                output[layer] = {}
                attrs = dataset[layer].attributes
                # Only cache epoch, last_modified and time requests for 60 seconds
                if 'units' in items and 'units' in attrs:
                    output[layer]['units'] = attrs['units']
                if 'long_name' in items and 'long_name' in attrs:
                    output[layer]['long_name'] = attrs['long_name']
                if 'bounds' in items:
                    try:
                        if not self.cache:
                            raise KeyError
                        key = ('metadata_units', layer)
                        output[layer]['bounds'] = self.cache.get_value(key)[:]
                    except KeyError:
                        lon = gridutils.get_lon(dataset[layer], dataset)
                        lat = gridutils.get_lat(dataset[layer], dataset)
                        minx, maxx = float(np.min(lon)), float(np.max(lon))
                        miny, maxy = float(np.min(lat)), float(np.max(lat))
                        output[layer]['bounds'] = [minx, miny, maxx, maxy]
                        if self.cache:
                            self.cache.set_value(key, output[layer]['bounds'],
                                                 expiretime=864000)
                if 'time' in items:
                    tm = gridutils.get_time(dataset[layer], dataset)
                    if tm is not None:
                        output[layer]['time'] = [t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in tm]
                        expiretime = 60
                if not output[layer]:
                    del output[layer]
            global_attrs = dataset.attributes['NC_GLOBAL']
            if 'epoch' in items and 'epoch' in global_attrs:
                output['epoch'] = global_attrs['epoch']
                expiretime = 60
            if 'last_modified' in items and last_modified is not None:
                last_modified = datetime.fromtimestamp(
                      time.mktime(parsedate(last_modified))). \
                      strftime('%Y-%m-%dT%H:%M:%SZ')
                output['last_modified'] = last_modified
                expiretime = 60
            output = json.dumps(output)

            if hasattr(dataset, 'close'): dataset.close()
            if last_modified is not None and self.cache:
                key = ('metadata_all', last_modified)
                self.cache.set_value(key, output[:], expiretime=expiretime)
            return [output]
        return serialize

def parse_dict_querystring_lower(environ):
    """Parses query string into dict with keys in lower case."""
    query = parse_dict_querystring(environ)
    # Convert WMS argument keys to lower case
    lowerlist = []
    for k,v in query.iteritems():
        if k.lower() in WMS_ARGUMENTS:
            lowerlist.append(k)
    for k in lowerlist:
        v = query.pop(k)
        query[k.lower()] = v
    return query

def nice_points(bbox, npoints):
    """\
    Returns nice coordinate points for a bbox with a given
    space between. The coordinate points are padded so that
    the bbox is always contained in the points.
    """
    origo = (0.0, 0.0)
    xmin, ymin, xmax, ymax = bbox
    dx = (xmax-xmin)/float(npoints)
    dy = (ymax-ymin)/float(npoints)
    # Calculate start and end of slices
    x1 = int(np.floor(xmin/dx))*dx - dx
    x2 = int(np.ceil(xmax/dx))*dx + dx
    y1 = int(np.floor(ymin/dy))*dy - dy
    y2 = int(np.ceil(ymax/dy))*dy + dy
    xn = np.arange(x1, x2, dx)
    yn = np.arange(y1, y2, dy)
    return xn, yn

