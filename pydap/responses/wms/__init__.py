from __future__ import division

from cStringIO import StringIO
import re
import operator
import bisect
import json
import time
import copy
import cPickle
from rfc822 import parsedate

from paste.request import construct_url, parse_dict_querystring
from paste.httpexceptions import HTTPBadRequest, HTTPNotModified
from paste.util.converters import asbool
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.colors import BoundaryNorm
from matplotlib import rcParams
from matplotlib.colors import from_levels_and_colors
rcParams['xtick.labelsize'] = 'small'
rcParams['ytick.labelsize'] = 'small'
import iso8601
import netcdftime
import pyproj
try:
    from PIL import Image
except:
    PIL = None

from pydap.model import *
from pydap.exceptions import ServerError
from pydap.responses.lib import BaseResponse
from pydap.util.template import GenshiRenderer, StringLoader, TemplateNotFound
from pydap.util.safeeval import expr_eval
from pydap.lib import walk, encode_atom

from arrowbarbs import arrow_barbs
import projutils

WMS_ARGUMENTS = ['request', 'bbox', 'cmap', 'layers', 'width', 'height', 'transparent', 'time',
                 'styles', 'service', 'version', 'format', 'crs', 'bounds', 'srs']

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
          from pydap.responses.wms import get_lon, get_lat, get_time
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

from beaker.middleware import CacheMiddleware
def make_cache(app, global_conf, **local_conf):
    conf = global_conf.copy()
    conf.update(local_conf)
    return CacheMiddleware(app, conf)

class WMSResponse(BaseResponse):

    __description__ = "Web Map Service image"

    renderer = GenshiRenderer(
            options={}, loader=StringLoader( {'capabilities.xml': DEFAULT_TEMPLATE} ))

    def __init__(self, dataset):
        BaseResponse.__init__(self, dataset)
        self.headers.append( ('Content-description', 'dods_wms') )

    def __call__(self, environ, start_response):
        # Create a Beaker cache dependent on the query string, since
        # most (all?) pre-computed values will depend on the specific
        # dataset. We strip all WMS related arguments since they don't
        # affect the dataset.
        query = parse_dict_querystring_lower(environ)

        # Set class variable "colors" on first call
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


        try:
            dap_query = ['%s=%s' % (k, query[k]) for k in query
                    if k.lower() not in WMS_ARGUMENTS]
            dap_query = [pair.rstrip('=') for pair in dap_query]
            dap_query.sort()  # sort for uniqueness
            dap_query = '&'.join(dap_query)
            location = construct_url(environ,
                    with_query_string=True,
                    querystring=dap_query)
            self.cache = environ['beaker.cache'].get_cache(
                    'pydap.responses.wms+' + location)
            #self.cache = environ['beaker.cache'].get_cache('pydap.responses.wms')
        except KeyError:
            self.cache = None

        # Handle GetMap and GetCapabilities requests
        type_ = query.get('request', 'GetMap')
        if type_ == 'GetCapabilities':
            self.serialize = self._get_capabilities(environ)
            self.headers.append( ('Content-type', 'text/xml') )
            self.headers.append( ('Access-Control-Allow-Origin', '*') )
        elif type_ == 'GetMap':
            self.serialize = self._get_map(environ)
            self.headers.append( ('Content-type', 'image/png') )
        elif type_ == 'GetColorbar':
            self.serialize = self._get_colorbar(environ)
            self.headers.append( ('Content-type', 'image/png') )
        else:
            raise HTTPBadRequest('Invalid REQUEST "%s"' % type_)

        # Caching
        max_age = environ.get('pydap.responses.wms.max_age', None)
        s_maxage = environ.get('pydap.responses.wms.s_maxage', None)
        cc_str = 'public'
        if max_age is not None:
            cc_str = cc_str + ', max-age=%i' % int(max_age)
        if s_maxage is not None:
            cc_str = cc_str + ', s-maxage=%i' % int(s_maxage)
        if max_age is not None or s_maxage is not None:
            self.headers.append( ('Cache-Control', cc_str) )
        if 'HTTP_IF_MODIFIED_SINCE' in environ:
            cache_control = environ.get('HTTP_CACHE_CONTROL')
            if cache_control != 'no-cache':
                if_modified_since = time.mktime(parsedate(environ.get('HTTP_IF_MODIFIED_SINCE')))
                for header in environ['pydap.headers']:
                    if 'Last-modified' in header:
                        last_modified = time.mktime(parsedate(header[1]))
                        if if_modified_since <= last_modified:
                            raise HTTPNotModified


        return BaseResponse.__call__(self, environ, start_response)

    def _get_colorbar(self, environ):
        w, h = 100, 300
        query = parse_dict_querystring_lower(environ)
        dpi = float(environ.get('pydap.responses.wms.dpi', 80))
        cmapname = query.get('cmap', environ.get('pydap.responses.wms.cmap', 'jet'))
        orientation = query.get('styles', 'vertical')
        if orientation == 'horizontal':
            w, h = 250, 70
        figsize = w/dpi, h/dpi

        def serialize(dataset):
            fix_map_attributes(dataset)
            fig = Figure(figsize=figsize, dpi=dpi)
            fig.set_facecolor('white')
            fig.set_edgecolor('none')
            if orientation == 'vertical':
                ax = fig.add_axes([0.05, 0.05, 0.35, 0.90])
            else:
                ax = fig.add_axes([0.05, 0.55, 0.90, 0.40])
            if asbool(query.get('transparent', 'true')):
                fig.figurePatch.set_alpha(0.0)
                ax.axesPatch.set_alpha(0.5)

            # Plot requested grids.
            layers = [layer for layer in query.get('layers', '').split(',')
                    if layer] or [var.id for var in walk(dataset, GridType)]
            layer = layers[0]
            names = [dataset] + layer.split('.')
            grid = reduce(operator.getitem, names)

            if cmapname in self.colors:
                norm = self.colors[cmapname]['norm']
                cmap = self.colors[cmapname]['cmap']
                extend = cmap.colorbar_extend 
            else:
                actual_range = self._get_actual_range(grid)
                norm = Normalize(vmin=actual_range[0], vmax=actual_range[1])
                cmap = get_cmap(cmapname)
                extend = 'neither'

            cb = ColorbarBase(ax, cmap=cmap, norm=norm,
                    orientation=orientation, extend=extend)
            fontsize = 0
            if orientation == 'vertical':
                for tick in cb.ax.get_yticklabels():
                    txt = tick.get_text()
                    ntxt = len(txt)
                    fontsize = max(int(0.50*w/ntxt), fontsize)
                    fontsize = min(14, fontsize)
                for tick in cb.ax.get_yticklabels():
                    tick.set_fontsize(fontsize)
                    tick.set_color('black')
            else:
                ticks = cb.ax.get_xticklabels()
                for tick in ticks:
                    txt = tick.get_text()
                    ntxt = len(txt)
                    fontsize = max(int(1.25*w/(len(ticks)*ntxt)), fontsize)
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
            output = StringIO() 
            canvas.print_png(output)
            if hasattr(dataset, 'close'): dataset.close()
            return [ output.getvalue() ]
        return serialize

    def _get_actual_range(self, grid):
        try:
            actual_range = self.cache.get_value((grid.id, 'actual_range'))
        except (KeyError, AttributeError):
            try:
                actual_range = grid.attributes['actual_range']
            except KeyError:
                data = fix_data(np.asarray(grid.array[:]), grid.attributes)
                actual_range = np.min(data), np.max(data)
            if self.cache:
                self.cache.set_value((grid.id, 'actual_range'), actual_range)
        return actual_range

    def _get_map(self, environ):
        def prep_map(environ):
            # Calculate appropriate figure size.
            query = parse_dict_querystring_lower(environ)
            dpi = float(environ.get('pydap.responses.wms.dpi', 80))
            fill_method = environ.get('pydap.responses.wms.fill_method', 'contourf')
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
            return query, dpi, fill_method, vector_method, vector_color, time, \
                   figsize, bbox, cmapname, srs, styles, w, h
        query, dpi, fill_method, vector_method, vector_color, time, figsize, \
            bbox, cmapname, srs, styles, w, h = prep_map(environ)

        def serialize(dataset):
            fix_map_attributes(dataset)
            # It is apparently expensive to add an axes in matplotlib - so we cache the
            # axes (it is pickled so it is a threadsafe copy)
            try:
                fig = cPickle.loads(self.cache.get_value((figsize, 'figure')))
                ax = fig.get_axes()[0]
            except:
                fig = Figure(figsize=figsize, dpi=dpi)
                ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
                if self.cache:
                    self.cache.set_value((figsize, 'figure'), cPickle.dumps(fig))

            # Set transparent background; found through http://sparkplot.org/browser/sparkplot.py.
            if asbool(query.get('transparent', 'true')):
                fig.figurePatch.set_alpha(0.0)
                ax.axesPatch.set_alpha(0.0)
            
            # Plot requested grids (or all if none requested).
            layers = [layer for layer in query.get('layers', '').split(',')
                    if layer] or [var.id for var in walk(dataset, GridType)]
            for layer in layers:
                hlayers = layer.split('.')
                vlayers = hlayers[-1].split(':')
                if len(vlayers) == 1:
                    # Plot scalar field
                    names = [dataset] + hlayers
                    grid = reduce(operator.getitem, names)
                    if is_valid(grid, dataset):
                        if bbox is None:
                            lon = get_lon(grid, dataset)
                            lat = get_lat(grid, dataset)
                            bbox_local = [np.min(lon), np.min(lat), np.max(lon), np.max(lat)]
                        else:
                            bbox_local = bbox[:]
                        self._plot_grid(dataset, grid, time, bbox_local, (w, h), ax,
                                        srs, fill_method, cmapname)
                elif len(vlayers) == 2:
                    # Plot vector field
                    grids = []
                    for vlayer in vlayers:
                        names = [dataset] + hlayers[:-2] + [vlayer]
                        grid = reduce(operator.getitem, names)
                        if not is_valid(grid, dataset):
                            raise HTTPBadRequest('Invalid LAYERS "%s"' % layers)
                        if bbox is None:
                            lon = get_lon(grid, dataset)
                            lat = get_lat(grid, dataset)
                            bbox_local = [np.min(lon), np.min(lat), np.max(lon), np.max(lat)]
                        else:
                            bbox_local = bbox[:]
                        grids.append(grid)
                    self._plot_vector_grids(dataset, grids, time, bbox_local, (w, h),
                                            ax, srs, vector_method, vector_color)

            # Save to buffer.
            if bbox is not None:
                ax.axis( [bbox[0], bbox[2], bbox[1], bbox[3]] )
            ax.axis('off')
            canvas = FigureCanvas(fig)
            output = StringIO() 
            # Optionally convert to paletted png
            paletted = asbool(environ.get('pydap.responses.wms.paletted', 'false'))
            def convert_paletted(canvas, output):
                # Read image
                buf, size = canvas.print_to_buffer()
                im = Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
                # Find number of colors
                colors = im.getcolors(255)
                # Only convert if the number of colors is less than 256
                if colors is not None:
                    ncolors = len(colors)
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
                    im.save(output, 'png', optimize=False, transparency=ncolors)
                else:
                    canvas.print_png(output)
                return output
            if paletted:
                output = convert_paletted(canvas, output)
            else:
                canvas.print_png(output)
            if hasattr(dataset, 'close'): dataset.close()
            return [ output.getvalue() ]
        return serialize

    #@profile
    def _plot_vector_grids(self, dataset, grids, time, bbox, size, ax, srs,
                           vector_method, vector_color):
        # Slice according to time request (WMS-T).
        try:
            l = time_slice(time, grids[0], dataset)
        except IndexError:
            # Return empty image for out of time range requests
            return

        # Plot the data over all the extension of the bbox.
        # First we "rewind" the data window to the begining of the bbox:
        lon = get_lon(grids[0], dataset)
        cyclic = hasattr(lon, 'modulo')
        # contourf assumes that the lon/lat arrays are centered on data points
        lon = np.asarray(lon[:])
        lat = np.asarray(get_lat(grids[0], dataset)[:])

        # Project data
        # This operation is expensive - cache results using beaker
        try:
            lon_str, lat_str, dlon, do_proj = self.cache.get_value(
                  (grids[0].id, 'project_data'))
            lon = np.load(StringIO(lon_str))
            lat = np.load(StringIO(lat_str))
        except:
            lon, lat, dlon, do_proj = projutils.project_data(srs, bbox, lon, lat, cyclic)
            if self.cache:
                lon_str = StringIO()
                np.save(lon_str, lon)
                lat_str = StringIO()
                np.save(lat_str, lat)
                self.cache.set_value((grids[0].id, 'project_data'), 
                                     (lon_str.getvalue(), lat_str.getvalue(), dlon, do_proj))
        base_srs = 'EPSG:4326'
        p_base = pyproj.Proj(init=base_srs)
        p_query = pyproj.Proj(init=srs)

        # Now we plot the data window until the end of the bbox:
        w, h = size
        while np.min(lon) < bbox[2]:
            lon_save = lon[:]
            lat_save = lat[:]
            # Retrieve only the data for the request bbox, and at the 
            # optimal resolution (avoiding oversampling).
            nthin_lon = 36 # Make one vector for every nthin pixels
            nthin_lat = 54 # Make one vector for every nthin pixels
            if len(lon.shape) == 1:
                istep = max(1, int(np.floor( (nthin_lon * len(lon) * (bbox[2]-bbox[0])) / (w * abs(lon[-1]-lon[0])) )))
                jstep = max(1, int(np.floor( (nthin_lat * len(lat) * (bbox[3]-bbox[1])) / (h * abs(lat[-1]-lat[0])) )))
                #i0 = int(istep/3)
                #j0 = int(jstep/3)
                i0 = 0
                j0 = 0
                lon1 = lon[i0::istep]
                lat1 = lat[j0::jstep]
                # Find containing bound in reduced indices
                i0r, i1r = find_containing_bounds(lon1, bbox[0], bbox[2])
                j0r, j1r = find_containing_bounds(lat1, bbox[1], bbox[3])
                # Convert reduced indices to global indices
                i0 = istep*i0r
                i1 = istep*i1r
                j0 = jstep*j0r
                j1 = jstep*j1r
                lons = lon1[i0r:i1r]
                lats = lat1[j0r:j1r]
                data = [np.asarray(grids[0].array[...,j0:j1:jstep,i0:i1:istep]),
                        np.asarray(grids[1].array[...,j0:j1:jstep,i0:i1:istep])]
                # Fix cyclic data.
                if cyclic:
                    lons = np.ma.concatenate((lons, lon[0:1] + dlon), 0)
                    data[0] = np.ma.concatenate((
                        data[0], grids[0].array[...,j0:j1:jstep,0:1]), -1)
                    data[1] = np.ma.concatenate((
                        data[1], grids[1].array[...,j0:j1:jstep,0:1]), -1)

                X, Y = np.meshgrid(lons, lats)

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
                        ycond2 = ((lat[:-1,:] <= bbox[1]) & (lon[1:,:] >= bbox[3]))
                        ycond[:-1,:] = ycond2
                    if (not xcond.any() or not ycond.any()):
                        lon += dlon
                        continue

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
                    return
                i0r = max(np.min(I[xcond])-2, 0)
                i1r = min(np.max(I[xcond])+3, xcond.shape[1])
                j0r = max(np.min(J[ycond])-2, 0)
                j1r = min(np.max(J[ycond])+3, ycond.shape[0])
                # Convert reduced indices to global indices
                i0 = istep*i0r
                i1 = istep*i1r
                j0 = jstep*j0r
                j1 = jstep*j1r
                X = lon1[j0r:j1r,i0r:i1r]
                Y = lat1[j0r:j1r,i0r:i1r]
                data = [grids[0].array[...,j0:j1:jstep,i0:i1:istep],
                        grids[1].array[...,j0:j1:jstep,i0:i1:istep]]

            # Plot data.
            if data[0].shape: 
                # apply time slices
                if l is not None:
                    data = [np.asarray(data[0])[l],
                            np.asarray(data[1])[l]]

                # reduce dimensions and mask missing_values
                data[0] = fix_data(data[0], grids[0].attributes)
                data[1] = fix_data(data[1], grids[1].attributes)

                # plot
                if np.ma.count(data[0]) > 0:
                    if do_proj:
                        # Transform back to lat/lon (can be optimized)
                        lons, lats = pyproj.transform(p_query, p_base, X, Y)
                        u,v = projutils.rotate_vector(srs, data[0], data[1], lons, lats, 
                                            returnxy=False)
                    d = np.ma.sqrt(data[0]**2 + data[1]**2)
                    if vector_method == 'black_barbs':
                        ax.barbs(X, Y, data[0], data[1], pivot='middle',
                                 color=vector_color, antialiased=False)
                    elif vector_method == 'black_arrowbarbs':
                        arrow_barbs(ax, X, Y, data[0], data[1], pivot='middle',
                                 color=vector_color, antialiased=False)
                    else:
                        #if vector_method == 'black_quiver':
                        ax.quiver(X, Y, data[0]/d, data[1]/d, pivot='middle',
                                  units='inches', scale=4.0, scale_units='inches',
                                  width=0.02, color=vector_color, 
                                  antialiased=False)
                        
            lon = lon_save
            lat = lat_save
            lon += dlon


    #@profile
    def _plot_grid(self, dataset, grid, time, bbox, size, ax, srs, fill_method, cmapname='jet'):
        # Slice according to time request (WMS-T).
        try:
            l = time_slice(time, grid, dataset)
        except IndexError:
            # Return empty image for out of time range requests
            return

        # Plot the data over all the extension of the bbox.
        # First we "rewind" the data window to the begining of the bbox:
        lon = get_lon(grid, dataset)
        cyclic = hasattr(lon, 'modulo')
        if fill_method in ['contour', 'contourf']:
            # contourf assumes that the lon/lat arrays are centered on data points
            lon = np.asarray(lon[:])
            lat = np.asarray(get_lat(grid, dataset)[:])
        else:
            # the other fill methods assume that the lon/lat arrays are bounding
            # the data cells
            if not 'bounds' in lon.attributes:
                raise ServerError('Bounds attributes missing in NetCDF file')
            lon_bounds_name = lon.attributes['bounds']
            lon_bounds = np.asarray(dataset[lon_bounds_name][:])
            lon = np.concatenate((lon_bounds[:,0], lon_bounds[-1:,1]), 0)
            lat = get_lat(grid, dataset)
            lat_bounds_name = lat.attributes['bounds']
            lat_bounds = np.asarray(dataset[lat_bounds_name][:])
            lat = np.concatenate((lat_bounds[:,0], lat_bounds[-1:,1]), 0)

        # Project data
        # This operation is expensive - cache results using beaker
        try:
            lon_str, lat_str, dlon, do_proj = self.cache.get_value(
                  (grid.id, 'project_data'))
            lon = np.load(StringIO(lon_str))
            lat = np.load(StringIO(lat_str))
        except:
            lon, lat, dlon, do_proj = projutils.project_data(srs, bbox, lon, lat, cyclic)
            if self.cache:
                lon_str = StringIO()
                np.save(lon_str, lon)
                lat_str = StringIO()
                np.save(lat_str, lat)
                self.cache.set_value((grid.id, 'project_data'), 
                                     (lon_str.getvalue(), lat_str.getvalue(), dlon, do_proj))

        # Now we plot the data window until the end of the bbox:
        w, h = size
        #while np.ma.min(lon) < bbox[2]:
        while np.min(lon) < bbox[2]:
            lon_save = lon[:]
            lat_save = lat[:]
            # Retrieve only the data for the request bbox, and at the 
            # optimal resolution (avoiding oversampling).
            if len(lon.shape) == 1:
                i0, i1 = find_containing_bounds(lon, bbox[0], bbox[2])
                j0, j1 = find_containing_bounds(lat, bbox[1], bbox[3])
                istep = max(1, int(np.floor( (len(lon) * (bbox[2]-bbox[0])) / (w * abs(lon[-1]-lon[0])) )))
                jstep = max(1, int(np.floor( (len(lat) * (bbox[3]-bbox[1])) / (h * abs(lat[-1]-lat[0])) )))
                istep = 1
                jstep = 1
                lons = lon[i0:i1:istep]
                lats = lat[j0:j1:jstep]
                if fill_method in ['contour', 'contourf']:
                    data = grid.array[...,j0:j1:jstep,i0:i1:istep]
                else:
                    data = grid.array[...,j0:j1-1:jstep,i0:i1-1:istep]

                # Fix cyclic data.
                if cyclic and fill_method in ['contour', 'contourf']:
                    lons = np.ma.concatenate((lons, lon[0:1] + dlon), 0)
                    data = np.ma.concatenate((
                        data, grid.array[...,j0:j1:jstep,0:1]), -1)

                X, Y = np.meshgrid(lons, lats)

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
                        lon += dlon
                        continue
                i0 = np.min(I[xcond])-2
                i1 = np.max(I[xcond])+3
                lower = False
                if cyclic:
                    if i0 < 0:
                        lon = np.concatenate((lon[:,i0:]-dlon, lon), axis=1)
                        lat = np.concatenate((lat[:,i0:], lat), axis=1)
                        ii = xcond.shape[1] + i0
                        data = np.ma.concatenate((grid.array[...,ii:], 
                                                  grid.array[...]), axis=-1)
                        i1 = i1 - i0
                        i0 = 0
                        lower = True
                    if i1 > xcond.shape[1]:
                        di = i1 - xcond.shape[1]
                        lon = np.concatenate((lon, lon[:,:di]+dlon), axis=1)
                        lat = np.concatenate((lat, lat[:,:di]), axis=1)
                        if lower:
                            data = np.ma.concatenate((data[...], 
                                                      data[...,:di]), axis=-1)
                        else:
                            data = np.ma.concatenate((grid.array[...], 
                                                      grid.array[...,:di]), axis=-1)
                            lower = True
                else:
                    i0 = max(i0, 0)
                    i1 = min(i1, xcond.shape[1])

                j0 = max(np.min(J[ycond])-2, 0)
                j1 = min(np.max(J[ycond])+3, ycond.shape[0])
                istep = max(1, int(np.floor( (lon.shape[1] * (bbox[2]-bbox[0])) / (w * abs(np.max(lon)-np.amin(lon))) )))
                jstep = max(1, int(np.floor( (lon.shape[0] * (bbox[3]-bbox[1])) / (h * abs(np.max(lat)-np.amin(lat))) )))

                X = lon[j0:j1:jstep,i0:i1:istep]
                Y = lat[j0:j1:jstep,i0:i1:istep]
                if lower:
                    data = data[...,j0:j1:jstep,i0:i1:istep]
                else:
                    data = grid.array[...,j0:j1:jstep,i0:i1:istep]

            # Plot data.
            if data.shape: 
                # apply time slices
                if l is not None:
                    data = np.asarray(data[l])
                else:
                    # FIXME: Do the indexing here if we decide to go for just the first timestep
                    if len(data.shape) > 2:
                        # The array proxy object cannot be sliced prior to conversion
                        data = np.asarray(data)[0]
                    else:
                        data = np.asarray(data)

                # reduce dimensions and mask missing_values
                data = fix_data(data, grid.attributes)

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
                        #newlevels = _contour_levels(levels, cmap.colorbar_extend)
                        #cs = ax.contour(X, Y, data, colors='w', levels=newlevels, 
                        #            antialiased=False)
                        #ax.clabel(cs, inline=1, fontsize=10)
                    elif fill_method == 'contour':
                        newlevels = _contour_levels(levels, cmap.colorbar_extend)
                        cs = plot_method(X, Y, data, colors='black', levels=newlevels, 
                                    antialiased=False)
                        ax.clabel(cs, inline=1, fontsize=10)
                    else:
                        plot_method(X, Y, data, norm=norm, cmap=cmap, antialiased=False)
            lon = lon_save
            lat = lat_save
            lon += dlon

    def _get_capabilities(self, environ):
        def serialize(dataset):
            fix_map_attributes(dataset)
            grids = [grid for grid in walk(dataset, GridType) if is_valid(grid, dataset)]

            # Set global lon/lat ranges.
            try:
                lon_range = self.cache.get_value('lon_range')
            except (KeyError, AttributeError):
                try:
                    lon_range = dataset.attributes['NC_GLOBAL']['lon_range']
                except KeyError:
                    lon_range = [np.inf, -np.inf]
                    for grid in grids:
                        lon = np.asarray(get_lon(grid, dataset)[:])
                        lon_range[0] = min(lon_range[0], np.min(lon))
                        lon_range[1] = max(lon_range[1], np.max(lon))
                if self.cache:
                    self.cache.set_value('lon_range', lon_range)
            try:
                lat_range = self.cache.get_value('lat_range')
            except (KeyError, AttributeError):
                try:
                    lat_range = dataset.attributes['NC_GLOBAL']['lat_range']
                except KeyError:
                    lat_range = [np.inf, -np.inf]
                    for grid in grids:
                        lat = np.asarray(get_lat(grid, dataset)[:])
                        lat_range[0] = min(lat_range[0], np.min(lat))
                        lat_range[1] = max(lat_range[1], np.max(lat))
                if self.cache:
                    self.cache.set_value('lat_range', lat_range)

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
            return [output.encode('utf-8')]
        return serialize


def is_valid(grid, dataset):
    return (get_lon(grid, dataset) is not None and 
            get_lat(grid, dataset) is not None)


def get_lon(grid, dataset):
    def check_attrs(var):
        if (re.match('degrees?_e', var.attributes.get('units', ''), re.IGNORECASE) or
            var.attributes.get('standard_name', '') == 'longitude'):
            return var
        #if (re.match('degrees?_e', var.attributes.get('units', ''), re.IGNORECASE) or
        #    var.attributes.get('axis', '').lower() == 'x' or
        #    var.attributes.get('standard_name', '') == 'longitude'):
        #    return var

    # check maps first
    for dim in grid.maps.values():
        if check_attrs(dim) is not None:
            return dim

    # check curvilinear grids
    if hasattr(grid, 'coordinates'):
        coords = grid.coordinates.split()
        for coord in coords:
            if coord in dataset and check_attrs(dataset[coord].array) is not None:
                return dataset[coord].array

    return None


def get_lat(grid, dataset):
    def check_attrs(var):
        #if (re.match('degrees?_n', var.attributes.get('units', ''), re.IGNORECASE) or
        #    var.attributes.get('axis', '').lower() == 'y' or
        #    var.attributes.get('standard_name', '') == 'latitude'):
        #    return var
        if (re.match('degrees?_n', var.attributes.get('units', ''), re.IGNORECASE) or
            var.attributes.get('standard_name', '') == 'latitude'):
            return var

    # check maps first
    for dim in grid.maps.values():
        if check_attrs(dim) is not None:
            return dim

    # check curvilinear grids
    if hasattr(grid, 'coordinates'):
        coords = grid.coordinates.split()
        for coord in coords:
            if coord in dataset and check_attrs(dataset[coord].array) is not None:
                return dataset[coord].array

    return None


def get_time(grid, dataset):
    for dim in grid.maps.values():
        if ' since ' in dim.attributes.get('units', ''):
            calendar = dim.attributes.get('calendar', 'standard')
            try:
                return netcdftime.num2date(np.asarray(dim), dim.units, calendar)
            except:
                pass

    return None


def fix_data(data, attrs):
    if 'missing_value' in attrs:
        data = np.ma.masked_equal(data, attrs['missing_value'])
    elif '_FillValue' in attrs:
        data = np.ma.masked_equal(data, attrs['_FillValue'])

    if attrs.get('scale_factor'): data *= attrs['scale_factor']
    if attrs.get('add_offset'): data += attrs['add_offset']

    while len(data.shape) > 2:
        ##data = data[0]
        data = np.ma.mean(data, 0)
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

def _contour_levels(levels, extend):
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

def time_slice(time, grid, dataset):
    """Slice according to time request (WMS-T)."""
    if time is not None:
        # NOTE: This is an expensive operation:
        values = np.array(get_time(grid, dataset))
        if len(values.shape) == 0:
            l = None
        else:
            l = np.zeros(values.shape, bool)  # get no data by default
            tokens = time.split(',')
            for token in tokens:
                if '/' in token: # range
                    start, end = token.strip().split('/')
                    start = iso8601.parse_date(start, default_timezone=None)
                    end = iso8601.parse_date(end, default_timezone=None)
                    l[(values >= start) & (values <= end)] = True
                else:
                    instant = iso8601.parse_date(token.strip().rstrip('Z'), default_timezone=None)
                    l[values == instant] = True
    else:
        l = None
    # TODO: Calculate index directly instead of array first
    # We do not need to be able to extract multiple time steps
    if l is not None:
        l = np.where(l == True)[0][0]
    return l
