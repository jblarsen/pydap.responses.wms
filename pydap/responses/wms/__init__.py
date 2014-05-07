from __future__ import division

from StringIO import StringIO
import re
import operator
import bisect
import json
import time
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


WMS_ARGUMENTS = ['request', 'bbox', 'cmap', 'layers', 'width', 'height', 'transparent', 'time']


DEFAULT_TEMPLATE = """<?xml version='1.0' encoding="UTF-8" standalone="no" ?>
<!DOCTYPE WMT_MS_Capabilities SYSTEM "http://schemas.opengis.net/wms/1.1.1/WMS_MS_Capabilities.dtd"
 [
 <!ELEMENT VendorSpecificCapabilities EMPTY>
 ]>

<WMT_MS_Capabilities version="1.1.1"
        xmlns="http://www.opengis.net/wms" 
        xmlns:py="http://genshi.edgewall.org/"
        xmlns:xlink="http://www.w3.org/1999/xlink">

<Service>
  <Name>${dataset.name}</Name>
  <Title>WMS server for ${dataset.attributes.get('long_name', dataset.name)}</Title>
  <OnlineResource xlink:href="$location"></OnlineResource>
</Service>

<Capability>
  <Request>
    <GetCapabilities>
      <Format>application/vnd.ogc.wms_xml</Format>
      <DCPType>
        <HTTP>
          <Get><OnlineResource xlink:href="$location"></OnlineResource></Get>
        </HTTP>
      </DCPType>
    </GetCapabilities>
    <GetMap>
      <Format>image/png</Format>
      <DCPType>
        <HTTP>
          <Get><OnlineResource xlink:href="$location"></OnlineResource></Get>
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
      <Extent py:if="time is not None" name="time" default="${time[0].isoformat()}/${time[-1].isoformat()}" nearestValue="0">${time[0].isoformat()}/${time[-1].isoformat()}</Extent>
    </Layer>
  </Layer>
</Capability>
</WMT_MS_Capabilities>"""


class WMSResponse(BaseResponse):

    __description__ = "Web Map Service image"

    renderer = GenshiRenderer(
            options={}, loader=StringLoader( {'capabilities.xml': DEFAULT_TEMPLATE} ))
    #with open('ifm_colors.json', 'r') as colorfile:
    with open('/home/prod/pydap-server/ifm_colors.json', 'r') as colorfile:
        colors = json.load(colorfile)
        for cname in colors:
            levs = colors[cname]['levels']
            cols = colors[cname]['colors']
            extend = colors[cname]['extend']
            cmap, norm = from_levels_and_colors(levs, cols, extend)
            colors[cname] = {'cmap': cmap, 'norm': norm}
        del levs, cols, extend, cmap, norm, cname

    def __init__(self, dataset):
        BaseResponse.__init__(self, dataset)
        self.headers.append( ('Content-description', 'dods_wms') )

    def __call__(self, environ, start_response):
        # Create a Beaker cache dependent on the query string, since
        # most (all?) pre-computed values will depend on the specific
        # dataset. We strip all WMS related arguments since they don't
        # affect the dataset.
        query = parse_dict_querystring(environ)
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
        except KeyError:
            self.cache = None

        # Handle GetMap and GetCapabilities requests
        type_ = query.get('REQUEST', 'GetMap')
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
        query = parse_dict_querystring(environ)
        dpi = float(environ.get('pydap.responses.wms.dpi', 80))
        figsize = w/dpi, h/dpi
        cmapname = query.get('CMAP', environ.get('pydap.responses.wms.cmap', 'jet'))

        def serialize(dataset):
            fix_map_attributes(dataset)
            fig = Figure(figsize=figsize, dpi=dpi)
            fig.set_facecolor('white')
            fig.set_edgecolor('none')
            ax = fig.add_axes([0.05, 0.05, 0.35, 0.90])
            if asbool(query.get('TRANSPARENT', 'true')):
                fig.figurePatch.set_alpha(0.0)
                ax.axesPatch.set_alpha(0.5)

            # Plot requested grids.
            layers = [layer for layer in query.get('LAYERS', '').split(',')
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
                    orientation='vertical', extend=extend)
            fontsize = 0
            for tick in cb.ax.get_yticklabels():
                txt = tick.get_text()
                ntxt = len(txt)
                fontsize = max(int(0.50*w/ntxt), fontsize)
                fontsize = min(14, fontsize)
            for tick in cb.ax.get_yticklabels():
                tick.set_fontsize(fontsize)
                tick.set_color('black')

            # Decorate colorbar
            if 'units' in grid.attributes and 'long_name' in grid.attributes:
                units = grid.attributes['units']
                long_name = grid.attributes['long_name'].capitalize()
                ax.set_ylabel('%s [%s]' % (long_name, units), fontsize=fontsize)

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
        # Calculate appropriate figure size.
        query = parse_dict_querystring(environ)
        dpi = float(environ.get('pydap.responses.wms.dpi', 80))
        fill_method = environ.get('pydap.responses.wms.fill_method', 'contourf')
        assert fill_method in ['contourf', 'pcolor', 'pcolormesh', 'pcolorfast']
        w = float(query.get('WIDTH', 256))
        h = float(query.get('HEIGHT', 256))
        time = query.get('TIME')
        figsize = w/dpi, h/dpi
        bbox = [float(v) for v in query.get('BBOX', '-180,-90,180,90').split(',')]
        cmapname = query.get('CMAP', environ.get('pydap.responses.wms.cmap', 'jet'))
        srs = query.get('SRS', 'EPSG:4326')

        def serialize(dataset):
            fix_map_attributes(dataset)
            fig = Figure(figsize=figsize, dpi=dpi)
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

            # Set transparent background; found through http://sparkplot.org/browser/sparkplot.py.
            if asbool(query.get('TRANSPARENT', 'true')):
                fig.figurePatch.set_alpha(0.0)
                ax.axesPatch.set_alpha(0.0)
            
            # Plot requested grids (or all if none requested).
            layers = [layer for layer in query.get('LAYERS', '').split(',')
                    if layer] or [var.id for var in walk(dataset, GridType)]
            for layer in layers:
                hlayers = layer.split('.')
                vlayers = hlayers[-1].split(':')
                if len(vlayers) == 1:
                    # Plot scalar field
                    names = [dataset] + hlayers
                    grid = reduce(operator.getitem, names)
                    if is_valid(grid, dataset):
                        self._plot_grid(dataset, grid, time, bbox, (w, h), ax,
                                        srs, fill_method, cmapname)
                elif len(vlayers) == 2:
                    # Plot vector field
                    grids = []
                    for vlayer in vlayers:
                        names = [dataset] + hlayers[:-2] + [vlayer]
                        grid = reduce(operator.getitem, names)
                        if not is_valid(grid, dataset):
                            raise HTTPBadRequest('Invalid LAYERS "%s"' % layers)
                        grids.append(grid)
                    self._plot_vector_grids(dataset, grids, time, bbox, (w, h),
                                            ax, srs)

            # Save to buffer.
            ax.axis( [bbox[0], bbox[2], bbox[1], bbox[3]] )
            ax.axis('off')
            canvas = FigureCanvas(fig)
            output = StringIO() 
            # Optionally convert to paletted png
            paletted = asbool(environ.get('pydap.responses.wms.paletted', 'false'))
            if paletted:
                # Read image
                buf, size = canvas.print_to_buffer()
                im = Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
                # Find number of colors
                colors = im.getcolors(256)
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
            else:
                canvas.print_png(output)
            if hasattr(dataset, 'close'): dataset.close()
            return [ output.getvalue() ]
        return serialize

    def _plot_vector_grids(self, dataset, grids, time, bbox, size, ax, srs):
        # Slice according to time request (WMS-T).
        if time is not None:
            values = np.array(get_time(grids[0], dataset))
            l = np.zeros(len(values), bool)  # get no data by default

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

        # Plot the data over all the extension of the bbox.
        # First we "rewind" the data window to the begining of the bbox:
        lon = get_lon(grids[0], dataset)
        cyclic = hasattr(lon, 'modulo')
        # contourf assumes that the lon/lat arrays are centered on data points
        lon = np.asarray(lon[:])
        lat = np.asarray(get_lat(grids[0], dataset)[:])

        # Project data
        base_srs = 'EPSG:4326'
        do_proj = srs != base_srs
        if do_proj:
            p_base = pyproj.Proj(init=base_srs)
            p_query = pyproj.Proj(init=srs)
            if len(lon.shape) == 1:
                lon, lat = np.meshgrid(lon, lat)
            lon, lat = pyproj.transform(p_base, p_query, lon, lat)
            is_latlong = p_query.is_latlong()
        else:
            is_latlong = True

        while is_latlong and np.min(lon) > bbox[0]:
            lon -= 360.0
        # Now we plot the data window until the end of the bbox:
        w, h = size
        while np.min(lon) < bbox[2]:
            # Retrieve only the data for the request bbox, and at the 
            # optimal resolution (avoiding oversampling).
            nthin = 36 # Make one vector for every nthin pixels
            if len(lon.shape) == 1:
                istep = max(1, int(np.floor( (nthin * len(lon) * (bbox[2]-bbox[0])) / (w * abs(lon[-1]-lon[0])) )))
                jstep = max(1, int(np.floor( (nthin * len(lat) * (bbox[3]-bbox[1])) / (h * abs(lat[-1]-lat[0])) )))
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
                if cyclic and is_latlong:
                    lons = np.ma.concatenate((lons, lon[0:1] + 360.0), 0)
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
                if is_latlong and (not xcond.any() or not ycond.any()):
                    lon += 360.0
                    continue

                istep = max(1, int(np.floor( (nthin * lon.shape[1] * (bbox[2]-bbox[0])) / (w * abs(np.max(lon)-np.amin(lon))) )))
                jstep = max(1, int(np.floor( (nthin * lon.shape[0] * (bbox[3]-bbox[1])) / (h * abs(np.max(lat)-np.amin(lat))) )))
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
                if data[0].any():
                    if do_proj:
                        # Transform back to lat/lon (can be optimized)
                        lons, lats = pyproj.transform(p_query, p_base, X, Y)
                        u,v = rotate_vector(srs, data[0], data[1], lons, lats, 
                                            returnxy=False)
                    d = np.sqrt(data[0]**2 + data[1]**2)
                    ax.quiver(X, Y, data[0]/d, data[1]/d, pivot='middle',
                              units='inches', scale=4.0, scale_units='inches',
                              width=0.02, antialiased=False)
            if not is_latlong:
                break
            else:
                lon += 360.0


    def _plot_grid(self, dataset, grid, time, bbox, size, ax, srs, fill_method, cmapname='jet'):
        # Slice according to time request (WMS-T).
        if time is not None:
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

        # Plot the data over all the extension of the bbox.
        # First we "rewind" the data window to the begining of the bbox:
        lon = get_lon(grid, dataset)
        cyclic = hasattr(lon, 'modulo')
        if fill_method == 'contourf':
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
        base_srs = 'EPSG:4326'
        do_proj = srs != base_srs
        if do_proj:
            p_base = pyproj.Proj(init=base_srs)
            p_query = pyproj.Proj(init=srs)
            if len(lon.shape) == 1:
                lon, lat = np.meshgrid(lon, lat)
            lon, lat = pyproj.transform(p_base, p_query, lon, lat)
            is_latlong = p_query.is_latlong()
        else:
            is_latlong = True

        while is_latlong and np.min(lon) > bbox[0]:
            lon -= 360.0
        # Now we plot the data window until the end of the bbox:
        w, h = size
        while np.min(lon) < bbox[2]:
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
                if fill_method == 'contourf':
                    data = np.asarray(grid.array[...,j0:j1:jstep,i0:i1:istep])
                else:
                    data = np.asarray(grid.array[...,j0:j1-1:jstep,i0:i1-1:istep])

                # Fix cyclic data.
                if cyclic and is_latlong and fill_method == 'contourf':
                    lons = np.ma.concatenate((lons, lon[0:1] + 360.0), 0)
                    data = np.ma.concatenate((
                        data, grid.array[...,j0:j1:jstep,0:1]), -1)
                # FIXME: convert lon[0:1] to local proj instead

                X, Y = np.meshgrid(lons, lats)

            elif len(lon.shape) == 2:
                i, j = np.arange(lon.shape[1]), np.arange(lon.shape[0])
                I, J = np.meshgrid(i, j)

                xcond = (lon >= bbox[0]) & (lon <= bbox[2])
                ycond = (lat >= bbox[1]) & (lat <= bbox[3])
                if is_latlong and (not xcond.any() or not ycond.any()):
                    lon += 360.0
                    continue

                if not xcond.any() or not ycond.any():
                    return
                i0 = max(np.min(I[xcond])-2, 0)
                i1 = min(np.max(I[xcond])+3, xcond.shape[1])
                j0 = max(np.min(J[ycond])-2, 0)
                j1 = min(np.max(J[ycond])+3, ycond.shape[0])
                istep = max(1, int(np.floor( (lon.shape[1] * (bbox[2]-bbox[0])) / (w * abs(np.max(lon)-np.amin(lon))) )))
                jstep = max(1, int(np.floor( (lon.shape[0] * (bbox[3]-bbox[1])) / (h * abs(np.max(lat)-np.amin(lat))) )))

                X = lon[j0:j1:jstep,i0:i1:istep]
                Y = lat[j0:j1:jstep,i0:i1:istep]
                data = grid.array[...,j0:j1:jstep,i0:i1:istep]
                # FIXME: Greenwich Meridian bug around here...

            # Plot data.
            if data.shape: 
                # apply time slices
                if l is not None:
                    data = np.asarray(data)[l]
                else:
                    data = np.asarray(data)

                # reduce dimensions and mask missing_values
                data = fix_data(data, grid.attributes)

                # plot
                if data.any():
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
                        plot_method(X, Y, data, norm=norm, cmap=cmap, 
                                    levels=levels, antialiased=False,)
                    else:
                        plot_method(X, Y, data, norm=norm, cmap=cmap, antialiased=False)
            if not is_latlong:
                break
            else:
                lon += 360.0

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
            base = location.split('REQUEST=')[0].rstrip('?&')

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

def rotate_vector(srs,uin,vin,lons,lats,returnxy=False):
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
    base_srs = 'EPSG:4326'
    p_base = pyproj.Proj(init=base_srs)
    p_query = pyproj.Proj(init=srs)
    x, y = pyproj.transform(p_base, p_query, lons, lats)
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
    xn, yn = pyproj.transform(p_base, p_query, lon1, lat1)

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

