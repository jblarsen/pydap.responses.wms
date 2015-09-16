#!/usr/bin/env python
# encoding: utf8
"""
This module tests the performance of the WMS server and produces a profile for
further analysis.
"""
import urllib
import urlparse
import StringIO
import cProfile
import time
import timeit
import traceback
from pydap.handlers.netcdf import Handler
import pyproj
import wms_utils

from beaker.cache import CacheManager
from beaker.util import parse_cache_config_options

cache_opts = { 'cache.type': 'memory' }
cache = CacheManager(**parse_cache_config_options(cache_opts))

# Override openURL in the WebMapService class to call our handler instead.
# When this is done we can use the WebMapService class on our test handler.
class openURL_mock:
    def __init__(self, path_info, handler):
        self.handler = handler
        self.base_env = {'SCRIPT_NAME': '',
                         'REQUEST_METHOD': 'GET', 
                         'PATH_INFO': path_info,
                         'beaker.cache': cache,
                         'pydap.responses.wms.fill_method': 'contourf',
                         'pydap.responses.wms.paletted': True,
                         'wsgi.url_scheme': 'http',
                         'SERVER_NAME': 'dummy_server',
                         'SERVER_PORT': '90000',
                         'SERVER_PROTOCOL': 'HTTP/1.1'}

    def __call__(self, url_base, data, method='Get', cookies=None, 
                 username=None, password=None, timeout=30):
        """\
        Function to open urls - used to override the openURL method in OWSlib 
        with a version which calls our handler instead.
        """
        env = self.base_env.copy()
        env['QUERY_STRING'] = data
        result = self.handler(env, start_response_mock)
        u = StringIO.StringIO(result[0])
        return u

def start_response_mock(status, response_headers, exc_info=None):
    if status != '200 OK':
        print status
        print response_headers
        raise

class WMSResponse(object):
    def __init__(self, datapath):
        self.datapath = datapath
        self.handler = Handler(datapath)
        self.url = 'http://fakeurl.wms'
        self.path_info = '/' + datapath + '.wms'

        # Monkey patching openURL with our own version of it
        self.base_env = {'SCRIPT_NAME': '',
                         'REQUEST_METHOD': 'GET', 
                         'PATH_INFO': self.path_info,
                         'beaker.cache': cache,
                         'pydap.responses.wms.fill_method': 'contourf',
                         'pydap.responses.wms.paletted': True,
                         'wsgi.url_scheme': 'http',
                         'SERVER_NAME': 'dummy_server',
                         'SERVER_PORT': '90000',
                         'SERVER_PROTOCOL': 'HTTP/1.1'}
        self.base_query = {'SERVICE': 'WMS',
                           'REQUEST': 'GetMap',
                           'VERSION': '1.1.1',
                           'STYLES': '',
                           'FORMAT': 'image/png',
                           'TRANSPARENT': 'true',
                           'HEIGHT': 512,
                           'WIDTH': 512,
                           'BBOX': '-180.0,-90.0,180.0,90.0',
                           'SRS': 'EPSG:4326'}

    def get_layers(self):
        """Return layers in WMS."""
        # Mock OWSLib
        openURL = wms_utils.owslib.wms.openURL
        wms_utils.owslib.wms.openURL = openURL_mock(self.path_info, self.handler)
        # Get layers
        layers = wms_utils.get_layers(self.url)
        # Restore mocked object
        wms_utils.owslib.wms.openURL = openURL
        return layers

    def get_srs(self, layer):
        """Return supported SRS' for this WMS."""
        # Mock OWSLib
        openURL = wms_utils.owslib.wms.openURL
        wms_utils.owslib.wms.openURL = openURL_mock(self.path_info, self.handler)
        # Get layers
        crs_list = wms_utils.get_crs(self.url, layer)
        # Restore mocked object
        wms_utils.owslib.wms.openURL = openURL
        return crs_list

    #@profile
    def request(self, layer, srs, dotiming=False, checkimg=False, saveimg=False,
                profiler=None):
        """Requests WMS image from server and returns image"""
        # For vector request only forward first layer
        flayer = layer.split(':')[0]

        # Mock OWSLib
        openURL = wms_utils.owslib.wms.openURL
        wms_utils.owslib.wms.openURL = openURL_mock(self.path_info, self.handler)
        env = self.base_env.copy()
        query = wms_utils.get_params_and_bounding_box(self.url, flayer, srs)
        t = wms_utils.get_time(self.url, flayer)
        if t is not None:
            query['TIME'] = t[len(t)/2]
        # Do not use flayer but layer
        query['LAYERS'] = layer
        qs = urllib.urlencode(query)
        env['QUERY_STRING'] = qs
        t = None
        if dotiming:
            t1 = time.clock()
        if profiler is not None:
            profiler.enable()
        result = self.handler(env, start_response_mock)
        if profiler is not None:
            profiler.disable()
        if dotiming:
            t2 = time.clock()
            t = (t2-t1)*1000
        wms_utils.owslib.wms.openURL = openURL
        if saveimg:
            open('tmp/png_%s_%s.png' % (layer, srs), 'wb').write(result[0])
        if checkimg:
            is_blank = wms_utils.check_blank(result[0])
            if is_blank:
                msg = "Error: Blank image returned for layer=%s; srs=%s" % (layer, srs)
                msg += "\nQuery string: " + qs
                raise SystemExit(msg)
            else:
                print('Image data seems OK')
        return result, t

    def request_images(self, url, layer, check_blank=False):
        """
        Check if the WMS layer at the WMS server specified in the
        URL returns a blank image when at the full extent
        """
        # get list of acceptable CRS' for the layer
        wms = WebMapService(url, version='1.1.1')
        crs_list = wms[layer].crsOptions
        print('Requesting these crs\' %s' % crs_list)
        for crs in crs_list:
            params = get_params_and_bounding_box(url, layer, crs)
            env = self.base_env.copy()
            env['QUERY_STRING'] = params
            resp = handler(env, start_response_mock)
            if check_blank:
                # a PNG image was returned
                is_blank = check_blank(resp)
                if is_blank:
                    raise SystemExit("A blank image was returned!")
                else:
                    print('Image data OK')

class TestWMSResponse(object):
    def setUp(self):
        #self.handler = Handler('data/NOAA/HYCOM/NOAA_HYCOM_GLOBAL_GREENLAND.nc')
        """
        datapaths= ['data/NOAA/GFS/NOAA_GFS_VISIBILITY.nc',
                    'data/NOAA/HYCOM/NOAA_HYCOM_GREENLAND.nc',
                    'data/NOAA/HYCOM/NOAA_HYCOM_MEDSEA.nc',
                    'data/FCOO/GETM/metoc.dk.2Dvars.1nm.2D.1h.NS1C-v001C.nc',
                    'data/FCOO/GETM/metoc.dk.velocities.1nm.surface.1h.NS1C-v001C.nc',
                    'data/FCOO/GETM/metoc.full_dom.2Dvars.3nm.2D.1h.NS1C-v001C.nc',
                    'data/FCOO/GETM/metoc.full_dom.velocities.surface.3nm.1h.NS1C-v001C.nc',
                    'data/FCOO/GETM/metoc.idk.2Dvars.600m.2D.1h.DK600-v001C.nc',
                    'data/FCOO/GETM/metoc.idk.velocities.600m.surface.1h.DK600-v001C.nc',
                    'data/FCOO/WW3/ww3fcast_sigwave_grd_DKinner_v001C.nc',
                    'data/FCOO/WW3/ww3fcast_sigwave_grd_NSBaltic_v001C.nc',
                    'data/DMI/HIRLAM/GETM_DMI_HIRLAM_T15_v004C.nc',
                    'data/DMI/HIRLAM/METPREP_DMI_HIRLAM_S03_NSBALTIC3NM_v003C.nc']
        """
        datapaths= ['data/NOAA/HYCOM/NOAA_HYCOM_EAST_AFRICA.nc',
                    'data/NOAA/HYCOM/NOAA_HYCOM_GREENLAND.nc',
                    'data/NOAA/HYCOM/NOAA_HYCOM_MEDSEA.nc',
                    'data/FCOO/GETM/idk.2Dvars.600m.2D.1h.DK600-v004C.nc',
                    'data/FCOO/GETM/idk.salt-temp.600m.surface.1h.DK600-v004C.nc',
                    'data/FCOO/GETM/idk.velocities.600m.surface.1h.DK600-v004C.nc',
                    'data/FCOO/GETM/nsbalt.2Dvars.1nm.2D.1h.DK1NM-v002C.nc',
                    'data/FCOO/GETM/nsbalt.salt-temp.1nm.surface.1h.DK1NM-v002C.nc',
                    'data/FCOO/GETM/nsbalt.velocities.1nm.surface.1h.DK1NM-v002C.nc',
                    'data/ECMWF/DXD/MAPS_ECMWF_DXD_AFR.nc',
                    'data/ECMWF/DXD/MAPS_ECMWF_DXD_DENMARK.nc',
                    'data/ECMWF/DXD/MAPS_ECMWF_DXD_GREENLAND.nc',
                    'data/ECMWF/DXP/MAPS_ECMWF_DXP_AFR.nc',
                    'data/ECMWF/DXP/MAPS_ECMWF_DXP_DENMARK.nc',
                    'data/ECMWF/DXP/MAPS_ECMWF_DXP_GREENLAND.nc',
                    'data/DMI/HIRLAM/DMI_HIRLAM_K05.nc',
                    'data/DMI/HIRLAM/GETM_DMI_HIRLAM_T15_v004C.nc',
                    'data/DMI/HIRLAM/METPREP_DMI_HIRLAM_S03_NSBALTIC3NM_v003C.nc']
        self.responses = [WMSResponse(datapath) for datapath in datapaths]
        #self.responses = self.responses[:4]

def make_requests(n=1, wmslist=None, layerlist=None, srslist=None, dotiming=False, 
                  doprofiling=False, checkimg=False, saveimg=False):
    """Make requests for all WMS data and all layers."""
    wms_response = TestWMSResponse()
    wms_response.setUp()
    if doprofiling:
        pr = cProfile.Profile()
    else:
        pr = None
    #wms = [wr for wr in wms_response.responses if wr.datapath == 'data/FCOO/WW3/ww3fcast_sigwave_grd_NSBaltic_v001C.nc'][0]
    #wms.request(layer='u', srs='EPSG:4326')
    t_sum = 0.0
    n_sum = 0
    if wmslist is None:
        wmslist = wms_response.responses
    else:
        wmslist = [wr for wr in wms_response.responses if wr.datapath in wmslist]
    for wms in wmslist:
        print('Processing %s' % wms.datapath)
        if layerlist is None:
            layer_list = wms.get_layers()
        else:
            layer_list = layerlist[:]
        for layer in layer_list:
            flayer = layer.split(':')[0]
            if srslist is None:
                try:
                    srs_list = wms.get_srs(layer=flayer)
                except:
                    #print('Could not find %s in this WMS' % layer)
                    continue
            else:
                srs_list = srslist[:]
            for srs in srs_list:
                t = 0.0
                for i in range(n):
                    result, tone = wms.request(layer=layer, srs=srs, dotiming=dotiming,
                             checkimg=checkimg, saveimg=saveimg,
                             profiler=pr)
                    if dotiming:
                        t += tone
                if dotiming:
                    t /= float(n)
                    print('    Time for layer=%s, srs=%s per plot: %i milliseconds' \
                          % (layer, srs, t))
                    t_sum += t
                    n_sum += 1
    if doprofiling:
        pr.dump_stats('wms.prof')

    if dotiming:
        t_ave = t_sum/float(n_sum)
        print('Average time for plotting one layer: %i milliseconds' % t_ave)

if __name__ == '__main__':
    import cProfile
    #make_requests(n=3, dotiming=True)
    #make_requests(n=3, doprofiling=True)
    #make_requests(n=1, doprofiling=False)
    #make_requests(n=3, checkimg=True, saveimg=True)
    # Check vector plots
    datapaths= ['data/NOAA/HYCOM/NOAA_HYCOM_EAST_AFRICA.nc',
                'data/NOAA/HYCOM/NOAA_HYCOM_GREENLAND.nc',
                'data/NOAA/HYCOM/NOAA_HYCOM_MEDSEA.nc',
                'data/FCOO/GETM/idk.2Dvars.600m.2D.1h.DK600-v004C.nc',
                'data/FCOO/GETM/idk.salt-temp.600m.surface.1h.DK600-v004C.nc',
                'data/FCOO/GETM/idk.velocities.600m.surface.1h.DK600-v004C.nc',
                'data/FCOO/GETM/nsbalt.2Dvars.1nm.2D.1h.DK1NM-v002C.nc',
                'data/FCOO/GETM/nsbalt.salt-temp.1nm.surface.1h.DK1NM-v002C.nc',
                'data/FCOO/GETM/nsbalt.velocities.1nm.surface.1h.DK1NM-v002C.nc',
                'data/ECMWF/DXD/MAPS_ECMWF_DXD_AFR.nc',
                'data/ECMWF/DXD/MAPS_ECMWF_DXD_DENMARK.nc',
                'data/ECMWF/DXD/MAPS_ECMWF_DXD_GREENLAND.nc',
                'data/ECMWF/DXP/MAPS_ECMWF_DXP_AFR.nc',
                'data/ECMWF/DXP/MAPS_ECMWF_DXP_DENMARK.nc',
                'data/ECMWF/DXP/MAPS_ECMWF_DXP_GREENLAND.nc',
                'data/DMI/HIRLAM/DMI_HIRLAM_K05.nc',
                'data/DMI/HIRLAM/GETM_DMI_HIRLAM_T15_v004C.nc',
                'data/DMI/HIRLAM/METPREP_DMI_HIRLAM_S03_NSBALTIC3NM_v003C.nc']
    #layers = ['u10:v10', 'u_velocity:v_velocity', 'uu:vv', 'u:v', 'U10M:V10M']
    """
    datapaths= ['data/NOAA/HYCOM/NOAA_HYCOM_EAST_AFRICA.nc',
                'data/NOAA/HYCOM/NOAA_HYCOM_GREENLAND.nc',
                'data/NOAA/HYCOM/NOAA_HYCOM_MEDSEA.nc',
                'data/FCOO/GETM/idk.2Dvars.600m.2D.1h.DK600-v004C.nc',
                'data/FCOO/GETM/idk.salt-temp.600m.surface.1h.DK600-v004C.nc',
                'data/FCOO/GETM/idk.velocities.600m.surface.1h.DK600-v004C.nc',
                'data/FCOO/GETM/nsbalt.2Dvars.1nm.2D.1h.DK1NM-v002C.nc',
                'data/FCOO/GETM/nsbalt.salt-temp.1nm.surface.1h.DK1NM-v002C.nc',
                'data/FCOO/GETM/nsbalt.velocities.1nm.surface.1h.DK1NM-v002C.nc',
                'data/DMI/HIRLAM/DMI_HIRLAM_K05.nc',
                'data/DMI/HIRLAM/GETM_DMI_HIRLAM_T15_v004C.nc',
                'data/DMI/HIRLAM/METPREP_DMI_HIRLAM_S03_NSBALTIC3NM_v003C.nc']
    """
    #datapaths= ['data/NOAA/HYCOM/NOAA_HYCOM_GREENLAND.nc']
    datapaths= ['data/ECMWF/DXD/MAPS_ECMWF_DXD_AFR.nc',
                'data/ECMWF/DXD/MAPS_ECMWF_DXD_DENMARK.nc',
                'data/ECMWF/DXD/MAPS_ECMWF_DXD_GREENLAND.nc']
    #datapaths= ['data/FCOO/GETM/idk.velocities.600m.surface.1h.DK600-v004C.nc']
    layers = ['U10', 'V10', 'windspeed']
    #make_requests(n=10, wmslist=datapaths, layerlist=layers, doprofiling=False, saveimg=False)
    #make_requests(n=4, wmslist=datapaths, layerlist=layers, dotiming=False, doprofiling=False, saveimg=False)
    #make_requests(n=2, wmslist=datapaths, layerlist=layers, dotiming=False, doprofiling=True, saveimg=False)
    make_requests(n=10, wmslist=datapaths, layerlist=layers, dotiming=True, doprofiling=False, saveimg=False)
