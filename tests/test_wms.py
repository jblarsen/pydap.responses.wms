# encoding: utf8
"""
Various useful utilities for testing the WMS server.
"""
# Standard library imports
import os
import urllib
import io
import cProfile
import time
import timeit
import traceback
import fnmatch
import unittest

# External imports
from pydap.handlers.netcdf import Handler
import pyproj

# Internal imports (present working directory)
import wms_utils

# Special cache setup
from beaker.cache import CacheManager
from beaker.util import parse_cache_config_options
cache_opts = { 'cache.type': 'memory' }
cache = CacheManager(**parse_cache_config_options(cache_opts))

# Override openURL in the WebMapService class to call our handler instead.
# When this is done we can use the WebMapService class on our test handler.
class openURL_mock:
    def __init__(self, path_info, handler):
        """Mock for openURL."""
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
        u = io.StringIO(result[0])
        return u

def start_response_mock(status, response_headers, exc_info=None):
    """Mock for start_response method."""
    if status != '200 OK':
        print(status)
        print(response_headers)
        raise

class WMSResponse(object):
    def __init__(self, datapath):
        """Class for operating on a single WMS source."""
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
                           'VERSION': '1.3.0',
                           'STYLES': '',
                           'FORMAT': 'image/png',
                           'TRANSPARENT': 'true',
                           'HEIGHT': 512,
                           'WIDTH': 512,
                           'BBOX': '-180.0,-90.0,180.0,90.0',
                           'SRS': 'EPSG:4326'}

    def get_layers(self, field_type):
        """Return layers in this WMS."""
        # Mock OWSLib
        openURL = wms_utils.owslib.wms.openURL
        wms_utils.owslib.wms.openURL = openURL_mock(self.path_info, self.handler)
        # Get layers
        layers = wms_utils.get_layers(self.url)
        # Remove coordinate layers (assumed to contain 'EPSG')
        layers = [layer for layer in layers if layer.find('EPSG') == -1]
        # Identify vector fields (currently hardcoded names)
        # and reconstruct layer list based on these
        if field_type == 'vector':
            vlayers = []
            vectors = {'u': 'v', 'u10': 'v10', 'u_velocity': 'v_velocity',
                       'uu': 'vv', 'U10M': 'V10M'}
            for layer in layers:
                if layer in vectors:
                    vlayers.append('%s:%s' % (layer, vectors[layer]))
            layers = vlayers

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
                """
            else:
                print('Image data seems OK')
                """
        return result, t

    def request_images(self, url, layer, check_blank=False):
        """
        Check if the WMS layer at the WMS server specified in the
        URL returns a blank image when at the full extent
        """
        # get list of acceptable CRS' for the layer
        wms = WebMapService(url, version='1.3.0')
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

class TestWMSResponse(unittest.TestCase):
    def setUp(self):
        """Setup test fixture."""
        # Create WMSResponse object for all files in 'subdir'
        subdir = 'data'
        datapaths = []
        for root, dirnames, filenames in os.walk(subdir):
          for filename in fnmatch.filter(filenames, '*.nc'):
              datapaths.append(os.path.join(root, filename))
        self.responses = [WMSResponse(datapath) for datapath in datapaths]
        self.n = 10

    def _time_common(self, field_type):
        """\
        Common method for timing methods.
        field_type is in ['scalar', 'vector']
        """
        t_sum = 0.0
        n_sum = 0
        for wms in self.responses:
            print('Timing %s fields in %s' % (field_type, wms.datapath))
            layer_list = wms.get_layers(field_type=field_type)
            for layer in layer_list:
                srs_list = wms.get_srs(layer=layer.split(':')[0])
                for srs in srs_list:
                    t = 0.0
                    for i in range(self.n):
                        result, t_one = wms.request(layer=layer, srs=srs,
                                                    dotiming=True)
                        t += t_one
                    t /= float(n)
                    print('    Time for layer=%s, srs=%s per plot: %i milliseconds' \
                          % (layer, srs, t))
                    t_sum += t
                    n_sum += 1
        t_ave = t_sum/float(n_sum)
        print('Average time for plotting one layer: %i milliseconds' % t_ave)

    def time_vector(self):
        """Time vector fields."""
        self._time_common('vector')

    def time_scalar(self):
        """Time scalar fields"""
        self._time_common('scalar')

    def _profile_common(self, field_type, filename):
        """\
        Common method for profiling methods.
        field_type is in ['scalar', 'vector']
        """
        pr = cProfile.Profile()
        for wms in self.responses:
            print('Profiling %s fields in %s' % (field_type, wms.datapath))
            layer_list = wms.get_layers(field_type=field_type)
            for layer in layer_list:
                srs_list = wms.get_srs(layer=layer.split(':')[0])
                for srs in srs_list:
                    for i in range(self.n):
                        result, t_one = wms.request(layer=layer, srs=srs,
                                                    doprofiling=True)
                    print('    Profiled layer=%s, srs=%s' % (layer, srs))
        pr.dump_stats(filename)

    def profile_vector(self):
        """Profile vector fields."""
        self._profile_common('vector', 'vector.prof')

    def profile_scalar(self):
        """Profile scalar fields"""
        self._profile_common('scalar', 'scalar.prof')

    def _test_common(self, field_type, validator=None):
        """\
        Common method for test methods.
        field_type is in ['scalar', 'vector']
        """
        for wms in self.responses:
            print('Testing %s fields in %s' % (field_type, wms.datapath))
            layer_list = wms.get_layers(field_type=field_type)
            for layer in layer_list:
                srs_list = wms.get_srs(layer=layer.split(':')[0])
                for srs in srs_list:
                    for i in range(2): # To test caching effects
                        result, t_one = wms.request(layer=layer, srs=srs,
                                                    checkimg=True)
                    print('    Tested layer=%s, srs=%s, n=%i' % (layer, srs, 2))
 
    def test_vector(self):
        """Test vector fields."""
        self._test_common('vector')

    def test_scalar(self):
        """Test scalar fields"""
        self._test_common('scalar')

if __name__ == '__main__':
    #unittest.main()
    wms_response = TestWMSResponse()
    wms_response.setUp()
    wms_response.time_scalar()
    wms_response.time_vector()
    #wms_response.test_scalar()
    #wms_response.test_vector()
    #wms_response.profile_scalar()
    #wms_response.profile_vector()
