# encoding: utf8
"""
Various useful utilities for testing the WMS server.
"""
# Standard library imports
import os
import urllib
import io
import cProfile
import math
import time
import timeit
import traceback
import fnmatch
import unittest
import urllib.parse

# External imports
import lxml.etree as etree
from owslib.wms import WebMapService
from pydap.handlers.netcdf import NetCDFHandler
import pyproj
import webtest

# Internal imports (present working directory)
import wms_utils

class WMSResponse(object):
    def __init__(self, datapath):
        """Class for operating on a single WMS source."""
        self.datapath = datapath
        self.handler = webtest.TestApp(NetCDFHandler(datapath))

        self.path_info = '/' + datapath + '.wms'

        # Find this directory
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.base_env = {
            'pydap.config': {
                'pydap.responses.wms.fill_method': 'contourf',
                'pydap.responses.wms.paletted': True,
                'pydap.responses.wms.allow_eval': True,
                'pydap.responses.wms.colorfile': dir_path + '/colors.json',
                'pydap.responses.wms.styles_file': dir_path + '/styles.json',
                'pydap.responses.wms.max_age': 600,
                'pydap.responses.wms.s_maxage': 93600,
                'pydap.responses.wms.max_image_size': 16777216,
                'pydap.responses.wms.localCache': True,
                'pydap.responses.wms.redis': False,
                'pydap.responses.wms.redis.host': 'localhost',
                'pydap.responses.wms.redis.port': 6379,
                'pydap.responses.wms.redis.db': 0,
                'pydap.responses.wms.redis.redis_expiration_time': 604800,
                'pydap.responses.wms.redis.distributed_lock': True
            }
        }

        self.base_query_map = {'SERVICE': 'WMS',
                               'REQUEST': 'GetMap',
                               'VERSION': '1.3.0',
                               'STYLES': '',
                               'FORMAT': 'image/png',
                               'TRANSPARENT': 'TRUE',
                               'HEIGHT': 512,
                               'WIDTH': 512,
                               'BBOX': '-180.0,-90.0,180.0,90.0',
                               'CRS': 'EPSG:4326'}
        self.base_query_cap = {'SERVICE': 'WMS',
                               'REQUEST': 'GetCapabilities',
                               'VERSION': '1.3.0',}
        #print('Getting Capabilities for %s' % self.path_info)
        env = self.base_env.copy()
        env['QUERY_STRING'] = urllib.parse.urlencode(self.base_query_cap)
        response = self.get(params=self.base_query_cap,
                            extra_environ=env, status=200)
        self.xml = response.normal_body
        try:
            self.wms = WebMapService(self.path_info, xml=self.xml,
                                     version='1.3.0')
        except:
            print('PATH_INFO', self.path_info)
            parser = etree.XMLParser(remove_blank_text=True)
            file_obj = io.BytesIO(self.xml)
            tree = etree.parse(file_obj, parser)
            x_str = etree.tounicode(tree, pretty_print=True)
            print('XML', x_str)
            raise

    def get(self, params=None, headers=None, extra_environ=None, status=None,
            verbose=False):
        if verbose:
           print('Requesting url=%s, params=%s, headers=%s, extra_environ=%s' \
                 % (self.path_info, params, headers, extra_environ))
        try:
            response = self.handler.get(self.path_info, params, headers,
                                        extra_environ, status)
        except webtest.app.AppError as error:
            # Construct URL for easy testing in browser
            url = self.path_info + '?' + urllib.parse.urlencode(params)
            print(url)
            raise
        return response

    def get_layers(self):
        """Return layers in this WMS."""
        # Get layers
        layers = list(self.wms.contents)

        # Remove coordinate layers (assumed to contain 'EPSG')
        #layers = [layer for layer in layers if layer.find('EPSG') == -1]
        return layers

    def get_styles(self, layer):
        """Return styles for a layer in this WMS."""
        return self.wms[layer].styles

    def get_crs(self, layer):
        """Return supported CRS' for this WMS."""
        # get list of acceptable CRS' for the layer
        return self.wms[layer].crsOptions

    def request(self, layer, crs, style, dotiming=False, checkimg=False,
                saveimg=False, profiler=None):
        """Requests WMS image from server and returns image"""
        env = self.base_env.copy()
        query = self.base_query_map.copy()
        bounds = list(self.wms[layer].boundingBoxWGS84)
        if crs != 'EPSG:4326':
            delta = 1.0 # We move 1 degrees away from the poles
            if math.isclose(bounds[1], -90.0):
                bounds[1] += delta
            if math.isclose(bounds[3], 90.0):
                bounds[3] -= delta
        bbox = wms_utils.get_bounding_box(bounds, crs)
        if bbox != '':
            query['BBOX'] = bbox
        query['CRS'] = crs
    
        t = self.wms[layer].timepositions
        if t is not None:
            t = [tv.strip() for tv in t]
            query['TIME'] = t[int(len(t)/2)]

        elevations = self.wms[layer].elevations
        if elevations is not None:
            elevations = [ev.strip() for ev in elevations]
            query['ELEVATION'] = elevations[int(len(elevations)/2)]

        query['LAYERS'] = layer
        query['STYLES'] = style

        t = None
        if dotiming:
            t1 = time.clock()
        if profiler is not None:
            profiler.enable()
        response = self.get(params=query,
                            extra_environ=self.base_env, status=200)
        result = response.body[:]

        if profiler is not None:
            profiler.disable()
        if dotiming:
            t2 = time.clock()
            t = (t2-t1)*1000
        if saveimg:
            with open('tmp/png_%s_%s.png' % (layer, crs), 'wb') as f:
                f.write(result)
        if checkimg:
            is_blank = wms_utils.check_blank(result)
            if is_blank:
                # Construct URL for easy testing in browser
                url = self.path_info + '?' + urllib.parse.urlencode(query)
                msg = "Error: Blank image returned for layer=%s; crs=%s" % (layer, crs)
                msg += "\nQuery: " + url
                raise SystemExit(msg)
                """
            else:
                print('Image data seems OK')
                """
        return result, t

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

    def _time_common(self):
        """\
        Common method for timing methods.
        """
        t_sum = 0.0
        n_sum = 0
        combinations = self._create_combinations()
        n = self.n
        for comb in combinations:
            wms = comb['wms']
            layer = comb['layer']
            crs = comb['crs']
            style = comb['style']
            print('Timing wms=%s, layer=%s, crs=%s, style=%s, n=%i' % \
                  (wms.datapath, layer, crs, style, n))
            t = 0.0
            for i in range(n):
                result, t_one = wms.request(layer=layer, crs=crs,
                                            style=style,
                                            dotiming=True)
                t += t_one
            t /= float(n)
            print('Time per plot: %i milliseconds' % t)
            t_sum += t
            n_sum += 1
        t_ave = t_sum/float(n_sum)
        print('Average time (n=%s) for plotting one layer: %i milliseconds' \
              % (n_sum, t_ave))

    def time(self):
        """Time fields"""
        self._time_common()

    def _profile_common(self, filename):
        """\
        Common method for profiling methods.
        """
        combinations = self._create_combinations()
        n = self.n
        pr = cProfile.Profile()
        for comb in combinations:
            wms = comb['wms']
            layer = comb['layer']
            crs = comb['crs']
            style = comb['style']
            print('Profiling wms=%s, layer=%s, crs=%s, style=%s, n=%i' % \
                  (wms.datapath, layer, crs, style, n))
            for i in range(n):
                result, t_one = wms.request(layer=layer, crs=crs,
                                            style=style,
                                            doprofiling=True)
        pr.dump_stats(filename)

    def profile(self):
        """Profile fields"""
        self._profile_common('pydap.prof')

    def _create_combinations(self):
        """Returns a list of WMS request input data combinations."""
        combinations = []
        for wms in self.responses:
            layer_list = wms.get_layers()
            for layer in layer_list:
                crs_list = wms.get_crs(layer=layer)
                for crs in crs_list:
                    styles_dict = wms.get_styles(layer=layer)
                    styles = list(styles_dict.keys())
                    # We always test the empty (default style)
                    styles.append('')
                    for style in styles:
                        combs = {
                            'wms': wms,
                            'layer': layer,
                            'crs': crs,
                            'style': style
                        }
                        combinations.append(combs)
        return combinations

    def _test_common(self, validator=None):
        """\
        Common method for test methods.
        """
        #from pympler import muppy, summary
        #from pympler import tracker
        #tr = tracker.SummaryTracker()
        combinations = self._create_combinations()
        n = 2 # Run test twice to test caching effects
        #combinations.reverse()
        for comb in combinations:
            #if comb['wms'].path_info != '/data/STENNIS/ACNFS/MAPS_STENNIS_ICE_v005C.nc.wms':
            #if comb['wms'].path_info != '/data/ECMWF/DXD/MAPS_ECMWF_DXD_GLOBAL.nc.wms':
            #    continue
            wms = comb['wms']
            layer = comb['layer']
            crs = comb['crs']
            style = comb['style']
            print('Testing wms=%s, layer=%s, crs=%s, style=%s, n=%i' % \
                  (wms.datapath, layer, crs, style, n))
            for i in range(n):
                result, t_one = wms.request(layer=layer, crs=crs,
                                            style=style,
                                            checkimg=True,
                                            saveimg=True)
                #tr.print_diff()                           
                #summary.print_(summary.summarize(muppy.get_objects()))                          
        print('Tested %i requests' % (len(combinations)*n))
 
    def test(self):
        """Test fields"""
        self._test_common()

if __name__ == '__main__':
    #unittest.main()
    wms_response = TestWMSResponse()
    wms_response.setUp()
    #wms_response.time()
    wms_response.test()
    #wms_response.profile()
