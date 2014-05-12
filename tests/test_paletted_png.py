#!/usr/bin/env python
# encoding: utf8
"""
This module tests the performance of converting PNGs to paletted PNGs in
the pydap WMS response.
"""
import urllib
import timeit
from pydap.handlers.netcdf import Handler
from pydap.responses.wms import WMSResponse

def start_response_mock(status, response_headers, exc_info=None):
    if status != '200 OK':
        print status
        print response_headers
        print exc_info
        raise ValueError

class TestWMSResponse(object):
    def setUp(self):
        self.handler = Handler('coads.nc')
        self.base_environ = {'SCRIPT_NAME': '',
                             'REQUEST_METHOD': 'GET', 
                             'PATH_INFO': '/coads.nc.wms',
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

    def time_png(self, paletted=False):
        """Test performance"""
        query = self.base_query.copy()
        env = self.base_environ.copy()
        env['pydap.responses.wms.paletted'] = paletted
        for layer in ['SST', 'AIRT', 'SPEH', 'WSPD', 'UWND', 'VWND', 'SLP']:
            query['LAYERS'] = layer
            qs = urllib.urlencode(query)
            env['QUERY_STRING'] = qs
            self.handler(env, start_response_mock)

    def size_png(self, paletted=False):
        """Test size"""
        query = self.base_query.copy()
        env = self.base_environ.copy()
        env['pydap.responses.wms.paletted'] = paletted
        results = []
        for layer in ['SST', 'AIRT', 'SPEH', 'WSPD', 'UWND', 'VWND', 'SLP']:
            query['LAYERS'] = layer
            qs = urllib.urlencode(query)
            env['QUERY_STRING'] = qs
            result = self.handler(env, start_response_mock)
            filename = 'png_%s_%s.png' % (layer, paletted)
            open(filename, 'wb').write(result[0])
            results.append(result)
        return results

def do_timing():
    # Perform timing
    print('Timing png generation. Please be patient...')
    n = 20
    t1 = timeit.timeit(stmt='wms.time_png(paletted=False)', setup='from __main__ import TestWMSResponse; wms = TestWMSResponse(); wms.setUp()', number=n)
    t2 = timeit.timeit(stmt='wms.time_png(paletted=True)', setup='from __main__ import TestWMSResponse; wms = TestWMSResponse(); wms.setUp()', number=n)
    dt = (t2 - t1)/(7.0*n)*1000 # 7 since we make 7 plots per call
    print('Average cost of paletting one png: %.2f milliseconds' % dt)

def do_image_size():
    # Image size
    wms = TestWMSResponse()
    wms.setUp()
    s1 = 0
    for r in wms.size_png(paletted=False):
        s1 += len(r[0])
    s2 = 0
    for r in wms.size_png(paletted=True):
        s2 += len(r[0])
    ds = (s1 - s2)/float(s1)*100
    print('Total size reduced from %i to %i (%.1f %%)' % (s1, s2, ds))

def do_profile(paletted):
    # Profiling
    wms = TestWMSResponse()
    wms.setUp()
    n = 20
    for i in range(n):
        wms.size_png(paletted=paletted)

if __name__ == '__main__':
    do_timing()
    do_image_size()
    #import cProfile
    #cProfile.run('do_profile(False)', 'wms_nopal.prof')
    #cProfile.run('do_profile(True)', 'wms_pal.prof')
