#!/usr/bin/env python
# encoding: utf8
"""
This module tests various plot methods in the pydap WMS response.
"""
import urllib
import timeit
import traceback
from pydap.handlers.netcdf import Handler
from pydap.responses.wms import WMSResponse

def start_response_mock(status, response_headers, exc_info=None):
    if status != '200 OK':
        print status
        print response_headers
        raise

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

    def time_png(self, fill_method, paletted, srs):
        """Test performance"""
        query = self.base_query.copy()
        query['SRS'] = srs
        env = self.base_environ.copy()
        env['pydap.responses.wms.fill_method'] = fill_method
        env['pydap.responses.wms.paletted'] = paletted
        for layer in ['SST', 'AIRT', 'SPEH', 'WSPD', 'UWND', 'VWND', 'SLP']:
            query['LAYERS'] = layer
            qs = urllib.urlencode(query)
            env['QUERY_STRING'] = qs
            self.handler(env, start_response_mock)

    def size_png(self, fill_method, paletted, srs):
        """Test size"""
        query = self.base_query.copy()
        query['SRS'] = srs
        env = self.base_environ.copy()
        env['pydap.responses.wms.fill_method'] = fill_method
        env['pydap.responses.wms.paletted'] = paletted
        results = []
        for layer in ['SST', 'AIRT', 'SPEH', 'WSPD', 'UWND', 'VWND', 'SLP']:
            query['LAYERS'] = layer
            qs = urllib.urlencode(query)
            env['QUERY_STRING'] = qs
            result = self.handler(env, start_response_mock)
            filename = '%s_%s.png' % (layer, fill_method)
            open(filename, 'wb').write(result[0])
            results.append(result)
        return results

def do_timing():
    # Perform timing
    print('Timing png generation. Please be patient...')
    n = 20
    for srs in ['EPSG:4326', 'EPSG:3857']:
        for fill_method in ['contourf', 'pcolormesh']:
            for paletted in [True, False]:
                stmt = 'wms.time_png(fill_method="%s", paletted="%s", srs="%s")' % (fill_method, paletted, srs)
                t = timeit.timeit(stmt=stmt, setup='from __main__ import TestWMSResponse; wms = TestWMSResponse(); wms.setUp()', number=n)
                t = t/float(7*n)*1000
                print('Time for %s (paletted=%s, srs=%s) per plot: %i milliseconds' % (fill_method, paletted, srs, t))

def do_image_size():
    # Image size
    wms = TestWMSResponse()
    wms.setUp()
    for srs in ['EPSG:4326', 'EPSG:3857']:
        for fill_method in ['contourf', 'pcolormesh']:
            for paletted in [True, False]:
                s = 0
                for r in wms.size_png(fill_method=fill_method, paletted=paletted, srs=srs):
                    s += len(r[0])
                print('Size for %s (paletted=%s, srs=%s): %i bytes' % (fill_method, paletted, srs, s))

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
