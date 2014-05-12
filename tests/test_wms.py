#!/usr/bin/env python
# encoding: utf8
"""
This module tests various aspects of the pydap WMS response.
"""
__author__ = "Jesper Baasch-Larsen"
__contact__ = "jla@fcoo.dk"
__copyright__ = "Copyright 2014, Danish Defence"
__license__ = "GNU General Public License v3.0"

# Standard library imports
import logging
import datetime
import copy
import multiprocessing as mp
import time
import calendar
import unittest
import json
import random
import urllib

# External imports

# Local imports
from pydap.handlers.netcdf import Handler
from pydap.responses.wms import WMSResponse

logging.basicConfig(level=logging.CRITICAL)

def start_response_mock(status, response_headers, exc_info=None):
    if status != '200 OK':
        print status
        print response_headers
        print exc_info
        raise ValueError

class TestWMSResponse(unittest.TestCase):
    def setUp(self):
        self.handler = Handler('coads.nc')
        self.base_environ = {'SCRIPT_NAME': '',
                             'REQUEST_METHOD': 'GET', 
                             'PATH_INFO': '/coads.nc.wms',
                             'SERVER_PROTOCOL': 'HTTP/1.1',
                             'pydap.responses.wms.paletted': False}
        self.base_query = {'SERVICE': 'WMS',
                           'REQUEST': 'GetMap',
                           'VERSION': '1.1.1',
                           'STYLES': '',
                           'FORMAT': 'image/png',
                           'TRANSPARENT': 'true',
                           'HEIGHT': 512,
                           'WIDTH': 512,
                           'BBOX': '-180.0,-90.0,180.0,90.0',
                           'CRS': '[object Object]',
                           'BOUNDS': '[object Object]',
                           'SRS': 'EPSG:4326'}

    def test_call(self):
        """\
        Test performance of paletting pngs.
        """
        query = self.base_query.copy()
        query['LAYERS'] = 'SST'
        qs = urllib.urlencode(query)
        env = self.base_environ.copy()
        env['QUERY_STRING'] = qs
        for i in range(100):
            self.time_request(env)
        #print result

    def time_request(self, env):
        """Wraps a request with timing and reporting."""
        start = time.time()
        result = self.handler(env, start_response_mock)
        elapsed = time.time() - start
        print('Request for %s took %f secs' % (env['QUERY_STRING'], elapsed))
        print('Returned response of size %i bytes' % len(result[0]))
        return result

if __name__ == '__main__':
    unittest.main()

