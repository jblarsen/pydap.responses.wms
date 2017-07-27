"""
Copyright (C) 2012 S. Girvin
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import sys
from PIL import Image
from io import BytesIO, StringIO
import requests
import owslib.wms
from owslib.wms import WebMapService
import pyproj
    
def check_blank(content):
    """
    Uses PIL (Python Imaging Library to check if an image is a single colour - i.e. blank
    Images are loaded via a binary content string
 
    Checks for blank images based on the answer at:
    http://stackoverflow.com/questions/1110403/how-can-i-check-for-a-blank-image-in-qt-or-pyqt
    """
 
    bio = BytesIO(content)
    im = Image.open(bio)
    # we need to force the image to load (PIL uses lazy-loading)
    # otherwise get the following error: AttributeError: 'NoneType' object has no attribute 'bands'
    im.load() 
    bands = im.split()
 
    # check if the image is completely white or black, if other background colours are used
    # these should be accounted for
    is_all_white = all(band.getextrema() == (255, 255)  for band in bands)
    is_all_black = all(band.getextrema() == (0, 0)  for band in bands)
 
    is_blank = (is_all_black or is_all_white)
 
    return is_blank
 
def get_default_parameters():
 
    params = {
        'TRANSPARENT': 'TRUE',
        'SERVICE': 'WMS',
        'VERSION': '1.3.0',
        'REQUEST': 'GetMap',
        'STYLES': '',
        'FORMAT': 'image/png',
        'WIDTH': '256',
        'HEIGHT': '256'}
 
    return params
 
def get_bounding_box(bounds, crs):
    base_crs = 'EPSG:4326'
    if crs != base_crs:
        p_base = pyproj.Proj(init=base_crs)
        p_query = pyproj.Proj(init=crs)

        llcrn = pyproj.transform(p_base, p_query, bounds[0], bounds[1])
        lrcrn = pyproj.transform(p_base, p_query, bounds[2], bounds[1])
        ulcrn = pyproj.transform(p_base, p_query, bounds[0], bounds[3])
        urcrn = pyproj.transform(p_base, p_query, bounds[2], bounds[3])
        xmin = min(llcrn[0], ulcrn[0])
        xmax = min(lrcrn[0], urcrn[0])
        ymin = min(llcrn[1], lrcrn[1])
        ymax = min(ulcrn[1], urcrn[1])
        bounds = (xmin, ymin, xmax, ymax, crs)
        bbox = ",".join([str(b) for b in bounds[:4]])
    else:
        bbox = ",".join([str(b) for b in bounds[:4]])
    return bbox

def get_params_and_bounding_box(url, layer_name, crs):
    params = get_default_parameters()
    
    # get bounding box for the layer
    wms = WebMapService(url, version='1.3.0')
    bounds = wms[layer_name].boundingBoxWGS84
 
    # set the custom parameters for the layer
    params['LAYERS'] = layer_name
    bbox = get_bounding_box(bounds, crs)
    if bbox != '':
        params['BBOX'] = bbox
    params['SRS'] = crs
 
    return params
 
def get_time(url, layer_name):
    # get bounding box for the layer
    wms = WebMapService(url, version='1.3.0')
    return wms[layer_name].timepositions
 
def get_crs(url, layer_name):
    """Return supported CRS' for this layer."""
    # get list of acceptable CRS' for the layer
    wms = WebMapService(url, version='1.3.0')
    crs_list = wms[layer_name].crsOptions
    return crs_list

def get_image(url, layer_name, check_blank=False):
    """
    Check if the WMS layer at the WMS server specified in the
    URL returns a blank image when at the full extent
    """
 
    # get list of acceptable CRS' for the layer
    wms = WebMapService(url, version='1.3.0')
    crs_list = wms[layer_name].crsOptions
    print('Requesting these crs\' %s' % crs_list)

    for crs in crs_list:
        params = get_params_and_bounding_box(url, layer_name, crs)
        resp = requests.get(url, params=params)
        print("The full URL request is '%s'" % resp.url)
 
        # this should be 200
        print("The HTTP status code is: %i" % resp.status_code)
        if resp.status_code != 200:
            raise SystemExit(resp.content)
        print('Status code OK')
 
        if resp.headers['content-type'] == 'image/png':
            if check_blank:
                # a PNG image was returned
                is_blank = check_blank(resp.content)
                if is_blank:
                    raise SystemExit("A blank image was returned!")
                else:
                    print('Image data OK')
        else:
            # if there are errors then these can be printed out here
            raise SystemExit(resp.content)
 
def get_layers(url):
    """
    Get a list of all the WMS layers available on the server.
    """
    wms = WebMapService(url, version='1.3.0')
    layer_names = list(wms.contents)
    return layer_names
    
def request_layers(url):
    """
    Get a list of all the WMS layers available on the server
    and loop through each one to check if it is blank
    """
    layer_names = get_layers(url)
    for l in layer_names:
        print("Checking '%s'..." % l)
        get_image(url, l, check_blank=True)
    
if __name__ == "__main__":
    #args = sys.argv[1:]
    baseurl = 'http://wms-dev01:8080'
    dataurls = ['/NOAA/GFS/NOAA_GFS_VISIBILITY.nc.wms',
                '/NOAA/HYCOM/NOAA_HYCOM_GLOBAL_GREENLAND.nc.wms',
                '/NOAA/HYCOM/NOAA_HYCOM_GLOBAL_MEDSEA.nc.wms',
                '/FCOO/GETM/metoc.dk.2Dvars.1nm.2D.1h.NS1C-v001C.nc.wms',
                '/FCOO/GETM/metoc.dk.velocities.1nm.surface.1h.NS1C-v001C.nc.wms',
                '/FCOO/GETM/metoc.full_dom.2Dvars.3nm.2D.1h.NS1C-v001C.nc.wms',
                '/FCOO/GETM/metoc.full_dom.velocities.surface.3nm.1h.NS1C-v001C.nc.wms',
                '/FCOO/GETM/metoc.idk.2Dvars.600m.2D.1h.DK600-v001C.nc.wms',
                '/FCOO/GETM/metoc.idk.velocities.600m.surface.1h.DK600-v001C.nc.wms',
                '/FCOO/WW3/ww3fcast_sigwave_grd_DKinner_v001C.nc.wms',
                '/FCOO/WW3/ww3fcast_sigwave_grd_NSBaltic_v001C.nc.wms',
                '/DMI/HIRLAM/GETM_DMI_HIRLAM_T15_v004C.nc.wms',
                '/DMI/HIRLAM/metoc.DMI_HIRLAM-S03_NSBALTIC_3NM_v004C.nc.wms',
               ]
    args = [baseurl + dataurl for dataurl in dataurls]
    for arg in args:
        print("Checking WMS url: '%s'" % arg)
        request_layers(arg)
