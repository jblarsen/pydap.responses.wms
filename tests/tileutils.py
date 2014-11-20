# encoding: utf8
"""
Utilities for calculating stuff about tiles.
"""
# Standard library imports
import math

# External imports
import numpy as np

def num2deg(xtile, ytile, zoom, tilesize):
    """\
    Return (lat, lng) for NW corner of the input tile for a given
    zoom level and tilesize.

    http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Python
    """
    minzoom = find_minzoom(tilesize)
    n = 2.0 ** (zoom - minzoom)
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def deg2num(lat_deg, lon_deg, zoom, tilesize):
    """\
    Return which tile the input (lat, lng) is in for a given
    zoom level and tilesize.

    http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Python
    """
    minzoom = find_minzoom(tilesize)
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** (zoom - minzoom)
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def find_minzoom(tilesize):
    """Returns lowest zoom level for a given tile size."""
    basesize = 256
    # Preconditions
    assert np.log2(tilesize) >= np.log2(basesize)
    assert np.log2(tilesize) % 1 == 0
    zoomlevel = int(np.log2(tilesize) - np.log2(basesize))
    # Postconditions
    assert zoomlevel >= 0
    return zoomlevel

def find_tilerange(zoomlevel, tilesize):
    """Returns range of tiles at a given zoom level for a given tile size."""
    minzoom = find_minzoom(tilesize)
    result = 2**(zoomlevel - minzoom)
    return result

def tile2bbox(xtile, ytile, zoomlevel, tilesize):
    """Returns bounding box given tile numberings and zoom level."""
    nw = num2deg(xtile, ytile, zoomlevel, tilesize)
    se = num2deg(xtile + 1, ytile + 1, zoomlevel, tilesize)
    bbox = [nw[1], se[0], se[1], nw[0]]
    return bbox

def _point_in_bbox(bbox, point):
    """Returns True if point is strictly inside bbox."""
    lat, lng = point
    west, south, east, north = bbox
    if south < lat < north and west < lng < east:
        return True
    return False

def _bbox_points_in_bbox(bbox_outer, bbox):
    """Returns True if any points in bbox are strictly inside bbox_outer."""
    west, south, east, north = bbox
    sw, se, ne, nw = (south, west), (south, east), (north, east), (north, west)
    corners = [sw, se, ne, nw]
    # Check if any of corners are inside the outer bbox:
    for corner in corners:
        if _point_in_bbox(bbox_outer, corner):
            return True

def _bbox_in_bbox(bbox_outer, bbox):
    """Returns True if bbox is (partially) inside bbox_outer."""
    west, south, east, north = bbox
    if _bbox_points_in_bbox(bbox_outer, bbox):
        return True
    if _bbox_points_in_bbox(bbox, bbox_outer):
        #print 'b', bbox
        return True
    if (bbox_outer[1] < south and bbox_outer[3] > north) and \
       (west < bbox_outer[0] < east and west < bbox_outer[2] < east):
        #print 'c', bbox
        return True
    if (south < bbox_outer[1] < north and south < bbox_outer[3] < north) and \
       (bbox_outer[0] < west and bbox_outer[2] > east):
        #print 'd', bbox
        return True
    return False

def test_bbox_in_bbox():
    """Tests _bbox_in_bbox method."""
    bbox1 = [0.0, 50.0, 30.0, 60.0]
    bbox2 = [10.0, 40.0, 20.0, 70.0]
    bbox3 = [15.0, 41.0, 18.0, 47.0]
    assert not _bbox_in_bbox(bbox3, bbox1)
    assert not _bbox_in_bbox(bbox1, bbox3)
    assert _bbox_in_bbox(bbox2, bbox3)
    assert _bbox_in_bbox(bbox3, bbox2)
    assert _bbox_in_bbox(bbox1, bbox2)
    assert _bbox_in_bbox(bbox2, bbox1)

def bbox2tile_bboxes(bbox_in, zoomlevel, tilesize, maxboxes=None):
    """\
    Returns all possible tile bounding boxes at a given zoom level
    for tiles inside the input bbox.

    If maxboxes is given we only return that many maxboxes selected
    deterministically by striding.
    """
    tilerange = find_tilerange(zoomlevel, tilesize)
    bboxes = []
    for x in range(tilerange):
        for y in range(tilerange):
            bbox = tile2bbox(x, y, zoomlevel, tilesize)
            if _bbox_in_bbox(bbox_in, bbox):
                bboxes.append(bbox)
    if maxboxes is not None:
        n = len(bboxes)
        stride = int(math.ceil(float(n)/maxboxes))
        bboxes = bboxes[::stride]
    return bboxes

def all_bbox2tile_bboxes(bbox_in, maxzoom, tilesize, maxboxes=None):
    """\
    Returns all possible tile bounding boxes up to a given zoom level
    for tiles inside the input bbox.
    """
    minzoom = find_minzoom(tilesize)
    bboxes = {}
    for zoomlevel in range(minzoom, maxzoom + 1):
        bboxes[zoomlevel] = bbox2tile_bboxes(bbox_in, zoomlevel, 
                                             tilesize, maxboxes)
    return bboxes

if __name__ == '__main__':
    #test_bbox_in_bbox()
    tilesize = 512
    minzoom = find_minzoom(tilesize)
    maxzoom = 10 # This is our current max zoom in IFM Maps
    area = [0.0, 40.0, 30.0, 65.0]
    nmax = None
    print('Finding in area %s' % area)
    bboxes = all_bbox2tile_bboxes(area, maxzoom, tilesize, nmax)
    for k, v in bboxes.iteritems():
        print k, len(v)
