from __future__ import division

from beaker.middleware import CacheMiddleware
def make_cache(app, global_conf, **local_conf):
    conf = global_conf.copy()
    conf.update(local_conf)
    return CacheMiddleware(app, conf)
