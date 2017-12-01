"""
Factory function for producing beaker cache WSGI middleware.
"""

from beaker.middleware import CacheMiddleware
def make_cache(app, global_conf, **local_conf):
    """Return beaker cache WSGI app."""
    conf = global_conf.copy()
    conf.update(local_conf)
    return CacheMiddleware(app, conf)
