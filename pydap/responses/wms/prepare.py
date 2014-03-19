import sys

import numpy as np

from pydap.handlers.netcdf import nc, var_attrs
from pydap.responses.wms import fix_data


def prepare_netcdf():
    filename = sys.argv[1]

    if nc.__module__ == 'pupynere':
        raise Exception, "Pupynere cannot open netcdf files in append mode. Please install either PyNIO, netCDF4, Scientific.IO.NetCDF or pynetcdf."

    f = nc(filename, 'a')

    # set actual range
    for name, var in f.variables.items():
        if name in f.dimensions or hasattr(var, 'actual_range'): continue
        data = fix_data(np.asarray(var[:]), var_attrs(var)) 
        var.actual_range = np.amin(data), np.amax(data)

    # create bounds for 1d dimension variables
    for name, dim in f.dimensions.items():
        var = f.variables[name]
        if hasattr(var, 'bounds'): continue
        ndims = len(var.shape)
        # We currently only do this for 1d variables
        if ndims > 1: continue
        vert_dim = u'nv'
        if vert_dim not in f.dimensions.keys():
            nv = f.createDimension(vert_dim, 2*ndims)
        bnds_name = name + '_bnds'
        var.bounds = bnds_name
        bdims = var.dimensions + (vert_dim,)
        bounds = f.createVariable(bnds_name, var.dtype, bdims)
        dv = 0.5*abs(var[-1]-var[0])/(len(var)-1)
        lbounds = var[:] - dv
        ubounds = var[:] + dv
        bounds[:,0] = lbounds
        bounds[:,1] = ubounds

    f.close()

if __name__ == '__main__':
    sys.exit(prepare_netcdf())
