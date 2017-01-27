import sys

import numpy as np

from pydap.handlers.netcdf import nc, var_attrs

def prepare_netcdf():
    filename = sys.argv[1]
    magnitude,direction = sys.argv[2].split(',')

    if nc.__module__ == 'pupynere':
        raise Exception, "Pupynere cannot open netcdf files in append mode. Please install either PyNIO, netCDF4, Scientific.IO.NetCDF or pynetcdf."

    f = nc(filename, 'a')

    m = f.variables[magnitude]
    d = f.variables[direction]

    # Create vector components
    u = m[:]*np.ma.sin(d[:])
    v = m[:]*np.ma.cos(d[:])

    name = m.long_name.lower()
    f.renameVariable(direction, 'u')
    uvar = f.variables['u']
    del uvar.standard_name
    uvar.long_name = "zonal " + name
    uvar.units = m.units
    f.renameVariable(magnitude, 'v')
    vvar = f.variables['v']
    del vvar.standard_name
    vvar.long_name = "meridional " + name
    vvar.units = uvar.units
    f.close()

if __name__ == '__main__':
    sys.exit(prepare_netcdf())
