import sys

import numpy as np

from pydap.handlers.netcdf import nc, var_attrs

def prepare_netcdf():
    filename = sys.argv[1]
    uname,vname = sys.argv[2].split(',')

    if nc.__module__ == 'pupynere':
        raise Exception, "Pupynere cannot open netcdf files in append mode. Please install either PyNIO, netCDF4, Scientific.IO.NetCDF or pynetcdf."

    f = nc(filename, 'a')

    # create new magnitude variable
    whitelist = ['long_name', 'units', '_FillValue', 'missing_value']
    u = f.variables[uname]
    v = f.variables[vname]
    fill_value = u.missing_value
    uv = np.ma.sqrt(u[:]**2 + v[:]**2)
    uvname = "%s_%s" % (uname, vname)
    uvf = f.createVariable(uvname, u.dtype, u.dimensions, fill_value=fill_value)
    uvf[:] = uv
    for attr in u.ncattrs():
        if attr in whitelist:
            attr_value = u.getncattr(attr)
            if attr == 'long_name':
                attr_value = attr_value.split(' ', 1)[1]
            uvf.setncattr(attr, attr_value)
    f.close()

if __name__ == '__main__':
    sys.exit(prepare_netcdf())
