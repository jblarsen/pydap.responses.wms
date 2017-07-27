from setuptools import setup, find_packages
import sys, os

version = '0.6'

install_requires=[
    # List your project dependencies here.
    # For more details, see:
    # http://packages.python.org/distribute/setuptools.html#declaring-dependencies
    'pydap',
    'Numpy',
    'Paste',
    'matplotlib',
    'coards',
    'iso8601',
    'beaker',
    'Pillow',
    'asteval',
    'lru-dict',
]

setup(version=version,
    name='pydap.responses.wms',
    description='WMS response for Pydap',
    long_description='''
Pydap is an implementation of the Opendap/DODS protocol, written from
scratch. This response enables Pydap to serve data as a WMS server.
    ''',
    keywords='wms opendap dods dap data science climate oceanography meteorology',
    classifiers=[
      # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    ],
    author='Roberto De Almeida',
    author_email='roberto@dealmeida.net',
    url='https://github.com/jblarsen/pydap.responses.wms',
    license='MIT',
    packages=find_packages('src'),
    package_dir = {'': 'src'},
    namespace_packages = ['pydap', 'pydap.responses'],
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    entry_points="""
        [pydap.response]
        wms = pydap.responses.wms:WMSResponse
    """,
)
