"""
Module containing a more restricted asteval.Interpreter class
"""
# Standard library imports
from __future__ import division, print_function
from sys import stdout
import math

# External imports
from asteval import Interpreter
HAS_NUMPY = False
try:
    import numpy
    HAS_NUMPY = True
except ImportError:
    pass

# Constants
FROM_PY = ()
FROM_MATH = ()
FROM_NUMPY = ('fmax', 'fmin', 'sign')
NUMPY_RENAMES = {}
LOCALFUNCS = {}

class RestrictedInterpreter(Interpreter):
    """\
    Mathematical expression compiler and interpreter.

    This is a more restricted version of asteval.Interpreter.
    """
    supported_nodes = ('binop', 'call', 'expr', 'module', 'name', 'num')

    def __init__(self, symtable=None, writer=None, use_numpy=True):
        self.writer = writer or stdout

        if symtable is None:
            symtable = {}
        self.symtable = symtable
        self._interrupt = None
        self.error = []
        self.error_msg = None
        self.expr = None
        self.retval = None
        self.lineno = 0
        self.use_numpy = HAS_NUMPY and use_numpy

        for sym in FROM_PY:
            if sym in __builtins__:
                symtable[sym] = __builtins__[sym]

        for symname, obj in LOCALFUNCS.items():
            symtable[symname] = obj

        for sym in FROM_MATH:
            if hasattr(math, sym):
                symtable[sym] = getattr(math, sym)

        if self.use_numpy:
            for sym in FROM_NUMPY:
                if hasattr(numpy, sym):
                    symtable[sym] = getattr(numpy, sym)
            for name, sym in NUMPY_RENAMES.items():
                if hasattr(numpy, sym):
                    symtable[name] = getattr(numpy, sym)

        self.node_handlers = dict(((node, getattr(self, "on_%s" % node))
                                   for node in self.supported_nodes))

if __name__ == '__main__':
    import numpy as np
    a = np.ones((5,5))
    b = np.zeros((5,5))
    raeval = RestrictedInterpreter()
    raeval.symtable['a'] = a
    raeval.symtable['b'] = b
    expr = 'fmin(100, fmax(0, fmax(15.3*a+4.3, 23.1*b-4.3)))'
    data = raeval(expr)
    print(data)
    expr = 'fmin(100, fmax(0, fmax(-15.3*sign(a)+4.3, 23.1*sign(b)-4.3)))'
    data = raeval(expr)
    print(data)
    expr = 'print("should throw NotImplementedError")'
    raeval(expr)
