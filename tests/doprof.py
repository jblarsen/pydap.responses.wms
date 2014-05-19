#!/usr/bin/env python
import sys
import pstats
p = pstats.Stats(sys.argv[1])
p.sort_stats('cumulative').print_stats(20)
p.sort_stats('time').print_stats(20)
#p.print_callers(20)

