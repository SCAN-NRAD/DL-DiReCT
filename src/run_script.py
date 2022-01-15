import os
import sys
import pathlib


def direct():
    here = pathlib.Path(__file__).parent.resolve()
    cmd='{}/../direct.sh {}'.format(here, ' '.join(sys.argv[1:]))
    os.system(cmd)

def dl_direct():
    here = pathlib.Path(__file__).parent.resolve()
    cmd='{}/../dl+direct.sh {}'.format(here, ' '.join(sys.argv[1:]))
    os.system(cmd)
