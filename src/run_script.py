import os
import sys
import pathlib


def run():
    script=pathlib.Path(sys.argv[0]).name
    here = pathlib.Path(__file__).parent.resolve()
    cmd='{}/../{}.sh {}'.format(here, script, ' '.join(sys.argv[1:]))
    sys.exit(os.system(cmd))
