import os

import linreg_main

DEFAULT_CFG_FILE = os.path.join('hw1', 'cfg', 'hw1-linreg.json')

if __name__ == '__main__':
    linreg_main.run_multi(DEFAULT_CFG_FILE, parallel=True)

