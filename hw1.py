import os

import linreg_main

DEFAULT_CFG_FILE = os.path.join('linreg', 'cfg', 'hw1.json')

if __name__ == '__main__':
    linreg_main.run_multi(DEFAULT_CFG_FILE, parallel=True)

