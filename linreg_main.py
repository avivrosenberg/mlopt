import argparse
import datetime as dt
import os
import sys

import linreg.config as cfg
import linreg.run as run
import linreg.plots as plt

DEFAULT_OUT_DIR = os.path.join('.', 'out', 'linreg')


def create_parser():
    def is_dir(dirname):
        if not os.path.isdir(dirname):
            raise argparse.ArgumentTypeError(f'{dirname} is not a directory')
        else:
            return dirname

    def is_file(filename):
        if not os.path.isfile(filename):
            raise argparse.ArgumentTypeError(f'{filename} is not a file')
        else:
            return filename

    p = argparse.ArgumentParser(
        description='MLOPT linreg',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--out-dir', '-o', type=str,
                   help='Output folder for results and plots',
                   default=DEFAULT_OUT_DIR, required=False)
    p.add_argument('--no-plot-title', action='store_true', required=False,
                   help='Suppress titles in plots')

    sp = p.add_subparsers(dest='subcmd', help='Sub-commands')

    # Multi-experiment run
    sp_multi = sp.add_parser('multi',
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                             help='Run multiple experiments based on config '
                                  'file (default if no sub-command given)')
    sp_multi.set_defaults(subcmd_fn=run_multi)
    sp_multi.add_argument('--cfg-file', '-i', type=is_file,
                          help='experiment configurations file (json)',
                          required=True)
    sp_multi.add_argument('--parallel', action='store_true',
                          help='Run experiments in parallel processes')

    # Single-experiment run
    sp_single = sp.add_parser('single',
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                              help='Run a single experiments based on config '
                                   'provided on the command line')
    sp_single.set_defaults(subcmd_fn=run_single)
    cfg_temp = cfg.ExperimentConfig(name='single-run')
    for k, v in cfg_temp._asdict().items():
        sp_single.add_argument(f'--{k}', type=type(v),
                               help=cfg.EXPERIMENT_PARAMS[k],
                               required=False, default=v)

    return p


def parse_cli(parser: argparse.ArgumentParser):
    parsed = parser.parse_args()
    if parsed.subcmd is None:
        parser.print_help()
        sys.exit()
    return parsed


def run_multi(cfg_file, out_dir=DEFAULT_OUT_DIR, parallel=False, **kw):
    timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(out_dir, timestamp)

    print(f'>>> Multi-experiment run, cfg_file={cfg_file}, out_dir={out_dir}')

    configurations = cfg.load_configs(cfg_file)
    print(f'>>> Running {len(configurations)} configurations: '
          f'{[c.name for c in configurations]}')

    results = run.run_configurations(configurations, parallel)

    results_filename = os.path.join(out_dir, 'results.pickle')

    print(f'>>> Writing results to {results_filename}')
    os.makedirs(out_dir, exist_ok=True)
    cfg.dump_results(results, results_filename)

    print(f'>>> Plotting results')
    plt.plot_experiments(results, out_dir, **kw)


def run_single(out_dir=DEFAULT_OUT_DIR, **kw):
    config_params = {k: v for k, v in kw.items()
                     if k in cfg.EXPERIMENT_PARAMS}
    exp_config = cfg.ExperimentConfig(**config_params)

    print(f'>>> Single-experiment run, config={config_params}')
    result = run.run_single_configuration(exp_config)

    print(f'>>> Saving plots to {out_dir}')
    plt.plot_experiment(result, out_dir, **kw)


if __name__ == '__main__':
    parsed_args = parse_cli(create_parser())
    parsed_args.subcmd_fn(**vars(parsed_args))
