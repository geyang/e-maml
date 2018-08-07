import inspect
import os
from functools import reduce
from os.path import basename, dirname, abspath, join, expanduser

import yaml
from termcolor import cprint

with open(os.path.join(os.path.dirname(__file__), ".yours"), 'r') as stream:
    rc = yaml.load(stream, Loader=yaml.BaseLoader)


class RUN:
    from ml_logger import logger

    server = rc.get('logging_server', expanduser("~/ml-logger-outputs"))
    prefix = f"{rc['username']}/{rc['project']}/{logger.now('%Y/%m-%d')}"


def dir_prefix(depth=-1):
    from ml_logger import logger

    caller_script = abspath(inspect.getmodule(inspect.stack()[1][0]).__file__)
    # note: for scripts in the `plan2vec` module this also works -- b/c we truncate fixed depth.
    script_path = logger.truncate(caller_script, depth=len(__file__.split('/')) - 1)
    prefix = os.path.join(RUN.prefix, script_path)
    return reduce(lambda p, i: dirname(p), range(-depth), prefix)


def config_charts(config_yaml="", path=None):
    from textwrap import dedent
    from ml_logger import logger

    if not config_yaml:
        caller_script = abspath(inspect.getmodule(inspect.stack()[1][0]).__file__)
        if path is None:
            path = logger.stem(caller_script) + ".charts.yml"
        try:  # first try the namesake chart file
            with open(os.path.join(os.path.dirname(caller_script), path), 'r') as s:
                config_yaml = s.read()
                cprint(f"Found ml-dash config file \n{path}", 'green')
        except:  # do not upload when can not find
            path = ".charts.yml"
            with open(os.path.join(os.path.dirname(caller_script), path), 'r') as s:
                config_yaml = s.read()
            cprint(f"Found ml-dash config file \n{path}", 'green')

    logger.log_text(dedent(config_yaml), ".charts.yml")


def thunk(fn, *ARGS, __prefix="", __timestamp='%H.%M/%S.%f', **KWARGS):
    """
    thunk for configuring the logger. The reason why this is not a decorator is

    :param fn: function to be called
    :param *ARGS: position arguments for the call
    :param __prefix: logging prefix for this run, default to "", where it does not do much.
    :param __timestamp: bool, default to True, whether post-fix with time stamps.
    :param **KWARGS: keyword arguments for the call
    :return: a thunk that can be called without parameters
    """
    from ml_logger import logger

    caller_script = abspath(inspect.getmodule(inspect.stack()[1][0]).__file__)
    # note: for scripts in the `plan2vec` module this also works -- b/c we truncate fixed depth.
    script_path = logger.truncate(caller_script, depth=len(__file__.split('/')) - 1)
    _ = [logger.now(__timestamp)] if __timestamp else []
    PREFIX = join(RUN.prefix, logger.stem(script_path), __prefix, *_)

    # todo: there should be a better way to log these.
    # todo: we shouldn't need to log to the same directory, and the directory for the run shouldn't be fixed.
    logger.configure(log_directory=RUN.server, prefix=PREFIX, asynchronous=False,  # use sync logger
                     max_workers=4, register_experiment=False)
    # the tension is in between creation vs run. Code snapshot are shared, but runs need to be unique.
    logger.log_params(
        run=logger.run_info(status="created", script_path=script_path),
        revision=logger.rev_info(),
        fn=logger.fn_info(fn), )
    logger.log_params(args=ARGS, kwargs=KWARGS)
    logger.diff(silent=True)

    import jaynes  # now set the job name to prefix
    if jaynes.RUN.mode != "local":
        runner_class, runner_args = jaynes.RUN.config['runner']
        if 'name' in runner_args:  # ssh mode does not have 'name'.
            runner_args['name'] = PREFIX.replace("geyang/", "")  # destroy my traces.
        del logger, jaynes, runner_args, runner_class
        cprint(f'{__file__}: Set up job name', "green")

    def _(*args, **kwargs):
        import traceback
        from ml_logger import logger

        assert not (args and ARGS), f"can not use position argument at both thunk creation as well as " \
            f"run.\n_args: {args}\nARGS: {ARGS}"

        logger.configure(log_directory=RUN.server, prefix=PREFIX, register_experiment=False, max_workers=10)
        logger.log_params(host=dict(hostname=logger.hostname), run=dict(status="running", startTime=logger.now()))

        try:
            _KWARGS = KWARGS.copy()
            _KWARGS.update(kwargs)

            fn(*(args or ARGS), **_KWARGS)

            logger.log_line("========= execution is complete ==========")
            logger.log_params(run=dict(status="completed", completeTime=logger.now()))
        except Exception as e:
            import time
            time.sleep(1)
            tb = traceback.format_exc()
            with logger.SyncContext():  # Make sure uploaded finished before termination.
                logger.log_text(tb, filename="traceback.err")
                logger.log_params(run=dict(status="error", exitTime=logger.now()))
                logger.log_line(tb)
                logger.flush()
            time.sleep(30)
            raise e

        import time
        time.sleep(30)

    return _
