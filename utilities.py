import datetime
import os


def get_time_string():
    return datetime.datetime.now().strftime("%d%m%Y-%H%M%S-%f")


def get_log_dir(name, env, LOG_DIR_ROOT):
    try:
        os.mkdir(LOG_DIR_ROOT)
    except:
        pass
    return "{3}/{0}_{1}_{2}".format(name,
                                    env,
                                    get_time_string(),
                                    LOG_DIR_ROOT)


def parse_time_string(s):
    return datetime.datetime.strptime(s, '%d%m%Y-%H%M%S-%f')
