import cProfile


def profile_func(fn):
    """
    Usage: Add the @profile_this decorator above the target function. Then read the spat out .prof file
    Can do that with snakeviz by typing "snakeviz file.prof" in terminal
    :param fn: Function to decorate and profile
    :return: 
    """
    def profiled_fn(*args, **kwargs):
        # name for profile dump
        fpath = fn.__name__ + '.prof'
        prof = cProfile.Profile()
        ret = prof.runcall(fn, *args, **kwargs)
        prof.dump_stats(fpath)
        return ret

    return profiled_fn