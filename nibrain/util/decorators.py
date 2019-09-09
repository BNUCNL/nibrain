# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import functools
import time

def timer(func):
    """
    timer decorator
    """    
    @functools.wraps(func)
    def function_timer(*args, **kwargs):
        """
        Timing other function
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start

        msg = "run time for {func} took {time} seconds to finish"
        print(msg.format(func = func.__name__, time = runtime))

        return value
    return function_timer
