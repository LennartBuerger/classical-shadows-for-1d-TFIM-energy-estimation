import time


def timed(timing_dict):
    def internal_timed(f):
        def timed_f(self, *args, **kwargs):
            start_time = time.time()
            result = f(self, *args, **kwargs)
            exec_time = time.time() - start_time
            if f.__name__ not in timing_dict:
                timing_dict[f.__name__] = 0.0
            timing_dict[f.__name__] += exec_time

            return result

        return timed_f

    return internal_timed
