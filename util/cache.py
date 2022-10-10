from joblib import Memory

def make_cacheable(obj, memory=None):
    """Use Joblib's Memory mechanism to cache a method or function.

    Args:
        obj (function or object method): Can be a function or instantiated object method.
        memory (Memory object or str, optional): If Memory object, it will get applied,
            If a string, this is the path to cache directory.

    Example:
        The following example applies the cacheable mechanism to a `costly_compute` function.

            import time
            import numpy as np

            def costly_compute(data, column_index=0):
                '''Simulate an expensive computation'''
                time.sleep(5)
                return data[column_index]

            rng = np.random.RandomState(42)
            data = rng.randn(int(1e5), 10)
            start = time.time()
            data_trans = costly_compute(data)
            end = time.time()

            print('The function took {:.2f} s to compute.'.format(end - start))
            print('The transformed data are:')
            print('{}'.format(data_trans))
            print()

            print('Making the data cacheable...')
            costly_compute = make_cacheable(costly_compute, './cachedir')
            start = time.time()
            data_trans = costly_compute(data)
            end = time.time()

            print('The function took {:.2f} s to compute.'.format(end - start))
            print('The transformed data are:')
            print('{}'.format(data_trans))
            print('Still slow, but the results are cached.')
            print()

            print('Execute again:')
            start = time.time()
            data_trans = costly_compute(data)
            end = time.time()

            print('The function took {:.2f} s to compute.'.format(end - start))
            print('The transformed data are:')
            print('{}'.format(data_trans))

            # Delete the cache
            costly_compute.clear(warn=False)

    """
    if memory is None:
        return
    if isinstance(memory, str):
        memory = Memory(location=memory, verbose=0)
    assert isinstance(memory, Memory), 'memory is not a Memory object.'
    
    return memory.cache(obj)
