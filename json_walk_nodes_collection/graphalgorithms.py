"""Utility functions for operating on graphs and trees."""
import logging

from dhi.dsmatch.util.misc import is_container

# https://stackoverflow.com/a/54000999/394430
def walk_collection(obj, callback=None, _path: list=[], **callback_kwargs):
    """Walk an arbitrarily nested structure of lists and/or dicts such as JSON.

    Args:
        obj (dict, list, set, tuple): Nested container object.
        callback (function, optional): If specified, this is the function that gets called on each node
            when the algorithm has arrived at it. If specified, this function must accept
            at least two first arguments: the current node, and the path to that node. Defaults to None.
        callback_kwargs (dict, optional): If `callback` is used, this permits kwargs to be passed to 
            that function. Defaults to {}.

    Example:

        def print_node(node, path):
            if is_container(node):
                print(f'CONTAINER: {type(node).__name__}, path={path}')
            else:
                print(f'LEAF: {type(node).__name__}, value={node}, path={path}')
            
        obj = {'a': 'A', 'b': 'B', 'alist': ['x', {'y': [1, 2, 3], 'z': ['values', 'of', 'z-key']}]}
        walk_collection(obj, callback=print_node)

    Outputs:

        CONTAINER: dict, path_keys=[]
        LEAF: str, value="A", path_keys=['a']
        LEAF: str, value="B", path_keys=['b']
        CONTAINER: list, path_keys=['alist']
        LEAF: str, value="x", path_keys=['alist']
        CONTAINER: dict, path_keys=['alist']
        CONTAINER: list, path_keys=['alist', 'y']
        LEAF: int, value="1", path_keys=['alist', 'y']
        LEAF: int, value="2", path_keys=['alist', 'y']
        LEAF: int, value="3", path_keys=['alist', 'y']
        CONTAINER: list, path_keys=['alist', 'z']
        LEAF: str, value="values", path_keys=['alist', 'z']
        LEAF: str, value="of", path_keys=['alist', 'z']
        LEAF: str, value="z-key", path_keys=['alist', 'z']

    """
    if isinstance(obj, dict):
        if callback:
            callback(obj, _path, **callback_kwargs)
        for k,v in obj.items():
            if is_container(v) is False:
                if callback:
                    _path.append(k)
                    callback(v, _path, **callback_kwargs)
                    _path.pop()
            elif isinstance(v, list) or isinstance(v, set):
                _path.append(k)
                if callback:
                    callback(v, _path, **callback_kwargs)
                for elem in v:
                    walk_collection(elem, callback, _path, **callback_kwargs)
                _path.pop()
            elif isinstance(v, dict):
                _path.append(k)
                walk_collection(v, callback, _path, **callback_kwargs)
                _path.pop()
            else:
                logging.warning(f'Type "{type(v).__name__}" not recognized: {".".join(_path)}. key={k}, val={v}')
    elif isinstance(obj, list):
        if callback:
            callback(obj, _path, **callback_kwargs)
        for elem in obj:
            walk_collection(elem, callback, _path, **callback_kwargs)
    elif callback and is_container(obj) is False:
        callback(obj, _path, **callback_kwargs)

