"""
Utilities for XML parsing and operations.

`..._gen()` functions are generators that typically walk the tree. Depending on the context, these functions may
need to be wrapped with `list()` so that they fully run.

We also use [Python's XPath functionality](https://docs.python.org/3.8/library/xml.etree.elementtree.html#xpath-support),
especially with findall() to find all nodes that match a particular query. 
If we want to call recursively on each match from that query, we use map(). 
The examples in the docstrings illustrate these concepts.
"""

import xml.etree.ElementTree as ET
import boto3

def print_node(node):
    """Print the XML tree from the given element node.

    Args:
        node (xml.etree.ElementTree.Element): The root to start printing from.
    """
    print(ET.tostring(node, encoding='utf8').decode('utf8'))
    
# def print_attrib_key_gen(node, key, depth_=0):
#     """Generator function to iteratively and recursively print children that have an attribute with a given key.
#     If the key does not exist as an attribute of the child, it is ignored.

#     Args:
#         node (ElementTree node): The root to start printing from.
#         key (str): The attribute to print
#     """
#     try:
#         print('  '*depth + node.attrib[key])
#     except:
#         pass
#     for m in n:
#         yield from print_attrib_key_gen(m, key, depth_=depth + 1)
        
def print_tags_gen(node, _depth=0):
    """Print the tags of all the children starting from `node`. 

    Args:
        node (xml.etree.ElementTree.Element): The root to start printing from.

    Example:

            list(print_tags_gen(root));

    """
    astr = '  '*_depth + node.tag
    print(astr)
    for n in node:
        yield from print_tags_gen(n, _depth=_depth + 1)

def print_detailed_node_gen(node, _depth=0):
    """Print the tag, attributes, and text of children starting from `node`.

    Args:
        n (xml.etree.ElementTree.Element): The node to print (and its subtree.)

    Example:

            list(print_detailed_node_gen(root));

    """
    astr = '  '*_depth + node.tag + str(node.attrib)
    if node.text:
        astr += f': {node.text.strip()}'
    print(astr)
    for n in node:
        yield from print_detailed_node_gen(n, _depth=_depth + 1)

def read_xml_file(path):
    """Given a complete file path or S3 URL, read the file as XML and return the ElementTree.

    Args:
        path (str): Local file path (relative or absolute), or S3 URL. 
        If the path is an S3 URL, provide it in the format `s3://bucket-name/path/that/leads/to/some.xml`.
    """
    if not path.lower().endswith('.xml'):
        raise ValueError(f'The path is "{path}", which does not end in ".xml"')
    if path.lower().startswith('s3://'):
        bucket = path.split('://')[-1].split('/')[0]
        key = path.split(bucket + '/')[-1]
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket, key)
        as_str = obj.get()["Body"].read().decode('utf-8')
    else:
        with open(path, 'r') as f:
            as_str = f.read()    
    return ET.ElementTree(ET.fromstring(as_str))


