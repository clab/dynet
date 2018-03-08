from __future__ import print_function
import re

INDENT = 1
NAME = 2
INHERIT = 3
ARGUMENTS = 3
PASS='    pass\n'
def pythonize_arguments(arg_str):
    """
    Remove types from function arguments in cython
    """
    out_args = []
    # If there aren't any arguments return the empty string
    if arg_str is None:
        return out_str
    args = arg_str.split(',')
    for arg in args:
        components = arg.split('=')
        name_and_type=components[0].split(' ')
        # There is probably type info
        if name_and_type[-1]=='' and len(name_and_type)>1:
            name=name_and_type[-2]
        else:
            name=name_and_type[-1]
        # if there are default parameters
        if len(components)>1:
            name+='='+components[1]

        out_args.append(name)
    return ','.join(out_args)


def get_indent(indent_str):
    """
    Check if the indent exists
    """
    if indent_str is None:
        return ''
    else:
        return indent_str


def get_inherit(inherit_str):
    """
    Check if there is a parent class
    """
    if inherit_str is None:
        return ''
    else:
        return inherit_str


def get_func_name(func_str):
    """
    Get function name, ie removes possible return type
    """
    name = func_str.split(' ')[-1]
    return name


def create_doc_copy(in_file='../../python/_dynet.pyx', out_file='dynet.py'):

    in_comment = False
    in_func = False

    with open(out_file, 'w+') as py:
        with open(in_file, 'r') as pyx:
            for l in pyx:
                # Check if this line is a function declaration (def or cpdef)
                is_func = re.match(r'(\s*)(?:cp)?def (.*)\((.*)\):', l, re.I)
                if is_func:
                    # If the previous line was a function, print pass
                    if in_func:
                        print(indent + PASS, file=py)
                    # Preserve indentation
                    indent = get_indent(is_func.group(INDENT))
                    # Get function name
                    name = get_func_name(is_func.group(NAME))
                    # Get arguments
                    arguments = pythonize_arguments(is_func.group(ARGUMENTS))
                    # Print declaration
                    print(indent + "def "+name+"("+arguments+"):", file=py)
                    # Now in function body
                    in_func = True
                    continue
                # Check if this line declares a class
                is_class = re.match(r'(\s*)(?:cdef )?class (.*)(\(.*\))?:', l, re.I)
                if is_class:
                    # Preserve indentation
                    indent = get_indent(is_class.group(INDENT))
                    # Get parent class
                    inherit = get_inherit(is_class.group(INHERIT))
                    # Print declaration
                    print(indent + "class "+is_class.group(NAME)+inherit+":", file=py)
                # Handle comments (better)
                is_comment = re.match(r'(\s*)"""(.*)', l, re.I) or ('"""' in l and in_comment) # This last case is to account for end of line """ to end the comment
                # If start or beginning of comment
                if is_comment:
                    # If end of comment, print the """
                    if in_comment:
                        print(l[:-1], file=py)
                    # Toggle in_comment indicator
                    in_comment = not in_comment
                    # If this is a single line comment, end in_comment scope
                    if l.count('"""') > 1:
                        in_comment = False
                # Print comment line
                if in_comment:
                    print(l[:-1], file=py)
                    continue
                # If not in comment anymore but still in function scope, print pass
                if in_func:
                    print(indent + PASS, file=py)
                    in_func = False
