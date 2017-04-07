import re

INDENT=1
NAME=2
INHERIT=3
SELF=3
in_comment=False
in_func=False
with open('_dynet.pyx','r') as pyx:
    for l in pyx:

        is_func = re.match(r'(\s*)(?:cp)?def (.* )?(.*)\((self)?.*\):',l,re.M | re.I | re.S)
        if is_func:
            if in_func:
                print(indent+'    pass')
            # print(is_func.groups())
            if is_func.group(INDENT) is None:
                indent=""
            else:
                indent=is_func.group(INDENT)
            if is_func.group(NAME+1) is not None:
                name=is_func.group(NAME+1)
                selfp1=1
            else:
                name=is_func.group(NAME)

                selfp1=0
            if is_func.group(SELF+selfp1) is None:
                self_=""
            else:
                self_=is_func.group(SELF+selfp1)

            print(indent+ "def "+name+"("+self_+"):")
            in_func=True
            continue

        is_class = re.match(r'(\s*)cdef class (.*)(\(.*\))?:',l,re.M | re.I | re.S)
        if is_class:
            # print(is_class.groups())
            if is_class.group(INDENT) is None:
                indent=""
            else:
                indent=is_class.group(INDENT)
            if is_class.group(INHERIT) is None:
                inherit=""
            else:
                inherit=is_class.group(INHERIT)
            print(indent+ "class "+is_class.group(NAME)+inherit+":")
        # print("")
        is_comment = '"""' in l
        # if is_func: print(is_comment)
        if is_comment:
            if in_comment:
                print(l[:-1])
            in_comment = not in_comment
        if in_comment:
            print(l[:-1])
            continue
        if in_func:
            print(indent + "    pass")
            in_func=False