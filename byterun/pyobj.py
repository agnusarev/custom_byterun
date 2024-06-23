"""Implementations of Python fundamental objects for Byterun."""

import collections
import dis
import inspect
import linecache
import re
from copy import copy
from sys import stderr
import types


def make_cell(value):
    # Thanks to Alex Gaynor for help with this bit of twistiness.
    # Construct an actual cell object by creating a closure right here,
    # and grabbing the cell object out of the function we create.
    fn = (lambda x: lambda: x)(value)
    return fn.__closure__[0]


class Function(object):
    __slots__ = [
        "func_code",
        "func_name",
        "func_defaults",
        "func_globals",
        "func_locals",
        "func_dict",
        "func_closure",
        "__name__",
        "__dict__",
        "__doc__",
        "_vm",
        "_func",
    ]

    def __init__(self, name, code, globs, defaults, kwdefaults, closure, vm):
        self._vm = vm
        self.func_code = code
        self.func_name = self.__name__ = name or code.co_name
        self.func_defaults = defaults
        self.func_globals = globs
        self.func_locals = self._vm.frame.f_locals
        self.__dict__ = {}
        self.func_closure = closure
        self.__doc__ = code.co_consts[0] if code.co_consts else None

        # Sometimes, we need a real Python function.  This is for that.
        kw = {
            "argdefs": self.func_defaults,
        }
        if closure:
            kw["closure"] = tuple(make_cell(0) for _ in closure)
        self._func = types.FunctionType(code, globs, **kw)

    def __repr__(self):  # pragma: no cover
        return f"<Function {self.func_name} at 0x{id(self):.08x}>"

    def __get__(self, instance, owner):
        if instance is not None:
            return Method(instance, owner, self)
        else:
            return self

    def __call__(self, *args, **kwargs):
        if re.search(r"<(?:listcomp|setcomp|dictcomp|genexpr)>$", self.func_name):
            # D'oh! http://bugs.python.org/issue19611 Py2 doesn't know how to
            # inspect set comprehensions, dict comprehensions, or generator
            # expressions properly.  They are always functions of one argument,
            # so just do the right thing.  Py3.4 also would fail without this
            # hack, for list comprehensions too. (Haven't checked for other 3.x.)
            assert len(args) == 1 and not kwargs, "Surprising comprehension!"
            callargs = {".0": args[0]}
        else:
            callargs = inspect.getcallargs(self._func, *args, **kwargs)
        frame = self._vm.make_frame(self.func_code, callargs, self.func_globals, {}, self.func_closure)
        CO_GENERATOR = 32  # flag for "this code uses yield"
        if self.func_code.co_flags & CO_GENERATOR:
            gen = Generator(frame, self._vm)
            frame.generator = gen
            retval = gen
        else:
            retval = self._vm.run_frame(frame)
        return retval


class Method(object):
    def __init__(self, obj, _class, func):
        self.im_self = obj
        self.im_class = _class
        self.im_func = func

    def __repr__(self):  # pragma: no cover
        name = "%s.%s" % (self.im_class.__name__, self.im_func.func_name)
        if self.im_self is not None:
            return f"<Bound Method {name} of {self.im_self}>"
        else:
            return f"<Unbound Method {name}>"

    def __call__(self, *args, **kwargs):
        if self.im_self is not None:
            return self.im_func(self.im_self, *args, **kwargs)
        else:
            return self.im_func(*args, **kwargs)


class Cell(object):
    """A fake cell for closures.

    Closures keep names in scope by storing them not in a frame, but in a
    separate object called a cell.  Frames share references to cells, and
    the LOAD_DEREF and STORE_DEREF opcodes get and set the value from cells.

    This class acts as a cell, though it has to jump through two hoops to make
    the simulation complete:

        1. In order to create actual FunctionType functions, we have to have
           actual cell objects, which are difficult to make. See the twisty
           double-lambda in __init__.

        2. Actual cell objects can't be modified, so to implement STORE_DEREF,
           we store a one-element list in our cell, and then use [0] as the
           actual value.

    """

    def __init__(self, value):
        self.contents = value

    def get(self):
        return self.contents

    def set(self, value):
        self.contents = value


Block = collections.namedtuple("Block", "type, handler, level")


class Frame(object):
    def __init__(self, f_code, f_globals, f_locals, f_closure, f_back):
        self.f_code = f_code
        self.py36_opcodes = list(dis.get_instructions(self.f_code))
        self.f_globals = f_globals
        self.f_locals = f_locals
        self.f_back = f_back
        self.stack = []
        # if f_back:
        #     self.f_builtins = f_back.f_builtins
        # else:
        #     self.f_builtins = f_locals["__builtins__"]
        #     if hasattr(self.f_builtins, "__dict__"):
        #         self.f_builtins = self.f_builtins.__dict__
        if f_back and f_back.f_globals is f_globals:
            # If we share the globals, we share the builtins.
            self.f_builtins = f_back.f_builtins
        else:
            try:
                self.f_builtins = f_globals["__builtins__"]
                if hasattr(self.f_builtins, "__dict__"):
                    self.f_builtins = self.f_builtins.__dict__
            except KeyError:
                # No builtins! Make up a minimal one with None.
                self.f_builtins = {"None": None}

        self.f_lineno = f_code.co_firstlineno
        self.f_lasti = 0

        self.cells = {} if f_code.co_cellvars or f_code.co_freevars else None
        for var in f_code.co_cellvars:
            # Make a cell for the variable in our locals, or None.
            self.cells[var] = Cell(self.f_locals.get(var))
        if f_code.co_freevars:
            assert len(f_code.co_freevars) == len(f_closure)
            self.cells.update(zip(f_code.co_freevars, f_closure))

        self.block_stack = []
        self.generator = None

    def __repr__(self):  # pragma: no cover
        return f"<Frame at 0x{id(self)}: {self.f_code.co_filename!r} @ {self.f_lineno:d}>"

    def line_number(self):
        """Get the current line number the frame is executing."""
        # We don't keep f_lineno up to date, so calculate it based on the
        # instruction address and the line number table.
        lnotab = self.f_code.co_lnotab
        byte_increments = (lnotab[0::2]).iterbytes()
        line_increments = (lnotab[1::2]).iterbytes()

        byte_num = 0
        line_num = self.f_code.co_firstlineno

        for byte_incr, line_incr in zip(byte_increments, line_increments):
            byte_num += byte_incr
            if byte_num > self.f_lasti:
                break
            line_num += line_incr

        return line_num


class Generator(object):
    def __init__(self, g_frame, vm):
        self.gi_frame = g_frame
        self.vm = vm
        self.started = False
        self.finished = False
        self.gi_code = g_frame.f_code
        self.__name__ = g_frame.f_code.co_name

    def __iter__(self):
        return self

    def next(self):
        return self.send(None)

    def send(self, value=None):
        if not self.started and value is not None:
            raise TypeError("Can't send non-None value to a just-started generator")
        self.gi_frame.stack.append(value)
        self.started = True
        self.running = True
        val = self.vm.resume_frame(self.gi_frame)
        if self.finished:
            self.running = False
            raise StopIteration(val)
        return val

    __next__ = next

class Traceback(object):
    def __init__(self, frame):
        self.tb_next = frame.f_back
        self.tb_lasti = frame.f_lasti
        self.tb_lineno = frame.f_lineno
        self.tb_frame = frame

    # Note: this can be removed when we have our own compatibility traceback.
    def print_tb(self, limit=None, file=stderr):
        """Like traceback.tb, but is a method."""
        tb = self
        while tb:
            f = tb.tb_frame
            filename = f.f_code.co_filename
            lineno = f.line_number()
            print(
                '  File "%s", line %d, in %s' % (filename, lineno, f.f_code.co_name),
                file=file,
            )
            linecache.checkcache(filename)
            line = linecache.getline(filename, lineno, f.f_globals)
            if line:
                print("    " + line.strip(), file=file)
            tb = tb.tb_next


def traceback_from_frame(frame):
    tb = None

    while frame:
        next_tb = Traceback(copy(frame))
        next_tb.tb_next = tb
        tb = next_tb
        frame = frame.f_back
    return tb
