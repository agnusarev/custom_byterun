"""A pure-Python Python bytecode interpreter."""

# Based on:
# pyvm2 by Paul Swartz (z3p), from http://www.twistedmatrix.com/users/z3p/

from __future__ import division, print_function

import dis
import linecache
import logging
import operator
import reprlib
import sys
import types

from .pyobj import Block, Cell, Frame, Function, Generator, traceback_from_frame

log = logging.getLogger(__name__)


# Create a repr that won't overflow.
repr_obj = reprlib.Repr()
repr_obj.maxother = 120
repper = repr_obj.repr


class VirtualMachineError(Exception):
    """For raising errors in the operation of the VM."""
    pass


class VirtualMachine(object):
    def __init__(self):
        # The call stack of frames.
        self.frames = []
        # The current frame.
        self.frame = None
        self.return_value = None
        self.last_exception = None

    def top(self):
        """Return the value at the top of the stack, with no changes."""
        return self.frame.stack[-1]

    def pop(self, i=0):
        """Pop a value from the stack.

        Default to the top of the stack, but `i` can be a count from the top
        instead.

        """
        return self.frame.stack.pop(-1 - i)

    def push(self, *vals):
        """Push values onto the value stack."""
        self.frame.stack.extend(vals)

    def popn(self, n):
        """Pop a number of values from the value stack.

        A list of `n` values is returned, the deepest value first.

        """
        if n:
            ret = self.frame.stack[-n:]
            self.frame.stack[-n:] = []
            return ret
        else:
            return []

    def peek(self, n):
        """Get a value `n` entries down in the stack, without changing the stack."""
        return self.frame.stack[-n]

    def jump(self, jump):
        """Move the bytecode pointer to `jump`, so it will execute next."""
        self.frame.f_lasti = jump

    def push_block(self, type, handler=None, level=None):
        if level is None:
            level = len(self.frame.stack)
        self.frame.block_stack.append(Block(type, handler, level))

    def pop_block(self):
        return self.frame.block_stack.pop()

    def make_frame(
        self, code, callargs={}, f_globals=None, f_locals=None, f_closure=None
    ):
        log.info("make_frame: code=%r, callargs=%s" % (code, repper(callargs)))
        if f_globals is not None:
            f_globals = f_globals
            if f_locals is None:
                f_locals = f_globals
        elif self.frames:
            f_globals = self.frame.f_globals
            f_locals = {}
        else:
            f_globals = f_locals = {
                "__builtins__": __builtins__,
                "__name__": "__main__",
                "__doc__": None,
                "__package__": None,
            }
        f_locals.update(callargs)
        frame = Frame(code, f_globals, f_locals, f_closure, self.frame)
        return frame

    def push_frame(self, frame):
        self.frames.append(frame)
        self.frame = frame

    def pop_frame(self):
        self.frames.pop()
        if self.frames:
            self.frame = self.frames[-1]
        else:
            self.frame = None

    def print_frames(self):
        """Print the call stack, for debugging."""
        for f in self.frames:
            filename = f.f_code.co_filename
            lineno = f.line_number()
            print(f'  File "{filename}", line {lineno:d}, in {f.f_code.co_name}')
            linecache.checkcache(filename)
            line = linecache.getline(filename, lineno, f.f_globals)
            if line:
                print("    " + line.strip())

    def resume_frame(self, frame):
        frame.f_back = self.frame
        val = self.run_frame(frame)
        frame.f_back = None
        return val

    def run_code(self, code, f_globals=None, f_locals=None):
        frame = self.make_frame(code, f_globals=f_globals, f_locals=f_locals)
        val = self.run_frame(frame)
        # Check some invariants
        if self.frames:  # pragma: no cover
            raise VirtualMachineError("Frames left over!")
        if self.frame and self.frame.stack:  # pragma: no cover
            raise VirtualMachineError("Data left on stack! %r" % self.frame.stack)

        return val

    def unwind_block(self, block):
        if block.type == "except-handler":
            offset = 3
        else:
            offset = 0

        while len(self.frame.stack) > block.level + offset:
            self.pop()

        if block.type == "except-handler":
            tb, value, exctype = self.popn(3)
            self.last_exception = exctype, value, tb

    def parse_byte_and_args(self):
        """Parse 1 - 3 bytes of bytecode into
        an instruction and optionally arguments.
        In Python3.6 the format is 2 bytes per instruction."""
        f = self.frame
        opoffset = f.f_lasti
        if f.py36_opcodes:
            currentOp = f.py36_opcodes[opoffset]
            byteCode = currentOp.opcode
            byteName = currentOp.opname
        else:
            byteCode = f.f_code.co_code[opoffset]
            byteName = dis.opname[byteCode]

        f.f_lasti += 1
        arg = None
        arguments = []
        if f.py36_opcodes and byteCode == dis.EXTENDED_ARG:
            # Prefixes any opcode which has an argument too big to fit into the
            # default two bytes. ext holds two additional bytes which, taken
            # together with the subsequent opcode’s argument, comprise a
            # four-byte argument, ext being the two most-significant bytes.
            # We simply ignore the EXTENDED_ARG because that calculation
            # is already done by dis, and stored in next currentOp.
            # Lib/dis.py:_unpack_opargs
            return self.parse_byte_and_args()
        if byteCode < dis.HAVE_ARGUMENT:
            return byteName, arguments, opoffset

        if f.py36_opcodes:
            intArg = currentOp.arg
        else:
            arg = f.f_code.co_code[f.f_lasti: f.f_lasti + 2]
            f.f_lasti += 2
            intArg = (arg[0]) + (arg[1] << 8)

        if byteCode in dis.hasconst:
            arg = f.f_code.co_consts[intArg]
        elif byteCode in dis.hasfree:
            if intArg < len(f.f_code.co_cellvars):
                arg = f.f_code.co_cellvars[intArg]
            else:
                var_idx = intArg - len(f.f_code.co_cellvars)
                arg = f.f_code.co_freevars[var_idx]
        elif byteCode in dis.hasname:
            arg = f.f_code.co_names[intArg]
        elif byteCode in dis.hasjrel:
            intArg += intArg
            if f.py36_opcodes:
                arg = f.f_lasti + intArg // 2
            else:
                arg = f.f_lasti + intArg
        elif byteCode in dis.hasjabs:
            intArg += intArg
            if f.py36_opcodes:
                arg = intArg // 2
            else:
                arg = intArg
        elif byteCode in dis.haslocal:
            arg = f.f_code.co_varnames[intArg]
        else:
            arg = intArg
        arguments = [arg]

        return byteName, arguments, opoffset

    def log(self, byteName, arguments, opoffset):
        """Log arguments, block stack, and data stack for each opcode."""
        op = f"{opoffset:d}: {byteName}"
        if arguments:
            op += f" {arguments[0]!r}"
        indent = "    " * (len(self.frames) - 1)
        stack_rep = repper(self.frame.stack)
        block_stack_rep = repper(self.frame.block_stack)

        log.info(f"  {indent}data: {stack_rep}")
        log.info(f"  {indent}blks: {block_stack_rep}")
        log.info(f"{indent}{op}")

    def dispatch(self, byteName, arguments):
        """Dispatch by bytename to the corresponding methods.
        Exceptions are caught and set on the virtual machine."""
        why = None
        try:
            if byteName.startswith("UNARY_"):
                self.unaryOperator(byteName[6:])
            elif byteName.startswith("BINARY_"):
                self.binaryOperator(byteName[7:])
            elif byteName.startswith("INPLACE_"):
                self.inplaceOperator(byteName[8:])
            elif "SLICE+" in byteName:
                self.sliceOperator(byteName)
            else:
                # dispatch
                bytecode_fn = getattr(self, f"byte_{byteName}", None)
                if not bytecode_fn:  # pragma: no cover
                    raise VirtualMachineError(f"unknown bytecode type: {byteName}")
                why = bytecode_fn(*arguments)

        except Exception:
            # deal with exceptions encountered while executing the op.
            self.last_exception = sys.exc_info()[:2] + (None,)
            log.exception("Caught exception during execution")
            why = "exception"

        return why

    def manage_block_stack(self, why):
        """Manage a frame's block stack.
        Manipulate the block stack and data stack for looping,
        exception handling, or returning."""
        assert why != "yield"

        block = self.frame.block_stack[-1]
        if block.type == "loop" and why == "continue":
            self.jump(self.return_value)
            why = None
            return why

        self.pop_block()
        self.unwind_block(block)

        if block.type == "loop" and why == "break":
            why = None
            self.jump(block.handler)
            return why

        if why == "exception" and block.type in ["setup-except", "finally"]:
            self.push_block("except-handler")
            exctype, value, tb = self.last_exception
            self.push(tb, value, exctype)
            # PyErr_Normalize_Exception goes here
            self.push(tb, value, exctype)
            why = None
            self.jump(block.handler)
            return why

        elif block.type == "finally":
            if why in ("return", "continue"):
                self.push(self.return_value)
            self.push(why)

            why = None
            self.jump(block.handler)
            return why

        return why

    def run_frame(self, frame):
        """Run a frame until it returns (somehow).

        Exceptions are raised, the return value is returned.

        """
        self.push_frame(frame)
        while True:
            byteName, arguments, opoffset = self.parse_byte_and_args()
            if log.isEnabledFor(logging.INFO):
                self.log(byteName, arguments, opoffset)

            # When unwinding the block stack, we need to keep track of why we
            # are doing it.
            # print(byteName, arguments, opoffset)
            why = self.dispatch(byteName, arguments)
            if why == "exception":
                # TODO: ceval calls PyTraceBack_Here, not sure what that does.
                pass

            if why == "reraise":
                why = "exception"

            if why != "yield":
                while why and frame.block_stack:
                    # Deal with any block management we need to do.
                    why = self.manage_block_stack(why)

            if why:
                break

        # TODO: handle generator exception state

        self.pop_frame()

        if why == "exception":
            if self.last_exception and self.last_exception[0]:
                raise self.last_exception[1]
            else:
                raise VirtualMachineError("Borked exception recording")

        return self.return_value

    # Stack manipulation

    def byte_LOAD_CONST(self, const):
        self.push(const)

    def byte_POP_TOP(self):
        self.pop()

    def byte_NOP(self):
        """Do nothing code. Used as a placeholder by the bytecode optimizer."""
        pass

    def byte_DUP_TOP(self):
        self.push(self.top())

    def byte_DUP_TOPX(self, count):
        items = self.popn(count)
        for i in [1, 2]:
            self.push(*items)

    def byte_DUP_TOP_TWO(self):
        a, b = self.popn(2)
        self.push(a, b, a, b)

    def byte_ROT_TWO(self):
        a, b = self.popn(2)
        self.push(b, a)

    def byte_ROT_THREE(self):
        a, b, c = self.popn(3)
        self.push(c, a, b)

    def byte_ROT_FOUR(self):
        a, b, c, d = self.popn(4)
        self.push(d, a, b, c)

    # Names

    def byte_LOAD_NAME(self, name):
        frame = self.frame
        if name in frame.f_locals:
            val = frame.f_locals[name]
        elif name in frame.f_globals:
            val = frame.f_globals[name]
        elif name in frame.f_builtins:
            val = frame.f_builtins[name]
        else:
            raise NameError(f"name '{name}' is not defined")
        self.push(val)

    def byte_STORE_NAME(self, name):
        self.frame.f_locals[name] = self.pop()

    def byte_DELETE_NAME(self, name):
        del self.frame.f_locals[name]

    def byte_LOAD_FAST(self, name):
        if name in self.frame.f_locals:
            val = self.frame.f_locals[name]
        else:
            raise UnboundLocalError(
                f"local variable '{name}' referenced before assignment"
            )
        self.push(val)

    def byte_STORE_FAST(self, name):
        self.frame.f_locals[name] = self.pop()

    def byte_DELETE_FAST(self, name):
        del self.frame.f_locals[name]

    def byte_LOAD_GLOBAL(self, name):
        f = self.frame
        if name in f.f_globals:
            val = f.f_globals[name]
        elif name in f.f_builtins:
            val = f.f_builtins[name]
        else:
            raise NameError(f"name '{name}' is not defined")
        self.push(val)

    def byte_STORE_GLOBAL(self, name):
        f = self.frame
        f.f_globals[name] = self.pop()

    def byte_LOAD_DEREF(self, name):
        self.push(self.frame.cells[name].get())

    def byte_STORE_DEREF(self, name):
        self.frame.cells[name].set(self.pop())

    def byte_LOAD_LOCALS(self):
        self.push(self.frame.f_locals)

    # Operators

    UNARY_OPERATORS = {
        "POSITIVE": operator.pos,
        "NEGATIVE": operator.neg,
        "NOT": operator.not_,
        "CONVERT": repr,
        "INVERT": operator.invert,
    }

    def unaryOperator(self, op):
        x = self.pop()
        self.push(self.UNARY_OPERATORS[op](x))

    BINARY_OPERATORS = {
        "POWER": pow,
        "MULTIPLY": operator.mul,
        "DIVIDE": getattr(operator, "div", lambda x, y: None),
        "FLOOR_DIVIDE": operator.floordiv,
        "TRUE_DIVIDE": operator.truediv,
        "MODULO": operator.mod,
        "ADD": operator.add,
        "SUBTRACT": operator.sub,
        "SUBSCR": operator.getitem,
        "LSHIFT": operator.lshift,
        "RSHIFT": operator.rshift,
        "AND": operator.and_,
        "XOR": operator.xor,
        "OR": operator.or_,
    }

    def binaryOperator(self, op):
        x, y = self.popn(2)
        self.push(self.BINARY_OPERATORS[op](x, y))

    def inplaceOperator(self, op):
        x, y = self.popn(2)
        if op == "POWER":
            x **= y
        elif op == "MULTIPLY":
            x *= y
        elif op in ["DIVIDE", "FLOOR_DIVIDE"]:
            x //= y
        elif op == "TRUE_DIVIDE":
            x /= y
        elif op == "MODULO":
            x %= y
        elif op == "ADD":
            x += y
        elif op == "SUBTRACT":
            x -= y
        elif op == "LSHIFT":
            x <<= y
        elif op == "RSHIFT":
            x >>= y
        elif op == "AND":
            x &= y
        elif op == "XOR":
            x ^= y
        elif op == "OR":
            x |= y
        else:  # pragma: no cover
            raise VirtualMachineError(f"Unknown in-place operator: {op!r}")
        self.push(x)

    def sliceOperator(self, op):
        start = 0
        end = None  # we will take this to mean end
        op, count = op[:-2], int(op[-1])
        if count == 1:
            start = self.pop()
        elif count == 2:
            end = self.pop()
        elif count == 3:
            end = self.pop()
            start = self.pop()
        el = self.pop()
        if end is None:
            end = len(el)
        if op.startswith("STORE_"):
            el[start:end] = self.pop()
        elif op.startswith("DELETE_"):
            del el[start:end]
        else:
            self.push(el[start:end])

    COMPARE_OPERATORS = [
        operator.lt,
        operator.le,
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
        lambda x, y: x in y,
        lambda x, y: x not in y,
        lambda x, y: x is y,
        lambda x, y: x is not y,
        lambda x, y: issubclass(x, Exception) and issubclass(x, y),
    ]

    def byte_COMPARE_OP(self, opnum):
        x, y = self.popn(2)
        self.push(self.COMPARE_OPERATORS[opnum](x, y))

    # Attributes and indexing

    def byte_LOAD_ATTR(self, attr):
        obj = self.pop()
        val = getattr(obj, attr)
        self.push(val)

    def byte_STORE_ATTR(self, name):
        val, obj = self.popn(2)
        setattr(obj, name, val)

    def byte_DELETE_ATTR(self, name):
        obj = self.pop()
        delattr(obj, name)

    def byte_STORE_SUBSCR(self):
        val, obj, subscr = self.popn(3)
        obj[subscr] = val

    def byte_DELETE_SUBSCR(self):
        obj, subscr = self.popn(2)
        del obj[subscr]

    # Building

    def byte_BUILD_TUPLE_UNPACK_WITH_CALL(self, count):
        # This is similar to BUILD_TUPLE_UNPACK, but is used for f(*x, *y, *z)
        # call syntax. The stack item at position count + 1 should be the
        # corresponding callable f.
        elts = self.popn(count)
        self.push(tuple(e for el in elts for e in el))

    def byte_BUILD_TUPLE_UNPACK(self, count):
        # Pops count iterables from the stack, joins them in a single tuple,
        # and pushes the result. Implements iterable unpacking in
        # tuple displays (*x, *y, *z).
        elts = self.popn(count)
        self.push(tuple(e for el in elts for e in el))

    def byte_BUILD_TUPLE(self, count):
        elts = self.popn(count)
        self.push(tuple(elts))

    def byte_BUILD_LIST(self, count):
        elts = self.popn(count)
        self.push(elts)

    def byte_BUILD_SET(self, count):
        elts = self.popn(count)
        self.push(set(elts))

    def byte_BUILD_CONST_KEY_MAP(self, count):
        # count values are consumed from the stack.
        # The top element contains tuple of keys
        # added in version 3.6
        keys = self.pop()
        values = self.popn(count)
        kvs = dict(zip(keys, values))
        self.push(kvs)

    def byte_BUILD_MAP(self, count):
        self.push({})
        return

    def byte_STORE_MAP(self):
        the_map, val, key = self.popn(3)
        the_map[key] = val
        self.push(the_map)

    def byte_UNPACK_SEQUENCE(self, count):
        seq = self.pop()
        for x in reversed(seq):
            self.push(x)

    def byte_BUILD_SLICE(self, count):
        if count == 2:
            x, y = self.popn(2)
            self.push(slice(x, y))
        elif count == 3:
            x, y, z = self.popn(3)
            self.push(slice(x, y, z))
        else:  # pragma: no cover
            raise VirtualMachineError("Strange BUILD_SLICE count: %r" % count)

    def byte_LIST_APPEND(self, count):
        val = self.pop()
        the_list = self.peek(count)
        the_list.append(val)

    def byte_SET_ADD(self, count):
        val = self.pop()
        the_set = self.peek(count)
        the_set.add(val)

    def byte_MAP_ADD(self, count):
        key, val = self.popn(2)
        the_map = self.peek(count)
        the_map[key] = val

    # Printing

    def byte_PRINT_EXPR(self):
        print(self.pop())

    def byte_PRINT_ITEM(self):
        item = self.pop()
        self.print_item(item)

    def byte_PRINT_ITEM_TO(self):
        to = self.pop()
        item = self.pop()
        self.print_item(item, to)

    def byte_PRINT_NEWLINE(self):
        self.print_newline()

    def byte_PRINT_NEWLINE_TO(self):
        to = self.pop()
        self.print_newline(to)

    def print_item(self, item, to=None):
        if to is None:
            to = sys.stdout
        if to.softspace:
            print(" ", end="", file=to)
            to.softspace = 0
        print(item, end="", file=to)
        if isinstance(item, str):
            if (not item) or (not item[-1].isspace()) or (item[-1] == " "):
                to.softspace = 1
        else:
            to.softspace = 1

    def print_newline(self, to=None):
        if to is None:
            to = sys.stdout
        print("", file=to)
        to.softspace = 0

    # Jumps

    def byte_JUMP_FORWARD(self, jump):
        self.jump(jump)

    def byte_JUMP_ABSOLUTE(self, jump):
        self.jump(jump)

    def byte_JUMP_IF_TRUE(self, jump):
        val = self.top()
        if val:
            self.jump(jump)

    def byte_JUMP_IF_FALSE(self, jump):
        val = self.top()
        if not val:
            self.jump(jump)

    def byte_POP_JUMP_IF_TRUE(self, jump):
        val = self.pop()
        if val:
            self.jump(jump)

    def byte_POP_JUMP_IF_FALSE(self, jump):
        val = self.pop()
        if not val:
            self.jump(jump)

    def byte_JUMP_IF_TRUE_OR_POP(self, jump):
        val = self.top()
        if val:
            self.jump(jump)
        else:
            self.pop()

    def byte_JUMP_IF_FALSE_OR_POP(self, jump):
        val = self.top()
        if not val:
            self.jump(jump)
        else:
            self.pop()

    # Blocks

    def byte_SETUP_LOOP(self, dest):
        self.push_block("loop", dest)

    def byte_GET_ITER(self):
        self.push(iter(self.pop()))

    def byte_GET_YIELD_FROM_ITER(self):
        tos = self.top()
        if isinstance(tos, types.GeneratorType) or isinstance(tos, types.CoroutineType):
            return
        tos = self.pop()
        self.push(iter(tos))

    def byte_FOR_ITER(self, jump):
        iterobj = self.top()
        try:
            v = next(iterobj)
            self.push(v)
        except StopIteration:
            self.pop()
            self.jump(jump)

    def byte_BREAK_LOOP(self):
        return "break"

    def byte_CONTINUE_LOOP(self, dest):
        self.return_value = dest
        return "continue"

    def byte_SETUP_EXCEPT(self, dest):
        self.push_block("setup-except", dest)

    def byte_SETUP_FINALLY(self, dest):
        self.push_block("finally", dest)

    def byte_END_FINALLY(self):
        v = self.pop()
        if isinstance(v, str):
            why = v
            if why in ("return", "continue"):
                self.return_value = self.pop()
            if why == "silenced":
                block = self.pop_block()
                assert block.type == "except-handler"
                self.unwind_block(block)
                why = None
        elif v is None:
            why = None
        elif issubclass(v, BaseException):
            exctype = v
            val = self.pop()
            tb = self.pop()
            self.last_exception = (exctype, val, tb)
            why = "reraise"
        else:  # pragma: no cover
            raise VirtualMachineError("Confused END_FINALLY")
        return why

    def byte_POP_BLOCK(self):
        self.pop_block()

    def byte_RAISE_VARARGS(self, argc):
        cause = exc = None
        if argc == 2:
            cause = self.pop()
            exc = self.pop()
        elif argc == 1:
            exc = self.pop()
        return self.do_raise(exc, cause)

    def do_raise(self, exc, cause):
        if exc is None:  # reraise
            exc_type, val, tb = self.last_exception
            if exc_type is None:
                return "exception"  # error
            else:
                return "reraise"

        elif type(exc) is type:
            # As in `raise ValueError`
            exc_type = exc
            val = exc()  # Make an instance.
        elif isinstance(exc, BaseException):
            # As in `raise ValueError('foo')`
            exc_type = type(exc)
            val = exc
        else:
            return "exception"  # error

        # If you reach this point, you're guaranteed that
        # val is a valid exception instance and exc_type is its class.
        # Now do a similar thing for the cause, if present.
        if cause:
            if type(cause) is type:
                cause = cause()
            elif not isinstance(cause, BaseException):
                return "exception"  # error

            val.__cause__ = cause

        self.last_exception = exc_type, val, val.__traceback__
        return "exception"

    def byte_POP_EXCEPT(self):
        block = self.pop_block()
        if block.type != "except-handler":
            raise Exception("popped block is not an except handler")
        self.unwind_block(block)

    def byte_SETUP_WITH(self, dest):
        ctxmgr = self.pop()
        self.push(ctxmgr.__exit__)
        ctxmgr_obj = ctxmgr.__enter__()
        self.push_block("finally", dest)
        self.push(ctxmgr_obj)

    def byte_WITH_CLEANUP_START(self):
        u = self.top()
        v = None
        w = None
        if u is None:
            exit_method = self.pop(1)
        elif isinstance(u, str):
            if u in {"return", "continue"}:
                exit_method = self.pop(2)
            else:
                exit_method = self.pop(1)
        elif issubclass(u, BaseException):
            w, v, u = self.popn(3)
            tp, exc, tb = self.popn(3)
            exit_method = self.pop()
            self.push(tp, exc, tb)
            self.push(None)
            self.push(w, v, u)
            block = self.pop_block()
            assert block.type == "except-handler"
            self.push_block(block.type, block.handler, block.level - 1)

        res = exit_method(u, v, w)
        self.push(u)
        self.push(res)

    def byte_WITH_CLEANUP_FINISH(self):
        res = self.pop()
        u = self.pop()
        if type(u) is type and issubclass(u, BaseException) and res:
            self.push("silenced")

    def byte_WITH_CLEANUP(self):
        # The code here does some weird stack manipulation: the exit function
        # is buried in the stack, and where depends on what's on top of it.
        # Pull out the exit function, and leave the rest in place.
        v = w = None
        u = self.top()
        if u is None:
            exit_func = self.pop(1)
        elif isinstance(u, str):
            if u in ("return", "continue"):
                exit_func = self.pop(2)
            else:
                exit_func = self.pop(1)
            u = None
        elif issubclass(u, BaseException):
            w, v, u = self.popn(3)
            tp, exc, tb = self.popn(3)
            exit_func = self.pop()
            self.push(tp, exc, tb)
            self.push(None)
            self.push(w, v, u)
            block = self.pop_block()
            assert block.type == "except-handler"
            self.push_block(block.type, block.handler, block.level - 1)
        else:  # pragma: no cover
            raise VirtualMachineError("Confused WITH_CLEANUP")
        exit_ret = exit_func(u, v, w)
        err = (u is not None) and bool(exit_ret)
        if err:
            self.push("silenced")

    # Functions

    def byte_MAKE_FUNCTION(self, argc):
        name = self.pop()
        code = self.pop()
        globs = self.frame.f_globals
        closure = self.pop() if (argc & 0x8) else None
        # ann = self.pop() if (argc & 0x4) else None
        kwdefaults = self.pop() if (argc & 0x2) else None
        defaults = self.pop() if (argc & 0x1) else None
        fn = Function(name, code, globs, defaults, kwdefaults, closure, self)
        self.push(fn)

    def byte_LOAD_CLOSURE(self, name):
        self.push(self.frame.cells[name])

    def byte_MAKE_CLOSURE(self, argc):
        name = self.pop()
        closure, code = self.popn(2)
        defaults = self.popn(argc)
        globs = self.frame.f_globals
        fn = Function(name, code, globs, defaults, None, closure, self)
        self.push(fn)

    def byte_CALL_FUNCTION_EX(self, arg):
        varkw = self.pop() if (arg & 0x1) else {}
        varpos = self.pop()
        return self.call_function(0, varpos, varkw)

    def byte_CALL_FUNCTION(self, arg):
        try:
            return self.call_function(arg, [], {})
        except TypeError as exc:
            tb = self.last_traceback = traceback_from_frame(self.frame)
            self.last_exception = (TypeError, exc, tb)
            return "exception"

    def byte_CALL_FUNCTION_VAR(self, arg):
        args = self.pop()
        return self.call_function(arg, args, {})

    def byte_CALL_FUNCTION_KW(self, argc):
        kwargnames = self.pop()
        lkwargs = len(kwargnames)
        kwargs = self.popn(lkwargs)
        arg = argc - lkwargs
        return self.call_function(arg, [], dict(zip(kwargnames, kwargs)))

    def byte_CALL_FUNCTION_VAR_KW(self, arg):
        args, kwargs = self.popn(2)
        return self.call_function(arg, args, kwargs)

    def call_function(self, arg, args, kwargs):
        lenKw, lenPos = divmod(arg, 256)
        namedargs = {}
        for i in range(lenKw):
            key, val = self.popn(2)
            namedargs[key] = val
        namedargs.update(kwargs)
        posargs = self.popn(lenPos)
        posargs.extend(args)

        func = self.pop()
        # frame = self.frame
        if hasattr(func, "im_func"):
            # Methods get self as an implicit first parameter.
            if func.im_self is not None:
                posargs.insert(0, func.im_self)
            # The first parameter must be the correct type.
            if not isinstance(posargs[0], type(func.im_class)):
                raise TypeError(
                    f"unbound method {func.im_func.func_name}() must be called with {func.im_class.__name__} instance "
                    f"as first argument (got {type(posargs[0]).__name__} instance instead)"
                )
            func = func.im_func
        # print(func)
        # print(posargs)
        # print(namedargs)
        # import inspect
        # if inspect.isclass(func):
        #     print("jdkjfghdkjfgh")
        retval = func(*posargs, **namedargs)
        self.push(retval)

    def byte_RETURN_VALUE(self):
        self.return_value = self.pop()
        if self.frame.generator:
            self.frame.generator.finished = True
        return "return"

    def byte_YIELD_VALUE(self):
        self.return_value = self.pop()
        return "yield"

    def byte_LOAD_METHOD(self, name):
        tos = self.pop()
        if hasattr(tos, name):
            self.push(getattr(tos, name))
        else:
            self.push(None)

    def byte_CALL_METHOD(self, count):
        posargs = self.popn(count)
        self.call_function(0, posargs, {})

    def byte_CALL_FINALLY(self, delta):
        self.push(self.frame.f_lasti)
        self.jump(delta)

    def byte_BEGIN_FINALLY(self):
        self.push(None)

    def byte_POP_FINALLY(self, preserve_tos: int):
        v = self.pop()
        if v is None:
            why = None
        elif issubclass(v, BaseException):
            # from trepan.api import debug; debug()
            exctype = v
            val = self.pop()
            tb = self.pop()
            self.last_exception = (exctype, val, tb)

            # FIXME: pop 3 more values
            why = "reraise"
            raise VirtualMachineError("POP_FINALLY not finished yet")
        else:  # pragma: no cover
            raise VirtualMachineError("Confused POP_FINALLY")
        return why

    def byte_YIELD_FROM(self):
        u = self.pop()
        x = self.top()

        try:
            if not isinstance(x, Generator) or u is None:
                # Call next on iterators.
                retval = next(x)
            else:
                retval = x.send(u)
            self.return_value = retval
        except StopIteration as e:
            self.pop()
            self.push(e.value)
        else:
            self.jump(self.frame.f_lasti - 1)
            return "yield"

    # Importing

    def byte_IMPORT_NAME(self, name):
        level, fromlist = self.popn(2)
        frame = self.frame
        self.push(__import__(name, frame.f_globals, frame.f_locals, fromlist, level))

    def byte_IMPORT_STAR(self):
        # TODO: this doesn't use __all__ properly.
        mod = self.pop()
        for attr in dir(mod):
            if attr[0] != "_":
                self.frame.f_locals[attr] = getattr(mod, attr)

    def byte_IMPORT_FROM(self, name):
        mod = self.top()
        self.push(getattr(mod, name))

    def byte_RERAISE(self):
        # FIXME
        raise RuntimeError("RERAISE not implemented yet")
        pass

    def byte_WITH_EXCEPT_START(self):
        # FIXME
        raise RuntimeError("WITH_EXCEPT_START not implemented yet")
        pass

    def byte_LOAD_ASSERTION_ERROR(self):
        """
        Pushes AssertionError onto the stack. Used by the `assert` statement.
        """
        self.push(AssertionError)

    def byte_LIST_TO_TUPLE(self):
        """
        Pops a list from the stack and pushes a tuple containing the same values.
        """
        self.push(tuple(self.pop()))

    def byte_IS_OP(self, invert: int):
        """Performs is comparison, or is not if invert is 1."""
        TOS1, TOS = self.popn(2)
        if invert:
            self.push(TOS1 is not TOS)
        else:
            self.push(TOS1 is TOS)
        pass

    def byte_JUMP_IF_NOT_EXC_MATCH(self, target: int):
        """Tests whether the second value on the stack is an exception
        matching TOS, and jumps if it is not.  Pops two values from
        the stack.
        """
        TOS1, TOS = self.popn(2)
        # FIXME: not sure what operation should be used to test not "matches".
        if not issubclass(TOS1, TOS):
            self.jump(target)
        return

    def byte_GEN_START(self, kind):
        """Pops TOS. If TOS was not None, raises an exception. The kind
        operand corresponds to the type of generator or coroutine and
        determines the error message. The legal kinds are 0 for
        generator, 1 for coroutine, and 2 for async generator.
        """
        self.pop()
        assert kind in (0, 1, None)

    def byte_CONTAINS_OP(self, invert: int):
        """Performs in comparison, or not in if invert is 1."""
        TOS1, TOS = self.popn(2)
        if invert:
            self.push(TOS1 not in TOS)
        else:
            self.push(TOS1 in TOS)
        return

    def byte_LIST_EXTEND(self, i):
        """Calls list.extend(TOS1[-i], TOS). Used to build lists."""
        TOS = self.pop()
        destination = self.peek(i)
        assert isinstance(destination, list)
        destination.extend(TOS)

    def byte_SET_UPDATE(self, i):
        """Calls set.update(TOS1[-i], TOS). Used to build sets."""
        TOS = self.pop()
        destination = self.peek(i)
        assert isinstance(destination, set)
        destination.update(TOS)

    def byte_DICT_MERGE(self, i):
        """Like DICT_UPDATE but raises an exception for duplicate keys."""
        TOS = self.pop()
        assert isinstance(TOS, dict)
        destination = self.peek(i)
        assert isinstance(destination, dict)
        dups = set(destination.keys()) & set(TOS.keys())
        if bool(dups):
            raise RuntimeError(f"Duplicate keys '{dups}' in DICT_MERGE")
        destination.update(TOS)

    def byte_DICT_UPDATE(self, i):
        """Calls dict.update(TOS1[-i], TOS). Used to build dicts."""
        TOS = self.pop()
        assert isinstance(TOS, dict)
        destination = self.peek(i)
        assert isinstance(destination, dict)
        destination.update(TOS)

    def byte_EXEC_STMT(self):
        stmt, globs, locs = self.popn(3)
        exec(stmt, globs, locs)

    def byte_LOAD_BUILD_CLASS(self):
        self.push(self.build_class)

    def byte_STORE_LOCALS(self):
        self.frame.f_locals = self.pop()

    def byte_SET_LINENO(self, lineno):
        self.frame.f_lineno = lineno

    def build_class(self, func, name, *bases, **kwds):
        "Like __build_class__ in bltinmodule.c, but running in the byterun VM."
        if not isinstance(func, Function):
            raise TypeError("func must be a function")
        if not isinstance(name, str):
            raise TypeError("name is not a string")
        metaclass = kwds.pop("metaclass", None)
        if metaclass is None:
            metaclass = type(bases[0]) if bases else type
        if isinstance(metaclass, type):
            metaclass = self.calculate_metaclass(metaclass, bases)

        try:
            prepare = metaclass.__prepare__
        except AttributeError:
            namespace = {}
        else:
            namespace = prepare(name, bases, **kwds)

        # Execute the body of func. This is the step that would go wrong if
        # we tried to use the built-in __build_class__, because __build_class__
        # does not call func, it magically executes its body directly, as we
        # do here (except we invoke our VirtualMachine instead of CPython's).
        frame = func._vm.make_frame(
            func.func_code,
            f_globals=func.func_globals,
            f_locals=namespace,
            f_closure=func.func_closure,
        )
        cell = func._vm.run_frame(frame)

        cls = metaclass(name, bases, namespace)
        if isinstance(cell, Cell):
            cell.set(cls)
        return cls

    def calculate_metaclass(self, metaclass, bases):
        "Determine the most derived metatype."
        winner = metaclass
        for base in bases:
            t = type(base)
            if issubclass(t, winner):
                winner = t
            elif not issubclass(winner, t):
                raise TypeError("metaclass conflict", winner, t)
        return winner
