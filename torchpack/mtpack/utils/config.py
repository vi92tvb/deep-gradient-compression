import collections
import importlib.util
import os
from collections import deque
import collections

collections.Mapping = collections.abc.Mapping
collections.Sequence = collections.abc.Sequence

import six

from .container import G

__all__ = ['Config', 'configs', 'update_configs_from_module',
           'update_configs_from_arguments']


class Config(G):
    def __init__(self, func=None, args=None, detach=False, **kwargs):
        if func is not None and not callable(func):
            raise TypeError(
                f'func "{repr(func)}" is not a callable function or class')
        if args is not None \
            and not isinstance(args, (collections.Sequence,
                                      collections.UserList)):
            raise TypeError(
                f'args "{repr(args)}" is not an iterable tuple or list')

        super().__init__(**kwargs)
        self._func_ = func
        self._args_ = args
        self._detach_ = detach

    def items(self):
        for k, v in super().items():
            if k not in ['_func_', '_args_', '_detach_']:
                yield k, v

    def keys(self):
        for k, v in self.items():
            yield k

    def values(self):
        for k, v in self.items():
            yield v

    def __call__(self, *args, **kwargs):
        if self._func_ is None:
            return self

        # override args
        if args:
            args = list(args)
        elif self._args_:
            args = list(self._args_)

        # override kwargs
        for k, v in self.items():
            if k not in kwargs:
                kwargs[k] = v

        # recursively call non-detached funcs
        queue = deque([args, kwargs])
        while queue:
            x = queue.popleft()

            if not isinstance(x, six.string_types) \
                    and isinstance(x, (collections.Sequence,
                                       collections.UserList)):
                items = enumerate(x)
            elif isinstance(x, (collections.Mapping, collections.UserDict)):
                items = x.items()
            else:
                items = []

            for k, v in items:
                if isinstance(v, tuple):
                    v = x[k] = list(v)
                elif isinstance(v, Config):
                    if v._detach_:
                        continue
                    v = x[k] = v()
                queue.append(v)

        return self._func_(*args, **kwargs)

    def __str__(self, indent=0):
        text = ''
        if self._func_ is not None:
            text += ' ' * indent + '[func] = ' + str(self._func_)
            if self._detach_:
                text += '(detach=' + str(self._detach_) + ')'
            text += '\n'
            if self._args_:
                for k, v in enumerate(self._args_):
                    text += ' ' * indent + \
                        '[args:' + str(k) + '] = ' + str(v) + '\n'

        for k, v in self.items():
            text += ' ' * indent + '[' + str(k) + ']'
            if isinstance(v, Config):
                text += '\n' + v.__str__(indent + 2)
            else:
                text += ' = ' + str(v)
            text += '\n'

        while text and text[-1] == '\n':
            text = text[:-1]
        return text

    def __repr__(self):
        text = ''
        if self._func_ is not None:
            text += repr(self._func_)

        items = []
        if self._func_ is not None and self._args_:
            items += [repr(v) for v in self._args_]
        items += [str(k) + '=' + repr(v) for k, v in self.items()]
        if self._func_ is not None and self._detach_:
            items += ['detach=' + repr(self._detach_)]

        text += '(' + ', '.join(items) + ')'
        return text

    @staticmethod
    def update_from_modules(*modules):
        for module in modules:
            module = module.replace('.py', '').replace('/', '.')
            importlib.import_module(module)

    @staticmethod
    def update_from_arguments(*args):
        update_configs_from_arguments(args)


configs = Config()


def update_configs_from_module(*mods):
    imported_modules = set()

    # from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    def exec_module_once(module):
        if module in imported_modules:
            return
        imported_modules.add(module)
        spec = importlib.util.spec_from_file_location(
            os.path.basename(module), module)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

    for mod in mods:
        mod = os.path.normpath(mod)
        for index, char in enumerate(mod):
            if index == 0 or char == os.sep:
                submod = os.path.join(mod[:index], '__init__.py')
                if os.path.exists(submod):
                    exec_module_once(submod)
        exec_module_once(mod)


def update_configs_from_arguments(args):
    index = 0

    while index < len(args):
        arg = args[index]

        if arg.startswith('--configs.'):
            arg = arg.replace('--configs.', '')
        else:
            raise Exception('unrecognized argument "{}"'.format(arg))

        if '=' in arg:
            index, keys, val = index + \
                1, arg[:arg.index('=')].split('.'), arg[arg.index('=') + 1:]
        else:
            index, keys, val = index + 2, arg.split('.'), args[index + 1]

        config = configs
        for k in keys[:-1]:
            if k not in config:
                config[k] = Config()
            config = config[k]

        def parse(x):
            if (x[0] == '\'' and x[-1] == '\'') \
                    or (x[0] == '\"' and x[-1] == '\"'):
                return x[1:-1]
            try:
                x = eval(x)
            except:
                pass
            return x

        config[keys[-1]] = parse(val)
