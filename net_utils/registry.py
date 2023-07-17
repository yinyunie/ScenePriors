#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

import inspect

class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key, alter_key = None):
        if key in self._module_dict:
            return self._module_dict.get(key)
        else:
            return self._module_dict.get(alter_key, None)

    def _register_module(self, module_class):
        '''
        register a module.
        :param module_class (`nn.Module`): Module to be registered.
        :return:
        '''
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls