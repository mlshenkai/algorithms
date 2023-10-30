# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/10/13 8:39 PM
# @File: __init__.py
# @Email: mlshenkai@163.com
from .model import Model
from .losses import build_loss, build_kd_loss

__all__ = ['build_loss', 'build_model', 'build_kd_loss']
support_model = ['Model']


def build_model(model_name, **kwargs):
    assert model_name in support_model, f'all support model is {support_model}'
    model = eval(model_name)(kwargs)
    return model