'''
@Author: your name
@Date: 2020-01-06 17:08:52
@LastEditTime : 2020-01-07 13:25:31
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /KGCN_Keras-master/utils/io.py
'''
# -*- coding: utf-8 -*-

import os
import json
import pickle


def pickle_load(filename: str):
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        print(f'Logging Info - Loaded: {filename}')
    except EOFError:
        print(f'Logging Error - Cannot load: {filename}')
        obj = None

    return obj


def pickle_dump(filename: str, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Logging Info - Saved: {filename}')


def write_log(filename: str, log, mode='w'):
    with open(filename, mode) as writers:
        writers.write('\n')
        json.dump(log, writers, indent=4, ensure_ascii=False)


def format_filename(_dir: str, filename_template: str, **kwargs):
    """Obtain the filename of data base on the provided template and parameters"""
    filename = os.path.join(_dir, filename_template.format(**kwargs))
    return filename
