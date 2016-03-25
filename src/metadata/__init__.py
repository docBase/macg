# -*- coding: utf-8 -*-

import os

import pandas as pd
import numpy as np

pd.set_option('display.width', None)

BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR, '..', '..', 'data')


def set_id(row):
    if row['Series type'] == 'Aggregate':
        return row['GLOBAL ID']
    elif pd.isnull(row['parents']):
        return "{0:0>2}".format(int(row['cat_order']))
    else:
        return "{0}.{1:0>2}".format(row['parents'], int(row['cat_order']))

def get_metadata():
    metadata = pd.read_csv(DATADIR + '/GCI_Dataset_2006-2015.metadata.csv')
    fixed_ids = pd.read_csv(DATADIR + '/fixed_ids.csv')
    metadata = metadata.merge(fixed_ids, how='left')
    metadata['id'] = metadata.apply(lambda row: set_id(row), axis=1)
    metadata.sort_values('id', inplace=True)
    metadata.reset_index(inplace=True)
    return metadata

metadata = get_metadata()
