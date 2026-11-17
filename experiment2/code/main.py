#!/usr/bin/env python
# coding=utf-8
import pandas as pd

df = pd.read_csv('../data/bank-full.csv', sep=';')

'''
data like these:
   age           job  marital  education default  balance housing loan  ... day  month duration  campaign  pdays  previous  poutcome   y
0   58    management  married   tertiary      no     2143     yes   no  ...   5    may      261         1     -1         0   unknown  no
1   44    technician   single  secondary      no       29     yes   no  ...   5    may      151         1     -1         0   unknown  no
2   33  entrepreneur  married  secondary      no        2     yes  yes  ...   5    may       76         1     -1         0   unknown  no
3   47   blue-collar  married    unknown      no     1506     yes   no  ...   5    may       92         1     -1         0   unknown  no
4   33       unknown   single    unknown      no        1      no   no  ...   5    may      198         1     -1         0   unknown  no

'''
print(df.shape)
print(df.head())
print(df.info())
