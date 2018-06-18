# -*- coding: utf-8 -*-

# little path hack to access module which is one directory up.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mdn
