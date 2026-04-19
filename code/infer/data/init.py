#!/usr/bin/env python3
"""
数据集模块入口，新增数据集后在这里导入即可
"""
from .scienceqa import load_scienceqa
from .realworldqa_openqa import load_realworldqa_openqa

# 新增数据集示例：
# from .aokvqa import load_aokvqa
# from .vizwiz import load_vizwiz