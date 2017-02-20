# -*- coding: utf-8 -*-
import os
import yaml


def loader(file_path):
    return yaml.load(open(file_path))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_TOKEN = loader(os.path.join(BASE_DIR, 'config/slack.yaml'))['token']

default_reply = "スイマセン。其ノ言葉ワカリマセン"

PLUGINS = [
    'plugins',
]