import json
import math
from io import StringIO
import pandas as pd
import  copy

# monster app path
mh_play_it_path = '/Users/einvcz/workspace/monster/flutter/mh-play-kit/l10n/kit_ko.arb'

# excel_file_path
excel_file = '/Users/einvcz/workspace/python/offline/files/monster_app_translate.xlsx'


excel_map = {}

with open(mh_play_it_path, 'rb') as mp_file:
    file_content = mp_file.read()
    mp_map = json.loads(file_content)

df = pd.read_excel(excel_file, sheet_name='mh-play-kit')
key = df.iloc[0:, 0]
ko_translate_res = df.iloc[0:, len(df.columns) - 1]
if len(key) != len(ko_translate_res):
    print(f'data not consistency. sheet name is mp-play')

for i in range(len(key)):
    excel_map[key[i]] = ko_translate_res[i]

for key in mp_map.keys():
    if key in excel_map.keys() and not (isinstance(excel_map[key], float) and math.isnan(excel_map[key])):
        mp_map[key] = excel_map[key]

with open(mh_play_it_path, 'w') as res_file:
    res_file.write(json.dumps(mp_map, ensure_ascii=False, indent=2))




