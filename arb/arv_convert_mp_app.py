import json
import math
import pandas as pd


# monster app path
mp_app_path = '/Users/einvcz/workspace/monster/flutter/mp-app/l10n'

# excel_file_path
excel_file = '/Users/einvcz/workspace/python/offline/files/monster_app_translate.xlsx'

# xxx_ko.arb ,  account --> app
mp_sub_files = ['account', 'app', 'character', 'chat', 'community', 'device', 'health', 'play', 'purchase', 'user']
for file in mp_sub_files:
    excel_map = {}
    full_path = f'{mp_app_path}/{file}/app_ko.arb'

    with open(full_path, 'rb') as mp_file:
        file_content = mp_file.read()
        mp_map = json.loads(file_content)

    df = pd.read_excel(excel_file, sheet_name=file)
    key = df.iloc[0:, 0]
    ko_translate_res = df.iloc[0:, len(df.columns) - 1]
    if len(key) != len(ko_translate_res):
        print(f'data not consistency. sheet name is {file}')
        continue

    for i in range(len(key)):
        excel_map[key[i]] = ko_translate_res[i]

    for key in mp_map.keys():
        # if key in excel_map.keys():
        if key in excel_map.keys() and not (isinstance(excel_map[key], float) and math.isnan(excel_map[key])):
            mp_map[key] = excel_map[key]
            # if file == 'app':
            #     print(f'{key} {excel_map[key]} {type(excel_map[key])}')
    # print(json.dumps(mp_map, ensure_ascii=False))
    # break
    with open(full_path, 'w') as res_file:
        res_file.write(json.dumps(mp_map, ensure_ascii=False, indent=4))


# # reading excel
# excel_file = '/Users/einvcz/workspace/python/offline/files/monster_app_translate.xlsx'
# for sheet_name in pd.ExcelFile(excel_file).sheet_names:
#     df = pd.read_excel(excel_file, sheet_name=sheet_name)
#     key = df.iloc[0:, 0]
#     ko_translate_res = df.iloc[0:, len(df.columns) - 1]
#     if len(key) != len(ko_translate_res):
#         print(f'data not consistency. sheet name is {sheet_name}')
#         continue
#     buffer = StringIO()
#     for i in range(len(key)):
#         data = f'"{key[i]}": "{ko_translate_res[i]}"'
#         if i != len(key) - 1:
#             data += ',\n'
#         buffer.write(data)
#
#     resFilePath = '/Users/einvcz/workspace/python/offline/files/arb_res'
#     resFileName = resFilePath + f'/{sheet_name}_ko.arb'
#     with open(resFileName, 'w') as file:
#         file.write(buffer.getvalue())
