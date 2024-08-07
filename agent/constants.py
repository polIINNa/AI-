import json

import pandas as pd


with open('/Users/21109090/Downloads/deposits_2.json', 'r') as f:
    data = json.load(f)
output = []
for deposit_name in data.keys():
    d = {'Название': deposit_name}
    for key in data[deposit_name].keys():
        d[key] = data[deposit_name][key]
    output.append(d)
df = pd.json_normalize(output)
df_list = [df]
df_dic = {'df_deposit': df}
