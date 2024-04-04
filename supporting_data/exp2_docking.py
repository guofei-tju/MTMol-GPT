import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def process_data(data_ser, box_scale=3):
    iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
    # 下阈值
    val_low = data_ser.quantile(0.25) - iqr * 0.5
    # 上阈值
    val_up = data_ser.quantile(0.75) + iqr * 0.5
    # 异常值
    outlier = data_ser[(data_ser < val_low) | (data_ser > val_up)]
    # 正常值
    normal_value = data_ser[(data_ser >= val_low) & (data_ser < val_up)]
    return outlier, normal_value, (val_low, val_up)

docking_name = 'DRD2'

if docking_name == 'DRD2':
    docking = pd.read_csv('data/DRD2_docking_decision.csv')
elif docking_name =='HTR1A':
    docking = pd.read_csv('data/HTR1A_docking_decision.csv')


processed_docking_data = pd.DataFrame()
ranges = []
for cols in docking.columns:
    if cols == 'affinity_6LUQ_H-Chembl24':
        continue
    _, processed_data, range = process_data(docking[cols])
    processed_docking_data[cols] = processed_data
    ranges.append(range)

ax = plt.figure()
datas = [processed_docking_data['affinity-inactive'].dropna(),
         processed_docking_data['affinity_6LUQ_H-gail_sm'].dropna(),
         processed_docking_data['affinity_6LUQ_H-gail_sf'].dropna(),
         processed_docking_data['affinity-active'].dropna()]
# sns.violinplot(data=datas,color='skyblue')
plt.violinplot(datas,
               showextrema=False, showmeans=False, showmedians=False)
for index, data in enumerate(datas):
    quartile1, medians, quartile3 = np.percentile(np.array(data), [25, 50, 75])
    plt.scatter(index + 1, medians, marker='o', color='white', s=10, zorder=3)
    plt.vlines(index + 1, quartile1, quartile3, color='#1f77b4', linestyle='-', lw=5)

    upper_adjacent_value = np.clip(ranges[index][1], quartile3, data.max())
    lower_adjacent_value = np.clip(ranges[index][0], data.min(), quartile1)
    plt.vlines(index + 1, lower_adjacent_value, upper_adjacent_value, color='#1f77b4', linestyle='-', lw=1)
labels = ['Inactive', 'MTMol-GPT', 'SF-MTMol-GPT', 'Active']
plt.xticks(np.arange(1, len(labels) + 1), labels=labels)
plt.show()
print('...')
