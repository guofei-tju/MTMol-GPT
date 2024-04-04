from operator import itemgetter
import seaborn as sns
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem
from moses.metrics.SA_Score import sascorer
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# DRD2_docking_score = -9.68
# HTR1A_docking_score = -6.06
DRD2_docking_score = -9.65
HTR1A_docking_score = -9.26


def Estimate_logP(mol_list):
    logPs = []
    for mol in mol_list:
        if mol is None:
            logPs.append(0)
        else:
            try:
                logPs.append(Chem.Crippen.MolLogP(mol))
            except:
                logPs.append(0)
    return logPs


def Estimate_QED(mol_list):
    QEDS = []
    for mol in mol_list:
        if mol is None:
            QEDS.append(0)
        else:
            try:
                QEDS.append(QED.qed(mol))
            except:
                QEDS.append(0)
    return QEDS


def Estimate_SA(mol_list):
    SAs = []
    for mol in mol_list:
        if mol is None:
            SAs.append(0)
        else:
            SAs.append(sascorer.calculateScore(mol))
    return SAs


def cal_T_score(gen_mols, training_mols):
    score_list = []
    for gen_mol in gen_mols:
        gen_mol_score_list = []
        for mol in training_mols:
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 3, nBits=2048)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
            score = DataStructs.FingerprintSimilarity(gen_fp, fp)
            gen_mol_score_list.append(score)
        score_list.append(gen_mol_score_list)
    return score_list


target1 = 'DRD2'
target2 = 'HTR1A'
database = 'DLGN'
smiles_path = f'.data/'

# import the train and test data and the chembl data

target1_data_path = f'{smiles_path}/DLGN/{target1}_test.txt'
target1_data = pd.read_csv(target1_data_path, header=None).values.reshape(-1)

target2_data_path = f'{smiles_path}/DLGN/{target2}_test.txt'
target2_data = pd.read_csv(target2_data_path, header=None).values.reshape(-1)

X_target1 = [Chem.MolFromSmiles(s) for s in target1_data]
X_target2 = [Chem.MolFromSmiles(s) for s in target2_data]
target_docking_score = [DRD2_docking_score, HTR1A_docking_score]
# generated samples
smiles_df = pd.read_csv(f'data/MTMol-GPT/gail_e30_6LUQ_H_7E2Y_H_sample1000_size25_affinity.csv')


def plot_tsne_kde(smiles_df, X_train_fp, target1_data, target2_data, target_docking_score):
    # process target data
    target1_fp = np.array(
        [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 3, 2048).ToList() for x in target1_data])
    target2_fp = np.array(
        [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 3, 2048).ToList() for x in target2_data])
    # plot target tsne
    data_fp = np.concatenate((target1_fp, target2_fp, X_train_fp))
    tsne = TSNE(n_components=2, init='pca', perplexity=50, random_state=0)
    tsne_features = tsne.fit_transform(data_fp)

    fig = plt.figure(figsize=(10, 8))

    # sns.kdeplot(x=tsne_features[:len(target1_fp), 0], y=tsne_features[:len(target1_fp), 1],
    #             # hue=['dataset2'] * len(target2_tsne_features[:, 0])                ,
    #             cmap="Reds", bw=0.3, shade=True,
    #             )
    sns.kdeplot(x=tsne_features[len(target1_fp):len(target1_fp) + len(target2_fp), 0],
                y=tsne_features[len(target1_fp):len(target1_fp) + len(target2_fp), 1],
                # hue=['dataset1'] * len(target1_tsne_features[:, 0])                ,
                cmap="YlOrBr", bw=0.3, shade=True,
                # alpha=0.6,
                )
    # select_nodes = smiles_df[smiles_df['affinity_6LUQ_H-gail'] <= -9.68][smiles_df['affinity_7E2Y_H-gail'] <= -6.06][smiles_df['Ts_DRD2'] >= 0.5][smiles_df['Ts_HTR1A'] >= 0.5]
    # node_indexs = select_nodes['Unnamed: 0']
    # node_indexs = [372, 194, 294, 714, 943, 829, 926, 343]
    label_x = smiles_df['affinity_6LUQ_H-gail']
    label_y = smiles_df['affinity_7E2Y_H-gail']
    # DRD2_Ts = smiles_df['Ts_DRD2']
    # HTR1A_Ts = smiles_df['Ts_HTR1A']
    # label = label_x
    label = label_y
    # label = (label_x + label_y) / 2
    min_v = min((label))
    max_v = max((label))
    color = [plt.get_cmap("Blues", 100)(100 - int(float(i - min_v) / (max_v - min_v) * 100)) for i in
             (label.sort_values(ascending=False))]
    original_colors = color.copy()
    # colors
    # for color_index in node_indexs:
    #     color[color_index] = 'black'
    #     # 使用plt.text在散点图上添加文本
    #     corr_x = tsne_features[len(target1_fp) + len(target2_fp) + color_index, 0]
    #     corr_y = tsne_features[len(target1_fp) + len(target2_fp) + color_index, 1]
    #     plt.text(corr_x, corr_y, color_index, ha='center', va='bottom')
    x = tsne_features[len(target1_fp) + len(target2_fp):, 0]
    y = tsne_features[len(target1_fp) + len(target2_fp):, 1]
    sc = plt.scatter(tsne_features[len(target1_fp) + len(target2_fp):, 0][label.sort_values(ascending=False).index],
                     tsne_features[len(target1_fp) + len(target2_fp):, 1][label.sort_values(ascending=False).index],
                     c=color,
                     cmap='Blues', marker='o', picker=True)

    # sns.kdeplot(x=tsne_features[len(target1_fp) + len(target2_fp):, 0],
    #             y=tsne_features[len(target1_fp) + len(target2_fp):, 1],
    #             #             hue=['dataset1'] * len(target1_tsne_features[:, 0])                ,
    #             cmap="Blues", bw=0.25, shade=True,
    #             alpha=0.5,
    #             )
    xlim_min, xlim_max = -70, 75
    ylim_min, ylim_max = -50, 55
    plt.legend(loc="upper right")
    scatter_legend = plt.legend(handles=[plt.scatter([], [], cmap='Blues', label='Dataset generated')],
                                loc='upper right')
    # set color bar
    norm = plt.Normalize(max_v, min_v)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='Blues_r')
    # # 添加颜色条到图形
    plt.colorbar(sm)
    plt.axis('off')

    # plt.title('t-SNE Visualization of Generated and DRD2_test Molecules Fingerprints')
    # plt.title('t-SNE Visualization of Generated and HTR1A_test Molecules Fingerprints')
    # plt.title('t-SNE Visualization of Generated, DRD2_test and HTR1A_test Molecules Fingerprints')
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)

    # plt.xlabel('t-SNE Feature 1')
    # plt.ylabel('t-SNE Feature 2')
    # plt.savefig('test_DRD2_kde1.png')
    # plt.savefig('test_HTR1A_kde1.png')
    # plt.savefig('test_DRD2_HTR1A_kde1_colored.png')

    # selected_indices = set()

    # def onclick(event):
    #     if event.artist == sc:
    #         # 获取被点击点的索引
    #         ind = event.ind
    #         for index in ind:
    #             # 检查点是否已经被选中
    #             if index in selected_indices:
    #                 # 如果点已经被选中，则取消选中并恢复原始颜色
    #                 selected_indices.remove(index)
    #                 color[index] = original_colors[index]
    #             else:
    #                 # 如果点未被选中，则选中并改变颜色
    #                 selected_indices.add(index)
    #                 # color1 = 'purple'  # 可以根据需要更改颜色
    #                 color[index] = 'black'
    #                 print(
    #                     f"Index {index},smiles is: {list(smiles_df['smiles'].values)[index]}; \n"
    #                     f"docking score on DRD2 is: {label_x[index]}, on HTR1A is : {label_y[index]}, DRD2_Ts is: {DRD2_Ts[index]}, HTR1A_TS is: {HTR1A_Ts[index]}")
    #                 # for color_index in node_indexs:
    #                 #     color[color_index] = 'purple'
    #                 # 更新点的颜色
    #             sc.set_facecolors(color)
    #             fig.canvas.draw_idle()
    def select_area_and_get_indices(x, y):
        # 用户选择两个点定义矩形区域
        print("请点击散点图上的两个对角点来定义一个区域")
        points = plt.ginput(2, timeout=0)
        if len(points) != 2:
            print("未正确选择两个点，请重试")
            return None

            # 计算矩形区域的边界
        x_min, x_max = min(points[0][0], points[1][0]), max(points[0][0], points[1][0])
        y_min, y_max = min(points[0][1], points[1][1]), max(points[0][1], points[1][1])

        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                          fill=False, edgecolor='black', linewidth=2))
        plt.draw()

        # 找出在区域内的点的索引
        indices_in_area = np.where((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max))

        return indices_in_area[0]  # 返回点的索引数组

    # 获取区域内点的索引
    indices = select_area_and_get_indices(x, y)
    if indices is not None:
        print("在区域内的点的索引：", indices)

        # 将结果写入Excel文件
        df = pd.DataFrame({'Index': indices, 'smiles': smiles_df.iloc[indices]['smiles'], \
                           'DRD2_Ds': smiles_df.iloc[indices]['affinity_6LUQ_H-gail'], \
                           'HTR1A_Ds': smiles_df.iloc[indices]['affinity_7E2Y_H-gail'], \
                           'DRD2_Ts': smiles_df.iloc[indices]['Ts_DRD2'],
                           'HTR1A_Ts': smiles_df.iloc[indices]['Ts_HTR1A']})
        df.to_excel('points_test_HTR1A_in_area.xlsx', index=False)

    # 显示图形并等待用户操作
    plt.show()

    pass


# Convet the smiles seq to mol format
X_train = [Chem.MolFromSmiles(s) for s in list(smiles_df['smiles'].values)]

# get the fingerprint of train data
X_train_fp = np.array([AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048).ToList() for x in X_train])

plot_tsne_kde(smiles_df, X_train_fp, target1_data, target2_data, target_docking_score)

# smiles_df.dropna(inplace=True)
# # delete the duplicated data
# same_target1 = list(set(smiles_df['smiles']) & set(train_data1))
# same_target2 = list(set(smiles_df['smiles']) & set(train_data2))
# same_target3 = list(set(smiles_df['smiles']) & set(chembl_data))
# for i in same_target1:
#     smiles_df.drop(smiles_df[smiles_df['smiles'].str.contains(pat=f'{i}', regex=False) == True].index,
#                    inplace=True)
# for i in same_target2:
#     smiles_df.drop(smiles_df[smiles_df['smiles'].str.contains(pat=f'{i}', regex=False) == True].index,
#                    inplace=True)
# for i in same_target3:
#     smiles_df.drop(smiles_df[smiles_df['smiles'].str.contains(pat=f'{i}', regex=False) == True].index,
#                    inplace=True)
#
# # Convet the smiles seq to mol format
# X_train = [Chem.MolFromSmiles(s) for s in list(smiles_df['smiles'].values)]
#
# # get the fingerprint of train data
# X_train_fp = np.array([AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048).ToList() for x in X_train])
#
# # cat the data
# smiles_df.drop(columns=['index'], inplace=True)
# smiles_df['mol'] = X_train
#
# # plot TSNE
# plot_tsne_cs(smiles_df, X_train_fp, X_target1, X_target2, target_docking_score)
#
#
# # select 3-d molecules
# def select_3d_lig(smiles_df):
#     # 计算iqr：数据四分之三分位值与四分之一分位值的差
#     iqr_target1 = smiles_df['affinity_6LUQ_H-gail'].quantile(0.75) - smiles_df['affinity_6LUQ_H-gail'].quantile(0.25)
#     # 根据iqr计算异常值判断阈值
#     u_th_target1 = smiles_df['affinity_6LUQ_H-gail'].quantile(0.75) + 1.5 * iqr_target1  # 上界
#     l_th_target1 = smiles_df['affinity_6LUQ_H-gail'].quantile(0.25) - 1.5 * iqr_target1  # 下界
#
#     iqr_target2 = smiles_df['affinity_7E2Y_H-gail'].quantile(0.75) - smiles_df['affinity_7E2Y_H-gail'].quantile(0.25)
#     # 根据iqr计算异常值判断阈值
#     u_th_target2 = smiles_df['affinity_7E2Y_H-gail'].quantile(0.75) + 1.5 * iqr_target2  # 上界
#     l_th_target2 = smiles_df['affinity_7E2Y_H-gail'].quantile(0.25) - 1.5 * iqr_target2  # 下界
#
#     return \
#         smiles_df[smiles_df['affinity_6LUQ_H-gail'] >= l_th_target1][smiles_df['affinity_6LUQ_H-gail'] <= u_th_target1][
#             smiles_df['affinity_7E2Y_H-gail'] >= l_th_target2][smiles_df['affinity_7E2Y_H-gail'] <= u_th_target2]
#
#
# select_3d_lig()
