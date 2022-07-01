import pandas as pd
import numpy as np
import matplotlib.pyplot as mp


# occurence of statistics
def make_count(flu_ann):
    dict_count = {}
    for degree in range(5):
        for flu in flu_ann:
            count = dict_count.get(str(degree + 1), 0)
            if flu == (degree + 1):
                count += 1
            dict_count[str(degree + 1)] = count
    return dict_count


# file name 
file_name = 'Human_evaluation_Ann1-6.xlsx'
# tab names
sheet_name = ["Human_evaluation_Ann1", "Human_evaluation_Ann2", "Human_evaluation_Ann3", "Human_evaluation_Ann4", "Human_evaluation_Ann5", "Human_evaluation_Ann6"]
df = pd.read_excel(file_name, sheet_name=sheet_name)

sheet_name_short = [name.split('_')[-1] for name in sheet_name]

# raw statistics for each page
row_ann1 = df['Human_evaluation_Ann1'].to_numpy().T
row_ann2 = df['Human_evaluation_Ann2'].to_numpy().T
row_ann3 = df['Human_evaluation_Ann3'].to_numpy().T
row_ann4 = df['Human_evaluation_Ann4'].to_numpy().T
row_ann5 = df['Human_evaluation_Ann5'].to_numpy().T
row_ann6 = df['Human_evaluation_Ann6'].to_numpy().T

"""Accuracy-related score """
acu_ann1 = row_ann1[3].tolist()
acu_ann2 = row_ann2[3].tolist()
acu_ann3 = row_ann3[3].tolist()
acu_ann4 = row_ann4[3].tolist()
acu_ann5 = row_ann5[3].tolist()
acu_ann6 = row_ann6[3].tolist()


acu_all_np = np.array([acu_ann1, acu_ann2, acu_ann3, acu_ann4, acu_ann5, acu_ann6])
list_acu = []
# rating score
item_num = 2
# correct num
sum_acu = 0.0
# num of sent in total
sen_total_num = len(acu_ann1)
for sen in acu_all_np.T:
    list_tmp = [0]*item_num
    for i in sen:
        if i == 1:
            list_tmp[0] += 1
            sum_acu += 1
        else:
            list_tmp[1] += 1
    list_acu.append(list_tmp)
p_l_total = 0

for sen_acu in list_acu:
    # P_1 = (10 ** 2 + 0 ** 2 - 10) / (10 * 9) = 1
    p_l = (sen_acu[0]**2 + sen_acu[1]**2 - (sum(sen_acu)))/(sum(sen_acu)*(sum(sen_acu)-1))
    p_l_total += p_l

ann_num = len(acu_all_np)

# p_1 = 34 / (5 * 10) = 0.68
c_p_l = sum_acu / (sen_total_num * ann_num)

# P_bar = (1 / 5) * (1 + 0.64 + 0.8 + 1 + 0.53) = 0.794
p_bar = (1/sen_total_num) * p_l_total

# P_bar_e = 0.68 ** 2 + 0.32 ** 2 = 0.5648
p_bar_e = c_p_l**2 + (1-c_p_l)**2

f_kappa = (p_bar - p_bar_e) / (1-p_bar_e)

print(f'flessis_kappa: {f_kappa:.5f}')

# kappa = sm.cohen_kappa_score(acu_ann1, acu_ann2)
# print(f'kappa: {kappa:.5f}')
# print(f'fleiss_kappa_acc:{ans:.5f}')
"""fluency-related statistics """


flu_ann1 = row_ann1[4]
flu_ann2 = row_ann2[4]
flu_ann3 = row_ann3[4]
flu_ann4 = row_ann4[4]
flu_ann5 = row_ann5[4]
flu_ann6 = row_ann6[4]
""" average fluency for each sent """
flu_all_np = np.array([flu_ann1, flu_ann2, flu_ann2, flu_ann4, flu_ann5, flu_ann6]).T
flu_all_df = pd.DataFrame(flu_all_np, columns=sheet_name_short, index=np.arange(1, 51, 1))
print('-------- fluency average for each sent--------')
print(flu_all_df.mean(axis=1))

"""plot the statistics  """




ann_list = [flu_ann1, flu_ann2, flu_ann3, flu_ann4, flu_ann5, flu_ann6]
flu_ann_list = [[i for i in make_count(ann).values()] for ann in ann_list]
# index
df_flu = pd.DataFrame(flu_ann_list, index=sheet_name_short, columns=['1', '2', '3', '4', '5'])

print('--------fluency 統計值--------')
print(df_flu)

# ans_f = fleiss_kappa(df_flu.to_numpy(), N=2, k=5, n=50)
# print(f'fleiss_kappa_flu:{ans_f:.5f}')

df_flu.plot(title='Fluency', kind='bar')
mp.show()

"""
Accuracy kappa: 0.5584 
