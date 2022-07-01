#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import sklearn.metrics as sm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import csv


PATH_1 = "./project3/project3_version1.jsonl"
PATH_2 = "./project3/project3_version2.jsonl"

# data transformation
def data_input(path,dict_label):
    data_list = []
    empty_span = []
    total = 0
    with open(path, encoding='utf-8-sig') as f:
        data = [json.loads(line) for line in f]
        for text_all in data:
            data_per_list = []
            for text_token in text_all['spans']:
                label = text_token['label']
                index = str(text_token["start"])+'_'+str(text_token["end"])
                label_count = dict_label.get(label,0)
                label_count += 1
                dict_label[label] = label_count
                tuple_tmp = (index, label)
                data_per_list.append(tuple_tmp)
            if data_per_list == []:
                empty_idx = data.index(text_all)
                print(f'{path.split("/")[-1]},第{empty_idx+1}筆, have empty text_all["spans"]')
                empty_span.append(empty_idx)
            data_list.append(data_per_list)
            total += len(text_all['spans'])
    return data_list,empty_span


""" token alignments, if the annotated tokens are different, just skip the instance"""

# compare token and record their list; return the list of index

# compare token and record their list; return the list of index

def compare_data(list_np1, list_np2, dict_no_idx,dict_no_lab):
    list_idx_np1 = list_np1[0].tolist()
    list_idx_np2 = list_np2[0].tolist()
    list_lab_np1 = list_np1[1]
    list_lab_np2 = list_np2[1]
    set1 = set(list_idx_np1)
    set2 = set(list_idx_np2)
    list_np1_no = list(set1-set2)
    list_np2_no = list(set2-set1)
    for i in list_np1_no:
        idx = list_idx_np1.index(i)
        label = list_lab_np1[idx]
        no_label_count1 = dict_no_idx.get(label, 0)
        dict_no_idx[label] = no_label_count1 + 1
    for u in list_np2_no:
        idx = list_idx_np2.index(u)
        label = list_lab_np2[idx]
        no_label_count2 = dict_no_idx.get(label, 0)
        dict_no_idx[label] = no_label_count2 + 1

    set1_plus_set2 = list(set1 & set2)

    for k in set1_plus_set2:
        idx1 = list_idx_np1.index(k)
        idx2 = list_idx_np2.index(k)
        label_1 = list_lab_np1[idx1]
        label_2 = list_lab_np2[idx2]
        if label_1 != label_2:
            no_label1 = dict_no_lab.get(label_1,0)
            dict_no_lab[label_1] = no_label1+1
            no_label2 = dict_no_lab.get(label_2,0)
            dict_no_lab[label_2] = no_label2+1
    return list_np1_no, list_np2_no


"""delete the data and sort them in order"""
def make_pre_data(data_row, data_del):
    for del_token in data_del:
        for idx in range(-1, -len(data_row) - 1, -1):
            if data_row[idx][0] == del_token:
                del data_row[idx]
                break
    data_row.sort()

#1. input data
dict_label = {}
data1_pre_row, data1_pre_empty = data_input(PATH_1, dict_label)
data2_pre_row, data2_pre_empty = data_input(PATH_2, dict_label)

""" retrieve the intersection of data1 and data2_empty, building the mask for empty_sapn """
same_empty = np.array(list(set(data1_pre_empty) | set(data2_pre_empty)))
mask_row = np.ones(len(data1_pre_row))

for e in same_empty:
    mask_row[e] = 0
# mask = mask_row > 0

mask = mask_row > 0

data1_row = np.array(data1_pre_row)[mask]
data2_row = np.array(data2_pre_row)[mask]
print('data1_row:', len(data1_row))
print('data2_row:', len(data2_row))
data1_empty = np.array(data1_pre_row)[~mask]
data2_empty = np.array(data2_pre_row)[~mask]
print('data1_empty:', len(data1_empty))
print('data2_empty:', len(data2_empty))

# dict_no_idx: 存放資料不同的token idx 數量統計
dict_no_idx = {}
for idx_e in range(len(data1_empty)):
    e_data1 = data1_empty[idx_e]
    e_data2 = data2_empty[idx_e]
    if e_data1 != []:
        for text1 in e_data1:
            e_label_1 = text1[1]
            no_label_count1 = dict_no_idx.get(e_label_1, 0)
            dict_no_idx[e_label_1] = no_label_count1 + 1
    if e_data2 !=[]:
        for text2 in e_data2:
            e_label_2 = text2[1]
            no_label_count2 = dict_no_idx.get(e_label_2, 0)
            dict_no_idx[e_label_2] = no_label_count2 + 1

print('data aligment; calculate the number of labels that are different from each other:',dict_no_idx)

# dict_no_lab: 存放相同token idx 但 不同的label 數量統計
dict_no_lab = {}
data1_row_final = []
data2_row_final = []

##################
iden_label = {}
# iden_knowledge, iden_skill, iden_area, iden_diploma, iden_major, iden_experience = 0,0,0,0,0,0
###############
dict2_label = {}
# knowledge_data2, skill_data2, area_data2, diploma_data2, major_data2, experience_data2 = 0,0,0,0,0,0
###############
dict1_label = {}
# knowledge_data1, skill_data1, area_data1, diploma_data1, major_data1, experience_data1 = 0,0,0,0,0,0


for i in range(len(data1_row)):
    # 2. transform list into numpy 
    data1 = np.array(data1_row[i]).T
    data2 = np.array(data2_row[i]).T
    #print('data2:', data2)
    # for data2_ent in data2[1]:
    #     if data2_ent == "knowledge":
    #         knowledge_data2 +=1
    #     elif data2_ent == "skills":
    #         skill_data2 +=1
    #     elif data2_ent == "areas":
    #         area_data2 +=1
    #     elif data2_ent == "diploma":
    #         diploma_data2 +=1
    #     elif data2_ent == "major":
    #         major_data2 +=1
    #     elif data2_ent == "experience":
    #         experience_data2 +=1
    for data2_ent in data2[1]:
        data2_label_count = dict2_label.get(data2_ent,0)
        dict2_label[data2_ent] = data2_label_count+1
        # if data2_ent == "knowledge":
        #     knowledge_data2 +=1
        # elif data2_ent == "skills":
        #     skill_data2 +=1
        # elif data2_ent == "areas":
        #     area_data2 +=1
        # elif data2_ent == "diploma":
        #     diploma_data2 +=1
        # elif data2_ent == "major":
        #     major_data2 +=1
        # elif data2_ent == "experience":
        #     experience_data2 +=1
    for da1_idx, da1_ent in zip(data1[0],data1[1]) :
        data1_label_count = dict1_label.get(da1_ent, 0)
        dict1_label[da1_ent] = data1_label_count + 1
        # if da1_ent == "knowledge":
        #     knowledge_data1 +=1
        # elif da1_ent == "skills":
        #     skill_data1 +=1
        # elif da1_ent == "areas":
        #     area_data1 +=1
        # elif da1_ent == "diploma":
        #     diploma_data1 +=1
        # elif da1_ent == "major":
        #     major_data1 +=1
        # elif da1_ent == "experience":
        #     experience_data1 +=1
        for da2_idx, da2_ent in zip(data2[0], data2[1]):
            if (da1_idx == da2_idx) and (da1_ent == da2_ent):
                iden_label_count = iden_label.get(da1_ent, 0)
                iden_label[da1_ent] = iden_label_count + 1
            # if (da1_idx == da2_idx) and (da1_ent =="knowledge" and da2_ent=="knowledge"):
            #     iden_knowledge +=1
            # elif (da1_idx == da2_idx) and (da1_ent =="skills" and da2_ent=="skills"):
            #     iden_skill +=1
            # elif (da1_idx == da2_idx) and (da1_ent =="areas" and da2_ent=="areas"):
            #     iden_area +=1
            # elif (da1_idx == da2_idx) and (da1_ent =="diploma" and da2_ent=="diploma"):
            #     iden_diploma +=1
            # elif (da1_idx == da2_idx) and (da1_ent =="major" and da2_ent=="major"):
            #     iden_major +=1
            # elif (da1_idx == da2_idx) and (da1_ent =="experience" and da2_ent=="experience"):
            #     iden_experience +=1
    # 3. extract the tokens from numpy array and retrive the token that are not aligned with each other.
    data1_del, data2_del = compare_data(data1, data2, dict_no_idx,dict_no_lab)
    # 4. delet and sort two data points 
    make_pre_data(data1_row[i], data1_del)
    make_pre_data(data2_row[i], data2_del)
    # assemble the processed data into a group
    for sen1 in data1_row[i]:
        data1_row_final.append(sen1)
    for sen2 in data2_row[i]:
        data2_row_final.append(sen2)

p_knowledge, r_knowledge = iden_label['knowledge'] / dict1_label['knowledge'], iden_label['knowledge'] / dict2_label[
    'knowledge']
f1_knowledge = (2 * p_knowledge * r_knowledge) / (p_knowledge + r_knowledge)
# print('fl knowledge: ', f1_knowledge)
print(f'fl knowledge: {f1_knowledge*100:.1f}')

p_skill, r_skill = iden_label['skills'] / dict1_label['skills'], iden_label['skills'] / dict2_label['skills']
f1_skill = (2 * p_skill * r_skill) / (p_skill + r_skill)
# print('fl skill: ', f1_skill)
print(f'fl skill: {f1_skill*100:.1f}')

p_area, r_area = iden_label['areas'] / dict1_label['areas'], iden_label['areas'] / dict2_label['areas']
f1_area = (2 * p_area * r_area) / (p_area + r_area)
# print('fl area: ', f1_area)
print(f'fl area: {f1_area*100:.1f}')

p_diploma, r_diploma = iden_label['diploma'] / dict1_label['diploma'], iden_label['diploma'] / dict2_label['diploma']
f1_diploma = (2 * p_diploma * r_diploma) / (p_diploma + r_diploma)
print(f'fl diploma: {f1_diploma*100:.1f}')

p_major, r_major = iden_label['major'] / dict1_label['major'], iden_label['major'] / dict2_label['major']
f1_major = (2 * p_major * r_major) / (p_major + r_major)
# print('fl major: ', f1_major)
print(f'fl major: {f1_major*100:.1f}')

p_experience, r_experience = iden_label['experience'] / dict1_label['experience'], iden_label['experience'] / \
                             dict2_label['experience']
f1_experience = (2 * p_experience * r_experience) / (p_experience + r_experience)
# print('fl experience: ', f1_experience)
print(f'fl experience: {f1_experience*100:.1f}')
# setup a dict for each label's F1-score 
dict_f1 = {'knowledge': f1_knowledge, 'skills': f1_skill, 'areas': f1_area, 'diploma': f1_diploma, 'major': f1_major,
           'experience': f1_experience}

"""
# p_knowledge, r_knowledge = iden_knowledge/ knowledge_data1, iden_knowledge/knowledge_data2
# f1_knowledge = (2*p_knowledge*r_knowledge) / (p_knowledge+r_knowledge)
# #print('fl knowledge: ', f1_knowledge)
#
# p_skill, r_skill = iden_skill/skill_data1, iden_skill/ skill_data2
# f1_skill = (2*p_skill*r_skill) / (p_skill +r_skill)
# #print('fl skill: ', f1_skill)
#
# p_area, r_area =  iden_area/ area_data1, iden_area/ area_data2
# f1_area = (2*p_area*r_area) / (p_area+r_area)
# #print('fl area: ', f1_area)
#
# p_diploma, r_diploma =  iden_diploma/ diploma_data1, iden_diploma/ diploma_data2
# f1_diploma = (2*p_diploma*r_diploma) / (p_diploma+r_diploma)
# #print('fl diploma: ', f1_diploma)
#
# p_major, r_major =  iden_major/ major_data1, iden_major/ major_data2
# f1_major = (2*p_major*r_major) / (p_major+r_major)
# #print('fl major: ', f1_major)
#
# p_experience, r_experience =  iden_experience/ experience_data1, iden_experience/ experience_data2
# f1_experience = (2*p_experience*r_experience) / (p_experience+r_experience)
# # print('fl experience: ', f1_experience)
"""
print('the number of label:',dict_label)
print(dict_no_idx)
print(len(data1_row_final))
print(len(data2_row_final))

# # 5. convert the list into numpy format and transpose the result 
data1_ad = np.array(data1_row_final).T
data2_ad = np.array(data2_row_final).T
# 6. transpose the result of label 
data1_use = data1_ad[1]
data2_use = data2_ad[1]

""" calculation for the number of diff in total  """

# data_label in total
total_label_num = 0
# data_label diff in total
total_label_diff = 0
# data_label_percent each abel/total >>> %total
data_label_diff_percent = {}
# data_label diff >>> each label the number of diff in total
dict_total_label_diff ={}
list_diff = []
for lab in dict_label:
    one_label_total = dict_label[lab]
    one_label_idx_diff = dict_no_idx.get(lab, 0)
    one_label_lab_diff = dict_no_lab.get(lab, 0)
    one_diff_sum = one_label_idx_diff + one_label_lab_diff
    dict_total_label_diff[lab] = one_diff_sum
    total_label_num += one_label_total
    total_label_diff += one_diff_sum
    one_labe_diff_percent = one_diff_sum/one_label_total*100
    data_label_diff_percent[lab] = one_labe_diff_percent
    tuple_tmp = (lab,one_label_total, one_diff_sum, f'{one_labe_diff_percent:.2f}%')
    list_diff.append(tuple_tmp)

print(f'data aligment; index difference: {dict_no_idx}')
print(f'data aligment; same index, differet label : {dict_no_lab}')

print('label diff ',list_diff)
print(f'sum of data : {total_label_num}, diff numbers: {total_label_diff},diff proportions: {total_label_diff/total_label_num*100:.2f}%')
labels = np.unique(data1_use)

# cohen_kappa_score
kappa = sm.cohen_kappa_score(data1_use, data2_use)
print(f'-----------cohen kappa:--------- \n {kappa}', '\n')

# confusion_matrix
cofuse = sm.confusion_matrix(data1_use, data2_use, labels = labels)
df = pd.DataFrame(cofuse, index=labels, columns=labels)
print(f'-----------confusion metrics:----------- \n {df}', '\n')

# classification report 
cr = sm.classification_report(data1_use, data2_use)
print(f'-----------classification report:----------- \n {cr}', '\n')

"""csv export """
list_output_csv = [('', 'labels', 'ident', 'diff', '%total', "F-score")]
for idx, lab in enumerate(iden_label):
    tuple_lab = (idx + 1, lab, iden_label[lab], dict_total_label_diff[lab], data_label_diff_percent[lab],
                 dict_f1[lab]*100)
    list_output_csv.append(tuple_lab)

csv_name = PATH_1.split('/')[-1].split('_')[0]
with open(f'{csv_name}.csv', 'w+', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(list_output_csv)

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.subplots_adjust(wspace=0.1)

sns.heatmap(cofuse, annot=True,
            cmap="Blues",  fmt='g', ax=ax1, linewidths=.5)

# sns.heatmap(cofuse, annot=True,
#             cmap="Blues",  fmt='g', ax=ax2, linewidths=.5)
# sns.heatmap(cofuse, annot=True,
#             cmap="Blues",  fmt='g', ax=ax3, linewidths=.5)
# sns.heatmap(cofuse, annot=True,
#             cmap="Blues",  fmt='g', ax=ax4, linewidths=.5)

ax1.set_title('IAA for the same annotator- first rounds')
ax1.set_xlabel('first round')
ax1.set_ylabel('second round')
ax1.xaxis.set_ticklabels(['areas', 'diploma', 'experience', 'knowledge','major', 'skills' ])
ax1.yaxis.set_ticklabels(['areas', 'diploma', 'experience', 'knowledge','major', 'skills' ])

# ax2.set_title('IAA for the same annotator- second rounds')
# ax2.set_xlabel('first round')
# ax2.set_ylabel('second round')
# ax2.xaxis.set_ticklabels(['areas', 'diploma', 'experience', 'knowledge','major', 'skills' ])
# ax2.yaxis.set_ticklabels(['areas', 'diploma', 'experience', 'knowledge','major', 'skills' ])

# ax3.set_title('IAA for the same annotator- third rounds')
# ax3.set_xlabel('first round')
# ax3.set_ylabel('second round')
# ax3.xaxis.set_ticklabels(['areas', 'diploma', 'experience', 'knowledge','major', 'skills' ])
# ax3.yaxis.set_ticklabels(['areas', 'diploma', 'experience', 'knowledge','major', 'skills' ])

# ax4.set_title('IAA for the same annotator- final')
# ax4.set_xlabel('first round')
# ax4.set_ylabel('second round')
# ax4.xaxis.set_ticklabels(['areas', 'diploma', 'experience', 'knowledge','major', 'skills' ])
# ax4.yaxis.set_ticklabels(['areas', 'diploma', 'experience', 'knowledge','major', 'skills' ])

fig.subplots_adjust(wspace=0.01)

plt.show()



