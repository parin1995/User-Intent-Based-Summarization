import json
import nltk
from nltk import tokenize
import pandas as pd
from os.path import isfile, join, exists
from os import listdir, makedirs
import re

def build_train_dataset(input_data):
    df_final=pd.DataFrame()
    for i in range(5):
        state = input_data['summaries'][i]['state_name']
        txt_file = open('Dataset/StateDocuments/' + state + '.txt',"r")
        wiki_para = txt_file.read()
        sentences = tokenize.sent_tokenize(wiki_para)
        summary = input_data['summaries'][i]['sentences']
        dic = {}
        dic['sentence'] = []
        dic['target'] = []
        for sent in sentences:
            dic['sentence'].append(sent)
            if sent in summary:
                dic['target'].append(1)
            else:
                dic['target'].append(0)
        temp_df = pd.DataFrame(dic)
        df_final = df_final.append(temp_df, ignore_index=True)
    return df_final



def build_test_dataset(input_data, dest_path):
    df_final=pd.DataFrame()
    for i in range(5,8):
        state = input_data['summaries'][i]['state_name']
        txt_file = open('Dataset/StateDocuments/' + state + '.txt',"r")
        wiki_para = txt_file.read()
        sentences = tokenize.sent_tokenize(wiki_para)
        summary = input_data['summaries'][i]['sentences']
        dic = {}
        dic['sentence'] = []
        dic['target'] = []
        for sent in sentences:
            dic['sentence'].append(sent)
            if sent in summary:
                dic['target'].append(1)
            else:
                dic['target'].append(0)
        temp_df = pd.DataFrame(dic)
        temp_df.to_csv(dest_path + '/test/test_set_'+state+'.csv',index=False)
    return df_final


intent_list=[]
my_path = "../../../Dataset/"
for file in listdir("../../../Dataset"):
    dest_path = "../../../data/"
    file_name = join(my_path,file)
    if isfile(file_name) and file.startswith("user"):
        f = open(join(my_path,file))
        data = json.load(f)
        intent_list.append(data['intent'])
        dest_path = dest_path + "_".join(re.sub(r'[^\w\s]','',data['intent']).split(' ')) + "/" + re.sub(r'\.txt', '',file)
        makedirs(dest_path + "/test")
        df_train = build_train_dataset(data)
        df_train.to_csv(dest_path + '/train_set.csv',index=False)
        build_test_dataset(data, dest_path)