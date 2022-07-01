import pandas as pd

df = pd.read_csv('./inference_result.csv')

text = df['text'].tolist()
prediction =  df['prediction'].tolist()

x = []
y = []
for input_text, target_text in zip(prediction,text):
    
    if len(str(input_text).split('|')) == 1:
        continue
    else:
        x.append(input_text)
        y.append(target_text)
        
data = {'prefix':"webNLG",
       'input_text':x,
       'target_text':y}
df = pd.DataFrame(data)
df.to_csv('./test_result_revised.csv')

###############################################
df = pd.read_csv('./test_result.csv')

text = df['target_text'].tolist()
prediction =  df['input_text'].tolist()
truth = df['truth'].tolist()
x = []
y = []
z = []
for input_text, target_text, gold in zip(prediction,text, truth):
    
    if len(str(input_text).split('|')) == 1:
        continue
    else:
        x.append(input_text)
        y.append(target_text)
        z.append(truth)
        
data = {'prefix':"webNLG",
       'input_text':x,
       'target_text':y, 
       'truth':z}
df = pd.DataFrame(data)

df.to_csv('./test_result.csv')


###############################################

# split the inference data into train, dev and test with the proportion of 8:1:1
data_file = "./test_result_revised.csv"
train_df=pd.read_csv(data_file)
# train_df=train_df.iloc[  :50,:]
length = len(train_df)
train =train_df.iloc[  :int(length*0.8),:]
validate =train_df.iloc[int(length*0.8)  :int(length*0.9),:]
test = train_df.iloc[ int(length*0.9) :,:]

print('the data we"re going to split is：', len(train_df))
print('the length of train split：', len(train))
print('the length of dev split：', len(validate))
print('the length of test split：', len(test))

###############################################
train.to_csv("./train.csv")
validate.to_csv("./validate.csv")
test.to_csv("./test.csv")