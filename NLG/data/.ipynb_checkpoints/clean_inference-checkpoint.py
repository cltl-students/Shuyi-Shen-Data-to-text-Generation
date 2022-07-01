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
df.to_csv('./test_result.csv')