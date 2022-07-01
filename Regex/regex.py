# coding: utf-8
# ipython nbconvert test_j.ipynb --to script
import pandas as pd
import json
import re

class regexWash:
    def __init__(self):
        self.list_job = []
        self.list_one = []
        self.list_two = []
        # keyword list 
        self.list_key = ['ability', 'familiar', 'understanding', 'experience', 'knowledge', 'proficien', 'background',
                         'bachelor', 'master', 'phd', 'bs', 'ms', 'ma', 'msc', 'degree', 'diploma']
        
        # list containing unnecessary keywords 
        self.list_not_key = ['will','would','countries',"'ll","officers", 'geographical', ":","gain", 'EXPERIENCE', 'SKILLS', 'Accountability', "Disability","MAKE"]

        # Record the number of sentences 
        self.num_sen = 0
        
        # Current progress 
        self.num_now = 0
        
    def load_json(self):
        # Load Json file 
        self.data = pd.read_json(r'./IT_job_ads.json')
        # columns = data.columns.tolist()

    def wash_first(self):
        # Filter data for the first round 
        for i in range(len(self.data['description'].tolist())):
            text_row = self.data['description'].tolist()[i]
            list_tmp = []
            if type(text_row) == str:
                for t in text_row.split('\n'):
                    for key in self.list_key:
                        if key in t.lower():
                            list_tmp.append(t)
                           
                            break
            self.list_one.append(list_tmp)
            self.list_one_num =len(self.list_one)

    def wash_second(self):
         # Filter data for the second round using Regex and remove the unwanted sentences 
         #based on defined keyword lists 
        
        while len(self.list_one) > 0:
            self.num_now+=1
            print(f"Current Progress: {self.num_now}/{self.list_one_num}")
            list_here = self.list_one.pop()
            list_tmp = []
            for text_one in list_here:
                
                text_piece = text_one.split('.')
                for goal in text_piece:
                    res = self.req.findall(goal)
                    if len(res) > 0:
                        no_num = 0
                        for nkey in self.list_not_key:
                            if nkey in res[0][0]:
                                no_num += 1
                        if no_num < 1:
                            self.num_sen += 1
                            if len(res[0][0].split(' ')) > 1:
                                list_tmp.append(res[0][0])
            if len(list_tmp) > 0:
                self.list_two.append(list_tmp)


    def req_setting(self):
        # setting for Regex
        understand = "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(U|u)nderstanding\s(at|with|of|){0,4}.*"
        ability= "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(A|a)bility.*"
        # able = "^[a-z]{0,4}\s?[a-z]{0,4}\s?able.*"
        knowledge = "^[A-Za-z]{0,4}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(K|k)nowledg(e|eable)\s(at|with|of|into|to|in){0,4}.*"
        desire= "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?desire(at|with|of|into|to|in){0,4}.*"
        passion= "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?passion\s(for|at|with|of|into|to|in){0,4}.*"
        familiarity = "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(F|f)amilia(r|rity)\s(at|with|of|into|to|in){0,4}.*"
        proficiency = "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(P|p)roficien(cy|t)\s(at|with|of|into|to|in){0,4}.*"
        Efficiency = "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(E|e)fficien(cy|t)\s(at|with|of|into|to|in){0,4}.*"
        Interest = "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(I|i)teres(t|ted)\s(of|in).*"
        Experience_1 = "^[0-9]-?[0-9]?.?\s(years)?\s[A-Za-z\s]{0,20}\s?(E|e)xperience\s(at|with|of|into|to|in){0,4}.*"
        Experience_2 = "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(E|e)xperience\s(at|with|of|into|to|in){0,4}.*"
        # Experience = "[0-9]?.*?[a-z]{0,4}\sexperience(at|with|of|into){0,4}.*"
        Background = "^[A-Za-z]{0,4}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(B|b)ackground\s(in|of|with).*"
        Focus = "^[A-Za-z]{0,4}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(F|f)ocus.*"
        Approach = "^[A-Za-z]{0,4}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(A|a)pproach.*"
        Fluent = "^[A-Za-z]{0,4}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(F|f)luent.*"
        Fluency = "^[A-Za-z]{0,4}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(F|f)luency.*"
        Degree = "^[A-Za-z]{0,4}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(D|d)egree.*"
        bachelor = "^[A-Za-z]{0,3}(B|b)achelor.*"
        master = "^[A-Za-z]{0,3}(M|m)aster.*"
        # Diploma_all = "[a-z]{0,3}(ba|bs|msc|ma|ms|phd){2:3}.*"
        Diploma_all = "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?([Bb][Aa]\b|[Bb][Ss]\b|[Mm][Ss]c\b|PhD\b|PHD\b).*"
        Expertise = "^[A-Za-z]{0,4}\s?[A-Za-z]{0,20}\s?(E|e)xper(t|tise|tised)\s(at|with|of|into|to|in){0,4}.*"
        Awareness = "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?\b(A|a)wareness\s(at|with|of|into|to|in).*"
        Track_record = "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(T|t)rack record\s(at|with|of|into|to|in){0,4}.*"
        Passionate = "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(P|p)assionate\s(about).*"
        Eagerness = "^[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?[A-Za-z]{0,20}\s?(E|e)agerness\s(of|at|in|to){0,4}.*"
        # self.req = re.compile(f"({Diploma_all}|{Experience_1}|{Experience_2}|{knowledge}|{ability})",re.S)

        self.req = re.compile(f"({desire}|{passion}|{Background}|{Efficiency}|{understand}|{knowledge}|{ability}|{familiarity}|{proficiency}|{bachelor}|{master}|{Experience_1}|{Experience_2}|{Degree}|{Fluency}|{Fluent}|{Approach}|{Focus}|{Diploma_all}|{Expertise}|{Awareness}|{Track_record}|{Passionate})",re.S)

    def result_output(self):
        list_final = self.make_dataform()
        # Write in data
        with open('IT_data.jsonl', 'w+', encoding='utf-8-sig') as f:
            for pair_dict in list_final:
                target = json.dumps(pair_dict)
                f.write(target)
                f.write('\n')
                

    def make_dataform(self):
        # Convert the extracted data to a list of dictionary that matches prodigy setting 
        list_final = []
        for pair in self.list_two:
            dict_tmp = {}
            str_tmp = '\n'.join(pair)
            if str_tmp:
                dict_tmp['text'] = str_tmp
                dict_tmp['source'] = 'indeed.com'
                list_final.append(dict_tmp)
        return list_final

    def main(self):
        self.load_json()
        self.wash_first()
        self.req_setting()
        self.wash_second()
        self.result_output()
        print(f'data points in total: {self.list_one_num}')
        print(f'average sentences per data point: {self.num_sen / self.list_one_num}')


if __name__ =='__main__':
    washer = regexWash()
    import time
    start = time.time()
    washer.main()
    print(f'spent time:{time.time()-start}s')