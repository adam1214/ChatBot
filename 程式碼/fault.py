# -*- coding: utf-8 -*-
#!/usr/bin/python
from gensim.models import word2vec
from gensim.test.utils import common_texts
import jieba

model="word2vec_50.bin"
model_w2v=word2vec.Word2Vec.load(model)
questions = []
answers = []
with open("Gossiping-QA-Dataset.txt",encoding='utf-8')as f:
     for line in f:
          questions.append(line.strip().split('\t')[0])
          B = line.split('\t')[1].strip()
          answers.append(B)

## Open inputfile
fp = open('PPT_test_corpus.txt', "r",encoding="utf-8")
line = fp.readline()

output = open("F64051114.csv", "w")
## 用 while 逐行讀取檔案內容，直至檔案結尾
#ttt=0
while line:
    text=line.strip().split('\t')[0]
     
    option1=line.strip().split('\t')[1]
    option1=option1[3:]
    option2=line.strip().split('\t')[2]
    option2=option2[3:]
    option3=line.strip().split('\t')[3]
    option3=option3[3:]
    option4=line.strip().split('\t')[4]
    option4=option4[3:]

    index=0
    com=0
    answer_same_or_not=0
    for candidate in questions:
        if text == candidate:
            answer_same_or_not=1
            if answers[index] == option1:
                output.write("[1]\n")
                com=1
            elif answers[index] == option2:
                output.write("[2]\n")
                com=1
            elif answers[index] == option3:
                output.write("[3]\n")
                com=1
            elif answers[index] == option4:
                output.write("[4]\n")
                com=1
            break
        index=index+1
    
    if answer_same_or_not==0: #沒有找到一模一樣的問題，需做Gossiping-QA-Dataset.txt內所有問題與該問題之相似度分析
        #這邊重點是要找到正確的index問題，給下面的if條件輸出正確選項
        print("沒有找到一模一樣的，需做問題相似度分析\n")

        text=line.strip().split('\t')[0]
        words=list(jieba.cut(text.strip()))
        word=[]

        for w in words:
            if w not in model_w2v.wv.vocab:
                print("input word %s not in dict. skip this turn"% w)
            else:
                word.append(w)

        index=0
        b=0
        res=[]
        for candidate in questions:
            candidate_seg=list(jieba.cut(candidate.strip(),cut_all=False))
            temp=[]
            for c in candidate_seg:
                if c  in model_w2v.wv.vocab:
                    #print("Candidate word %s not in dict. skip this turn"% c)
                    temp.append(c)
            if len(temp)!=0 :          
                score=model_w2v.n_similarity(word,temp)
            else:
                score=0
            if score >= 0.99:   #score若大於0.99則直接輸出index值
                b=1
                break
            resultInfo={'id': index,"score": score,"text": answers[index]}
            res.append(resultInfo)
            index += 1

        if b!=1:  #若沒有score大於等於0.99的，需要做score排序並找出score最大之問題之index
            res.sort(key=lambda x:x['score'],reverse=True)
            index=res[0]['id']

    if com==0: #沒有找到一模一樣的答案，需做option1234與answers[index]之間的相似度計算
        #沒有找到一模一樣的問題，肯定也還未印出正確選項
        if answer_same_or_not==0:
            print(questions[index])
        print(answers[index])
        print(option1)
        print(option2)
        print(option3)
        print(option4)
        res=[]

        words=list(jieba.cut(answers[index].strip()))
        word=[]

        for w in words:
            if w not in model_w2v.wv.vocab:
                #print("input word %s not in dict. skip this turn"% w)
                print("")
            else:
                word.append(w)
        
        candidate_seg=list(jieba.cut(option1.strip(),cut_all=False))
        temp=[]
        for c in candidate_seg:
            if c  in model_w2v.wv.vocab:
                #print("Candidate word %s not in dict. skip this turn"% c)
                temp.append(c)
            if len(temp)!=0 and len(word)!=0:          
                score=model_w2v.n_similarity(word,temp)
            else:
                score=0
        resultInfo={'id': 1,"score": score,"text": option1}
        res.append(resultInfo)

        candidate_seg=list(jieba.cut(option2.strip(),cut_all=False))
        temp=[]
        for c in candidate_seg:
            if c  in model_w2v.wv.vocab:
                #print("Candidate word %s not in dict. skip this turn"% c)
                temp.append(c)
            if len(temp)!=0 and len(word)!=0:          
                score=model_w2v.n_similarity(word,temp)
            else:
                score=0
        resultInfo={'id': 2,"score": score,"text": option2}
        res.append(resultInfo)

        candidate_seg=list(jieba.cut(option3.strip(),cut_all=False))
        temp=[]
        for c in candidate_seg:
            if c  in model_w2v.wv.vocab:
                #print("Candidate word %s not in dict. skip this turn"% c)
                temp.append(c)
            if len(temp)!=0 and len(word)!=0:          
                score=model_w2v.n_similarity(word,temp)
            else:
                score=0
        resultInfo={'id': 3,"score": score,"text": option3}
        res.append(resultInfo)

        candidate_seg=list(jieba.cut(option4.strip(),cut_all=False))
        temp=[]
        for c in candidate_seg:
            if c  in model_w2v.wv.vocab:
                #print("Candidate word %s not in dict. skip this turn"% c)
                temp.append(c)
            if len(temp)!=0 and len(word)!=0:          
                score=model_w2v.n_similarity(word,temp)
            else:
                score=0
        resultInfo={'id': 4,"score": score,"text": option4}
        res.append(resultInfo)

        res.sort(key=lambda x:x['score'],reverse=True)
        print(res)
        if res[0]['id']==1:
            output.write("[1]\n")
        elif res[0]['id']==2:
            output.write("[2]\n")
        elif res[0]['id']==3:
            output.write("[3]\n")
        elif res[0]['id']==4:
            output.write("[4]\n")
    line = fp.readline()
    '''
    ttt=ttt+1
    if ttt==3:
        break
    '''
    

fp.close()
output.close()