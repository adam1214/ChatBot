import jieba
import logging
from langconv import Converter

logging.getLogger().setLevel(logging.INFO)
#stopword_set=set()
output = open('Gossiping-QA-Dataset_seg.txt','w',encoding='utf-8')
with open('Gossiping-QA-Dataset.txt','r',encoding='utf-8') as content:
    for texts_num, line in enumerate(content):
        line = line.strip('\n')
        line = Converter('zh-hant').convert(line)
        line = line
        words = jieba.cut(line,cut_all=False)
        for word in words:
            #if word not in stopword_set:
                output.write(word+' ')
        output.write('\n')
        if(texts_num + 1) % 10000 == 0:
            logging.info("已完成前%d行的斷詞" % (texts_num + 1))
output.close()