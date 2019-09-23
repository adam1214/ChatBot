import jieba
import numpy as np
from scipy import spatial
from gensim.models import word2vec

def avg_feature_vector(words, model, num_features, index2word_set):
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words> 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def main():
	ans = []
	model = word2vec.Word2Vec.load("word2vec_50.bin")
	index2word_set = set(model.wv.index2word)
	fp = open('project_question_file.txt', "r",encoding="utf-8")
	line = fp.readline()
	ttt=0
	while line:
		score_list = []

		text=line.strip().split('\t')[0]
		option1=line.strip().split('\t')[1]
		option1=option1[3:]
		option2=line.strip().split('\t')[2]
		option2=option2[3:]
		option3=line.strip().split('\t')[3]
		option3=option3[3:]
		option4=line.strip().split('\t')[4]
		option4=option4[3:]   
		#print(text)
		#print(option1)
		#print(option2)
		#print(option3)
		#print(option4)
		q_v = avg_feature_vector(jieba.cut(text.strip()), model=model, num_features=50, index2word_set=index2word_set)
		a1_v = avg_feature_vector(jieba.cut(option1.strip()), model=model, num_features=50, index2word_set=index2word_set)
		a2_v = avg_feature_vector(jieba.cut(option2.strip()), model=model, num_features=50, index2word_set=index2word_set)
		a3_v = avg_feature_vector(jieba.cut(option3.strip()), model=model, num_features=50, index2word_set=index2word_set)
		a4_v = avg_feature_vector(jieba.cut(option4.strip()), model=model, num_features=50, index2word_set=index2word_set)
		score_list.append(1 - spatial.distance.cosine(q_v, a1_v))
		score_list.append(1 - spatial.distance.cosine(q_v, a2_v))
		score_list.append(1 - spatial.distance.cosine(q_v, a3_v))
		score_list.append(1 - spatial.distance.cosine(q_v, a4_v))
		for i in range(len(score_list)):
			if np.isnan(score_list[i]) == True:
				score_list[i] = 0
		#print(score_list)
		for i in range(len(score_list)):
			if score_list[i] == max(score_list):
				ans.append(i+1)
				break
		line=fp.readline()
		'''
		ttt=ttt+1
		if ttt==3:
			break
		'''

	with open("F64051114.csv", 'w', encoding='utf-8') as output :
		for i in range(len(ans)) :
			output.write('[' + str(ans[i]) + ']' + '\n')

if __name__ == "__main__":
    main()