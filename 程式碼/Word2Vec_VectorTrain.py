from gensim.models import word2vec
from gensim.test.utils import common_texts
sentences = word2vec.LineSentence('Gossiping-QA-Dataset_seg.txt')

model = word2vec.Word2Vec( sentences , size = 50 , window = 5, workers= 9, sg = 0 , min_count = 5)
model.save ('word2vec_50.bin')
#model = word2vec.Word2Vec.load('word2vec_50.bin')
#vec_obj = model.wv ["冰沙"]