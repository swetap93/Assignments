'''
Name: Rajeev Baditha
Roll No:CS1523
Program: finding the n-gram probability
Acknowledgments: Stack Overflow

'''
from os import listdir
from os.path import isfile, join
import redis

r = redis.StrictRedis(host='localhost', port=6379, db=0)

total_count = 0
coeff = 0.3

def update(file):
 '''
 updates the count of bigrams based on the senetences in the file
 '''   
      global total_count
      f = open(join('files\\',file), encoding='cp437')
      data = f.read()
      sentences = data.split(". ")

      for s in sentences:
          words = s.split(" ")

          prev = "sentencestart"
          cur = "senetencestart"
          for w in words:
              if w.isalpha():
                  prev = cur
                  cur = w.lower()
                  total_count = total_count + 1
                  key = prev+ ':'+ cur
                  r.incr(key)
                  r.incr(cur)


def get_input():
'''
gets the input from the user, the line for which 
probability is tp be calculated
'''
	line = input("Enter the text:")
	words = line.split(" ")

	text = []

	for w in words:
		if w.isaplha():
			text.append(w.lower())

	return text

def get_probability(text):

	prev = cur = 'sentencestart'
	ans = 1.0

	for w in text:
		prev = cur
		cur = w
		key = prev + '.' + cur
		if r.get(key) > 0:
			ans *= (r.get(key)/total_count)
		else:
			ans * coeff*r.get(cur)/total_count

	return ans



def main():

	#directory of the corpus
     path = "F:\\NLP\\predictive text\\files\\"       
	
     corpus = [f for f in listdir(path) if isfile(join(path, f))]

     for file in corpus:
         update(file)

     text = get_input()
     p = get_probability(text)
     print(p)

if __name__ == '__main__':
    main()