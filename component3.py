from MarcovModel import MarcovModel
import spacy
import sys
import random

'''
Experimentation with our updated model!
'''

# Sue, Yemi, Nicole
def component3a():
  print("-----working on component3a (Check 'results' folder)-----")
  f = open("results/component3a.txt", "a")

  # Load the standard English model suite
  nlp = spacy.load("en_core_web_sm")
  # Update the model suite to allow for a long corpus
  nlp.max_length = sys.maxsize

  # Prepare a corpus
  corpus = open("chatbot/language.txt").read().split("\n")

  length = len(corpus)
  sentences = [corpus[random.randint(0, length - 1)], corpus[random.randint(0, length - 1)], corpus[random.randint(0, length - 1)]]

  # POS-tag the corpus
  for i, sentence in enumerate(sentences):
    f.write(f"-----Sentence {i} of Chat_Bot/language.txt-----\n")
    with nlp.select_pipes(enable=["tok2vec", "tagger"]):
      tagged_tokens = nlp(sentence)
    # Print out the tagged tokens
    for token in tagged_tokens:
      f.write(f"{token}/{token.tag_}\n") 
  
  # Prepare a corpus
  corpus2 = open("corpora/donald_trump_collected_tweets.txt").read().split("\n")

  length = len(corpus2)
  sentences = [corpus2[random.randint(0, length - 1)], corpus2[random.randint(0, length - 1)], corpus2[random.randint(0, length - 1)]]
  
  # POS-tag the corpus
  for i, sentence in enumerate(sentences):
    f.write(f"-----Sentence {i} of Donald Trump-----\n")
    with nlp.select_pipes(enable=["tok2vec", "tagger"]):
      tagged_tokens = nlp(sentence)
    # Print out the tagged tokens
    for token in tagged_tokens:
      f.write(f"{token}/{token.tag_}\n") 

# Sue, Yemi
def component3b():
  print("-----working on component3b (Check 'results' folder)-----")
  f = open("results/component3b.txt", "a")
  f.write("\n-----POS ENGAGED-----\n\n")
  for order in range(1,8):
    model = MarcovModel("alexander_dumas_collected_works.txt", level = "word", order = order, pos = True)
    f.write(f"-----Results for {model.corpus_filename}, Order: {order} -----\n")
    f.write(model.generate(30) + "\n")

# Sue, Yemi
def component3c():
  print("-----working on component3c (Check 'results' folder)-----")
  f = open("results/component3c.txt", "a")
  f.write("-----without the POS-----")
  model = MarcovModel("alexander_dumas_collected_works.txt", level = "word", order = 5, pos = False)
  f.write(model.generate(30) + "\n")
  f.write("-----with the POS-----")
  model2 = MarcovModel("alexander_dumas_collected_works.txt", level = "word", order = 5, pos = True)
  f.write(model2.generate(30) + "\n")

def component3():
  #component3a()
  #component3b()
  component3c()

if __name__ == "__main__":
  component3()