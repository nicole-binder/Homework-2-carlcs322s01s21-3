import spacy

# Sue
def component5():
  print("-----working on component5 (Check 'results' folder)-----")
  nlp = spacy.load("en_core_web_sm")
  doc = nlp(open("news.txt").read())
  f = open("results/newsentity.txt", "a")
  for ent in doc.ents:
    f.write(ent.text)
    f.write(" ")
    f.write(ent.label_)
    f.write(" ")
    f.write(spacy.explain(ent.label_))
    f.write("\n")
  doc2 = nlp(open("news.txt").read().lower())
  f.write("\n\n")
  for ent in doc2.ents:
    f.write(ent.text)
    f.write(" ")
    f.write(ent.label_)
    f.write(" ")
    f.write(spacy.explain(ent.label_))
    f.write("\n")
  f.close()

if __name__ == "__main__":
  component5()

#https://edition.cnn.com/interactive/2021/04/uk/special-relationship-prince-philip-and-queen-romance-intl-cmd/