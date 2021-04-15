from MarcovModel import MarcovModel

def component4():
  # Basic idea for the hybrid generation is to prioritize meaning over syntax. In other words, we generate a text using the original generate method. Then you correct any jarring grammar mistakes with the pos generation scheme.
  print("-----working on component4 (Check 'results' folder)-----")
  f = open("results/component4.txt", "a")
  model = MarcovModel("arthur_conan_doyle_collected_works.txt", level = "word", order = 3, hybrid = True)
  for i in range(3):
    f.write("-----Hybrid Generate Engaged, Level: word, Order: 3-----\n")
    f.write(model.generate(50) + "\n")
  
if __name__ == "__main__":
  component4()