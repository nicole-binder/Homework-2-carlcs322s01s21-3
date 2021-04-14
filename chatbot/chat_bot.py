from bot_questions import Question
import random 
import spacy
import sys
from datetime import datetime

class ChatBot:
  def __init__(self):
    self.user_NER = dict()
    self.question_list = []
    self.initalize_questions()
    self.general_apologies = ["Sorry, just got a snack. What did you say?", "Woops, had to use the restroom. What was that?", "Ugh, my mom just called me. Can you say that again?"]
    self.general_responses = ["Oh, awesome!", "Neat!", "OMG same here!!"]
    self.language_responses = dict()

  # Nicole
  def initalize_questions(self):
    file = open("bot_questions.txt").read()
    questions = file.split("&\n")
    self.question_list = []

    for q in questions:
      question_parts = q.split("\n")
      NERs = question_parts[0].split()
      response_ner = NERs[0]
      question_ner = NERs[1]
      question = question_parts[1][1:]
      response = question_parts[3][1:]
      info_in_question = ""
      info_in_response = ""
      if question_parts[1][0] == "1":
        info_in_question = True
      else:
        info_in_question = False
      if question_parts[3][0] == "1":
        info_in_response = True
      else:
        info_in_response = False 
      
      question_item = Question(question, question_ner, response, response_ner, info_in_question, info_in_response)
      self.question_list.append(question_item)

  # Nicole, Yemi, Sue
  def ask_questions(self):
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = sys.maxsize
    #nlp.select_pipes(enable=["tok2vec", "tagger"])
    response_from_user = ""
    while response_from_user != "quit":
      for question in self.question_list: 
        if (question.info_in_question and (question.question_ner in self.user_NER.keys())) or not question.info_in_question: #don't ask question if required info isn't in dict
          # ask a question
          if question.info_in_question:
            print("Beans: " + question.question.replace(question.question_ner, self.user_NER[question.question_ner]) + " :")
          else:
            print("Beans: " + question.question + " :")
          
          # get a response from user
          response_from_user = str(input(">> "))
          if response_from_user == "quit":
            break
          response_from_user_parsed = nlp(response_from_user)
          
          # give a reply
          reply = ""
          if question.info_in_response: #only care about user answer is response needs info from user
            for ent in response_from_user_parsed.ents:
              self.user_NER[ent.label_] = ent.text
              # if multiple things qualify for the single NER tag, just replace the old one with the new one 
              # if the entity's label matches the question's expected ner,
              expected_response_ner = question.response_ner
              if (str(ent.label_) == expected_response_ner) or (question.question == "Do you speak any languages other than English?"):
                self.user_NER[expected_response_ner] = ent.text
                if expected_response_ner == "DATE":
                  age = datetime.now().year - int(self.user_NER[expected_response_ner])
                  reply = question.response.replace("AGE", str(age))
                elif expected_response_ner == "TIME" and question.question == "What time is it where you are?":
                  if int(self.user_NER["TIME"].split(":")[0]) < 4:
                    reply = question.response.replace("TIME", "lunch")
                  else:
                    reply = question.response.replace("TIME", "dinner")
                elif expected_response_ner == "LANGUAGE":
                  self.make_beans_multilingual()
                  if ent.text == "Chinese":
                    language  = "Mandarin"
                  else:
                    language = ent.text
                  if language in self.language_responses.keys():
                    reply = self.language_responses[language]
                  else:
                    reply = "I don't know that lanuage yet, but I plan on learning it!"
                else:
                  reply = question.response.replace(expected_response_ner, self.user_NER[expected_response_ner])
          
            if reply != "":
              print("Beans: " + reply + "\n")
            else:
              print("Beans: " + random.choice(self.general_apologies) + "\n")
              user_reply = input()
              if user_reply == "quit":
                response_from_user = "quit"
                break
              print("Beans: " + random.choice(self.general_responses) + "\n")
          else:
            print("Beans: " + question.response + "\n")
        else:
          pass
      
      response_from_user = "quit"

    if "TIME" not in self.user_NER.keys():
      print("Beans: Thanks for talking with me! Let's talk again soon!")
    else:
      txt = "Thanks for talking with me! Have a TIME!"
      try:
        if int(self.user_NER["TIME"].split(":")[0]) > 20:
          current_time = "good night"
        else:
          current_time = "good day"
        print("Beans: " + txt.replace("TIME", current_time))
      except:
        print("Beans: Thanks for talking with me! Let's talk again soon!")
  
  # Nicole
  def make_beans_multilingual(self):
    file = open("language.txt").read()
    languages = file.split("\n")
    for line in languages:
      split_line = line.split(",")
      self.language_responses[split_line[0]] = split_line[1]


if __name__ == "__main__": # (so we'll do this in component5.py, just to be consistent with file structure)
  beans_the_bot = ChatBot()
  start_questions = input("Would you like to chat with Beans the Bot? (Yes/No): ")
  if start_questions.lower() == "yes":
    print("\n Please use proper capitalization and proper names when talking with Beans, Beans is just learning English. \n")
    print("Type \'quit\' at any time to end your conversation with Beans.\n")
    beans_the_bot.ask_questions()
  else:
    print("Beans is sad, but understands. See you next time!")



  
  