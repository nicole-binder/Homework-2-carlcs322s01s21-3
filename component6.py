from chatbot.chat_bot import ChatBot

def component6():
  '''
  Build a chatbot.
  '''
  beans_the_bot = ChatBot()
  start_questions = input("Would you like to chat with Beans the Bot? (Yes/No): ")
  if start_questions.lower() == "yes":
    print("\n Please use proper capitalization and proper names when talking with Beans, Beans is just learning English. \n")
    print("Type \'quit\' at any time to end your conversation with Beans.\n")
    beans_the_bot.ask_questions()
  else:
    print("Beans is sad, but understands. See you next time!")

if __name__ == "__main__":
  component6()
