class Question:
  def __init__(self, question, question_ner, response, response_ner, info_in_question, info_in_response):
    self.question = question
    self.response = response
    self.question_ner = question_ner
    self.response_ner = response_ner
    self.info_in_question = info_in_question
    self.info_in_response = info_in_response
