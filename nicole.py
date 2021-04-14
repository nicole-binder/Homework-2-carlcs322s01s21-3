import nltk, re, pprint
import random
from nltk import word_tokenize, regexp_tokenize
import statistics
import spacy
import sys

class MarcovModel:
  def __init__(self, corpus_filename, level, order, pos=bool(False), hybrid=bool(False)):
    '''
    Creates a MarcovModel object.

    Args:
      corpus_filename: 
        string representing the path to a text file containing sample sentences
      level: 
        "character" or "word" (which mode to train the model in)
      order: 
        integer defining the model's order 
    '''
    self.corpus_filename = corpus_filename
    self.corpus, self.testset = self._load_corpus(corpus_filename)
    self.tokens = []
    self.pos = pos
    self.hybrid = hybrid
    if self.pos:
      self.level = "word"
    else:
      self.level = level
    self.order = order
    self.token_to_token_transitions = dict()
    self.pos_to_pos_transitions = dict()
    self.pos_to_token_transitions = dict()
    self.train()

  # Sue 
  def train(self):
    '''
    Populates 'transitions' dictionary of n-grams, where n is the given order of the model. In addition, calculates authorship_estimator (aka mean and stdev of likelihoods for the second half of the model).
    '''
    split_corpus = self.corpus.split("\n")

    # If the corpus is william shakespeare collected works, just reduce the size of the corpus for now (for future, make the code more efficient by serializing)
    if self.corpus_filename == "william_shakespeare_collected_works.txt":
      self.corpus = "\n".join(split_corpus[:len(split_corpus) // 3])
    else:
      self.corpus = "\n".join(split_corpus[:(len(split_corpus) * 8) // 10])
   
    corpus_to_be_used_for_estimation = split_corpus[((len(split_corpus) * 8) // 10) + 1:]

    '''
    POPULATING (appropriate) TRANSITIONS DICTIONARY portion
    '''
    if self.hybrid:
      self._hybrid_train()
    elif self.pos:
      self._pos_train()
    else:
      self._word_train()

    '''
    CALCULATING AUTHORSHIP ESTIMATOR portion
    '''
    # self.authorship_estimator = self._caculate_authorship_estimator(corpus_to_be_used_for_estimation)
  
  def _hybrid_train(self):
    # train using pos method
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = len(self.corpus)
    with nlp.select_pipes(enable=["tok2vec", "tagger"]):
      self.tokens = nlp(self.corpus)
    self._pos_train()

    # train using word method
    self.tokens = self._tokenize(self.corpus) # tokenize the corpus
    self._word_train()

  def _word_train(self):
    '''
    Trains the model based on token-token transitions (original)
    aka. populates token_to_token_transitions.
    '''
    self.tokens = self._tokenize(self.corpus)

    # puntuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~\t'''
    puntuations = '''\t'''
    # count how many times each token appears when a given n-gram in a nested list
    num = 0 # position of the first word in the n-gram in the corpus
    for token in self.tokens:
      # puntuation does not go into the n-gram
      if token not in puntuations:
        gram = [token] # a list of tokens) that go into the ngram
        cur_order = 1
        word_num = 1 # the length of the n-gram
        # create valid n-gram
        while cur_order < self.order:
          # make sure it is not out of index and the n-gram doesn't have puntuations
          if num+cur_order < len(self.tokens) and self.tokens[num+cur_order] not in puntuations:
            # gram = gram + " " + self.tokens)[num+cur_order]
            gram.append(self.tokens[num+cur_order])
            word_num += 1
          cur_order += 1
        
        gram = self._construct_text(gram).strip()
        
        # make sure n-gram do not contain puntuations and there is at least one more token in the corpus
        if word_num == self.order and num < len(self.tokens)-self.order:
          value = self.tokens[num+self.order] 
          # puntuation does not count as token
          if value not in puntuations:
            # create the dictionary with values in nested lists
            if gram in self.token_to_token_transitions:
              not_added = True
              for item in self.token_to_token_transitions[gram]: # "the" : [["fox", 3], ["bear", 5]]
                if item[0] == value:
                  item[1] += 1
                  not_added = False
              if not_added:
                self.token_to_token_transitions[gram].append([value,1])
            else:
              self.token_to_token_transitions[gram] = [[value,1]]   
      num += 1

    # calculate probablity and convert list to tuple
    
    all_keys = self.token_to_token_transitions.keys()
    for key in all_keys:
      total_appearance = 0
      specific_values = self.token_to_token_transitions[key]
      # calculate the total appearances
      # "the" : [["fox", 3], ["bear", 5]]
      for value in specific_values:
        total_appearance = total_appearance + value[1]
      # calculate the frequency_range for each token and convert the list to tuple
      range_num = 0 # start of a new range
      for value in specific_values:
        value[1] = (range_num, range_num+value[1]/total_appearance)
        range_num = value[1][1] # update lower bound
        # convert the nested list into a tuple
      token_num = 0
      while token_num < len(specific_values):
        specific_values[token_num] = tuple(specific_values[token_num])
        token_num += 1
  
  # Maanya and Nicole
  def _pos_train(self):
    '''
    Trains the model based on pos_tag - pos_tag transitions 
    aka. populates the transitions dictionary based on this scheme.
    '''

    '''
    POPULATING POS_TO_POS DICTIONARY portion
    '''
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = sys.maxsize
    with nlp.select_pipes(enable=["tok2vec", "tagger"]):
      self.tokens = nlp(self.corpus)
        
    for count in range(len(self.tokens)-self.order+1):
      n_gram = self.tokens[count:count+self.order] #exclusive i.e. l=[1,2,3], l[0:2] = [1,2]
      n_gram_tags = []
      #syntax tags will make up keys, so create list then convert to tuple for each key
      for token in n_gram:
        if "\n" in token.text:
          n_gram_tags.append("NLN")
        else:
          n_gram_tags.append(token.tag_)
      if count != len(self.tokens)-self.order:
        syntax_value = self.tokens[count+self.order].tag_
      else:
        #edge case: n_gram is at the end of the text
        syntax_value = None
      #NOTE: keys can't be mutable, so key is tuple instead of list
      n_gram_tags_tuple = tuple(n_gram_tags)
      if n_gram_tags_tuple not in self.pos_to_pos_transitions.keys():
        if syntax_value == None:
          #NOTE: may need to change assigned value because it may throw some errors depending on generate()
          self.pos_to_pos_transitions[n_gram_tags_tuple] = None
        else:
          self.pos_to_pos_transitions[n_gram_tags_tuple] = [[syntax_value,1]]
      else:
        for value in self.pos_to_pos_transitions[n_gram_tags_tuple]:
          already_a_value = False
          #update dicitonary
          if syntax_value == value[0]:
            value[1]+=1 # increment count
            already_a_value = True
        # if not already a value
        if already_a_value == False:
          self.pos_to_pos_transitions[n_gram_tags_tuple].append([syntax_value,1])
    #update dictionary so values are tuples and percents not numbers
    for key in self.pos_to_pos_transitions.keys():
      value_list = self.pos_to_pos_transitions[key]
      #ex. value_list = [['RB',1],['VBZ',2]]
      total_tag_count = 0
      if value_list == None:
        pass
      else:
        # add up the total count
        for tag in value_list:
          #ex. tag = ['RB',1]
          total_tag_count += tag[1]

        first_value = True
        previous_range = ()
        tuple_to_replace_list = []

        # calculate percentage
        for tag in value_list:
          percent = tag[1]/total_tag_count
          if first_value == True:
            percent_range = tuple((0.0,percent))
            first_value = False
            previous_range = percent_range
          else:
            percent_range = tuple((float(previous_range[1]),float(previous_range[1])+percent))
            previous_range = percent_range

          tuple_to_replace_list.append(tuple((tag[0],percent_range)))
        self.pos_to_pos_transitions[key] = tuple(tuple_to_replace_list)

    '''
    POPULATING POS_TO_TOKEN DICTIONARY portion
    '''
    self._map_pos_to_tokens()

  # Maanya
  def _map_pos_to_tokens(self):
    '''
    Requirements:
      Create a new subroutine in your training procedure that produces a dictionary mapping POS tags to probability ranges over the tokens) to which those tags were attached when the corpus was tagged. For instance, if the tag NP was attached to the token Northfield once
      and to the token Minnesota thrice, the value in this dictionary for the key NP would be
      [(“Northfield”, (0.0, 0.25)), (“Minnesota”, (0.25, 1.0))].

      populate self.pos_to_token_transitions dictionary
    '''
    #["UH": [("Hello", 1)], ".": [(".", 1)]]

    pos_token_list = self.generate_pos_tags(self.corpus)
    for pos_token_pair in pos_token_list:
        if pos_token_pair[0] not in self.pos_to_token_transitions:
          self.pos_to_token_transitions[pos_token_pair[0]] = [(pos_token_pair[1], 1)]
        else:
          present = False
            # token_value = ("Hello", 1)
            # self.pos_to_token_transitions[pos_token_pair[0]] = [("Hello", 1), ("Hey", 1)]
          for token_value in self.pos_to_token_transitions[pos_token_pair[0]]:
              if pos_token_pair[1] == token_value[0]:
                  present = True
                  token_value = list(token_value)
                  token_value[1] += 1
                  token_value = tuple(token_value)
          if present == False:
              self.pos_to_token_transitions[pos_token_pair[0]].append((pos_token_pair[1], 1))

    for key in self.pos_to_token_transitions.keys():
      value_list = self.pos_to_token_transitions[key]
      #ex. value_list = [(“Northfield”, 1), (“Minnesota”, 3)]
      total_tag_count = 0
      if value_list == None:
        pass
      else:
        # add up the total count
        for tag in value_list:
          #ex. tag = (“Northfield”, 1)
          total_tag_count += tag[1]

        first_value = True
        previous_range = ()
        tuple_to_replace_list = []

        # calculate percentage
        for tag in value_list:
          percent = tag[1]/total_tag_count
          if first_value == True:
            percent_range = tuple((0.0,percent))
            first_value = False
            previous_range = percent_range
          else:
            percent_range = tuple((float(previous_range[1]),float(previous_range[1])+percent))
            previous_range = percent_range

          tuple_to_replace_list.append(tuple((tag[0],percent_range)))
          self.pos_to_token_transitions[key] = tuple_to_replace_list
          # print(self.pos_to_token_transitions)

  # Maanya
  @staticmethod
  def generate_pos_tags(corpus):
    # Load the standard English model suite
    nlp = spacy.load("en_core_web_sm")
    # corpus = re.sub('\n', 'newline', corpus)
    
    # Update the model suite to allow for a long corpus
    nlp.max_length = len(corpus)
    # POS-tag the corpus
    with nlp.select_pipes(enable=["tok2vec", "tagger"]):
        tagged_tokens = nlp(corpus)
    # Print out the tagged tokens
    n_gram_tags = []
    for token in tagged_tokens:
        if "\n" in token.text:
            token.tag_ = 'NLN'
        # print(f"{token}/{token.tag_}") 
        n_gram_tags.append((token.tag_, token.text))
    return n_gram_tags
     

  #Maanya and Yemi
  @staticmethod
  def _load_corpus(corpus_filename):
    '''
    Returns the contents of a corpus loaded from a corpus file.

    Credit to James (Took from Comp Med HW file)

    Args:
      corpus_filename:
        The filename for the corpus that's to be loaded.

    Returns:
      A single string

    Raises:
      IOError:
        There is no corpus file with the given name in the 'corpora' folder.
    '''
    corpus_text = open(f"corpora/{corpus_filename}").read()
    return corpus_text[:(len(corpus_text)) // 10], corpus_text[:((len(corpus_text) * 8) // 10) + 1]

  def generate(self, length, prompt="\n"):
    '''
    Generates text based on the statistical language model.
    '''
    if self.hybrid:
      return self._hybrid_generate(length, prompt)
    elif self.pos:
      return self.adapted_pos_generate(length, prompt)
    else:
      return self._original_generate(length, prompt)
  
  # Nicole
  def adapted_pos_generate(self, length, prompt="\n"):
    '''
    Generates a text of 'length' tokens which begins with 'prompt' token if given one.

    Args:
      length: 
        length of the text to be generated
      prompt: 
        starting tokens) (default: "\n")
    
    Returns:
      A string containing the generated text
    '''
    gen_prompt_tags = self.generate_pos_tags(prompt)
    tagged_prompt = []
    for tag in gen_prompt_tags:
      tagged_prompt.append(tag[0])
    tagged_prompt = tuple(tagged_prompt)

    final_tags = list(tagged_prompt)
    n_gram = ""

    length_of_prompt = len(tagged_prompt)
    
    #prompt does not have a complete n-gram
    if length_of_prompt < self.order:
      n_gram, final_tags = self._find_n_gram_pos(tagged_prompt, final_tags, length_of_prompt, length)
    else: #prompt is longer than or equal to one n-gram, reduce/keep the same
      n_gram = tuple(tagged_prompt[length_of_prompt - self.order:])
      #check if n_gram is in our dictionary
      if n_gram not in self.pos_to_pos_transitions.keys():
        #find key containing prompt
        n_gram, final_tags = self._find_n_gram_pos(tagged_prompt, final_tags, length_of_prompt, length)

    while len(final_tags) < length:
      values = self.pos_to_pos_transitions.get(n_gram)
      if values is None:
        n_gram, final_tags = self._find_n_gram_pos(n_gram, final_tags, length_of_prompt, length)
        values = self.pos_to_pos_transitions.get(tuple(n_gram))
      random_num = random.random()
      # ["the": (("end", (0,.5)), ("fox", (.5,1)))]
      for t in values:
        probability_range = t[1]
        if random_num > probability_range[0] and random_num <= probability_range[1]:
          add_tag = t[0]
          final_tags.append(add_tag)
      #get last n token of generated text
      n_gram = tuple(final_tags[len(final_tags) - self.order:])
      
    return self._generate_text_from_tags(tagged_prompt, prompt, final_tags)
  
  def _find_n_gram_pos(self, tagged_prompt, final_tags, length_of_prompt, length):

    keys = self.pos_to_pos_transitions.keys()
    n_gram = ""
    prompt = tagged_prompt
    #find n-gram CONTAINING the prompt or shortened prompt
    x = 0 #variable to decrement token length of prompt (ex. "the brown" not found, then check if some key begins with "brown")
    while n_gram == "":
      for k in keys:
        #see if prompt is the start of key k
        shortened_key = k[0:length_of_prompt]
        #store to add to gen_text when valid key is found
        rest_of_key = list(k[length_of_prompt:])
        if shortened_key == prompt:
          n_gram = k
          if len(rest_of_key) > 1:
            for i in rest_of_key:
              final_tags.append(i)
          else:
            final_tags.append(rest_of_key)
          #add rest of key to gen_text, ex. key = "brown fox jumps", prompt = "the quick brown", gen_text = "the quick brown fox jumps", n_gram = brown fox jumps
          break #valid dictionary key found
      #if prompt not contained in any n-grams in dictionary, remove first token, check again
      x+=1
      prompt = tuple(tagged_prompt[x:])
      length_of_prompt = len(prompt)
      #if no words in the prompt in any dictionary key, choose a random key to start text generation
      if x == len(tagged_prompt):
        #note: random key not appended to gen_text
        entry_list = list(self.pos_to_pos_transitions.keys())
        n_gram = random.choice(entry_list)
    if len(final_tags) > length:
      final_tags = final_tags[0:self.order]
    return n_gram, final_tags
  
  def _generate_text_from_tags(self, tagged_prompt, prompt, final_tags):
    gen_text_list = []
    for tag in final_tags[len(tagged_prompt):]:
      values = self.pos_to_token_transitions[tag]
      random_num = random.random()
      for v in values:
        probability_range = v[1]
        if random_num > probability_range[0] and random_num <= probability_range[1]:
          gen_text_list.append(v[0])
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~\t\n'''
    last_char_punctuation = False
    for char in prompt:
      if char in punctuations:
        last_char_punctuation = True
    if last_char_punctuation == True:
      beginning = prompt
    else:
      beginning = prompt + " "
    return beginning + self._construct_text(gen_text_list)

  def _construct_text(self, tokens, first_token=0):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~\t\n'''
    text = ""
    if self.level == "character":
      for token in tokens:
        text+=token
    else:
      for token in tokens:
        if first_token == 1:
          text += token
          first_token+=1
        elif token in punctuations:
          text += token
        elif (self.pos == True or self.hybrid == True) and " " in token:
          text += " "
        else:
          text += " " + token
    return text
      

if __name__ == "__main__":
    model = MarcovModel(corpus_filename = "alexander_dumas_collected_works.txt", level = "word", order = 2, pos = True)
    print(model.generate(20, "hello this is me."))