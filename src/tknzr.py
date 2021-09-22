import re
import contractions as con
from nltk.tokenize import word_tokenize
from nltk import download 
from spellchecker import SpellChecker
from nltk import TweetTokenizer

download('punkt')

def make_spellchecker():
    """
    Initialise spellchecker object with a dictionary based on the words in the pre-trained embeddings
    """
    spell = SpellChecker(language=None, local_dictionary='../../../data/external/custom_spell.json')
    spell.word_frequency.add("_possessivetag_")
    return spell 

def make_tokenizer():
    """
    Initialise spellchecker object with a dictionary based on the words in the pre-trained embeddings
    """
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    return tokenizer 

def tokenize1(string, spellcheck, tokenizer, spell):
    """
    takes string input and tokenizes into a list of strings. 
    #runningtime is 0(n)*6

    Examples
    >>> tokenize1('MC Hammer (He is credited as Hammer) portrays a druglord...so you do a math. Bruce Payne', False, make_tokenizer(), make_spellchecker())
    ['mc', 'hammer', '(', 'he', 'is', 'credited', 'as', 'hammer', ')', 'portrays', 'a', 'druglord', '...', 'so', 'you', 'do', 'a', 'math', '.', 'bruce', 'payne']
    >>> tokenize1('I am soooooo baaaad!!!!')
    ['i', 'am', 'sooo', 'baaad', '!', '!', '!']
    >>> tokenize1('diller/vagene')
    ['diller', '/', 'vagene']
    >>> tokenize1('what now..little boy.....')
    ['what', 'now', '...', 'little', 'boy', '...']
    >>> tokenize1("Film's have 'Act Two' which sucks.")
    ['film', "'s", 'have', "'", 'act', 'two', "'", 'which', 'sucks', '.']
    """

    string = re.sub('\&quot;', '', string)
    string = re.sub("\'s ", " _possessivetag_ " , string)
    string = re.sub('(?<=[.,])(?=[^\W])', ' ' , string)
    string = re.sub("\--", ' -- ' , string)
    #string = re.sub('(?P<rep>.)(?P=rep){3,}', '\g<rep>\g<rep>\g<rep>', string)
    string = re.sub("\d+|\+"," \g<0> ", string)
    
    string = con.fix(string)

    tokens = tokenizer.tokenize(string)


    if spellcheck == True:
        # find those words that may be misspelled
        tokens = [spell.correction(token) if len(token) > 5 else token for token in tokens]
    
    tokens = ["'s" if token == "_possessivetag_" else token for token in tokens]


    return tokens

def tokenize2(string):
    """
    takes string input and tokenizes into a list of strings. 
    #runningtime is 0(n)*6

    Examples
    >>> tokenize2('MC Hammer (He is credited as Hammer) portrays a druglord...so you do a math. Bruce Payne')
    ['mc', 'hammer', '(', 'he', 'is', 'credited', 'as', 'hammer', ')', 'portrays', 'a', 'druglord', '...', 'so', 'you', 'do', 'a', 'math', '.', 'bruce', 'payne']
    >>> tokenize2('I am soooooo baaaad!!!!')
    ['i', 'am', 'sooo', 'baaad', '!', '!', '!']
    >>> tokenize2('diller/vagene')
    ['diller', '/', 'vagene']
    >>> tokenize2('what now..little boy.....')
    ['what', 'now', '...', 'little', 'boy', '...']
    >>> tokenize2("Film's have 'Act Two' which sucks.")
    ['film', "'s", 'have', "'", 'act', 'two', "'", 'which', 'sucks', '.']
    """

    string = string.lower()
    string = re.sub('\/', ' / ', string)
    string = re.sub('_', ' ', string)
    string = re.sub('-', ' - ', string)
    string = re.sub("[\s]'\\b", " ' ", string)
    string = re.sub('\&quot;', '', string)
    string = re.sub('\.(?P<name>([^\s\.]))', '. \g<name>', string)
    string = re.sub('\.\.', '...', string)
    string = re.sub('(?P<rep>.)(?P=rep){3,}', '\g<rep>\g<rep>\g<rep>', string)
    tokens = word_tokenize(string)
    return tokens
