import sys
import string
import os
import pickle
from bs4 import BeautifulSoup
from pprint import pprint

DATA_FOLDER = 'reuters-dataset'
# All data files are named reut2-0[0-21].sgm
DATA_FILENAME = 'reut2-0%s.sgm'
DATA_NO_OF_FILES = 22

#Path for saving data
SAVE_FOLDER = 'pickels'
SAVE_ALL_ENTRIES_FILE = 'all_parsed_entries.pkl'

#Stop words to filter out before doing anything to the data
#taken from https://www.textfixer.com/tutorials/common-english-words.txt
# NOTE: We have no idea if this is the same as used by original authors
STOP_WORDS = [
    "a","able","about","across","after","all","almost","also","am","among","an","and","any",
    "are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do",
    "does","either","else","ever","every","for","from","get","got","had","has","have","he",
    "her","hers","him","his","how","however","i","if","in","into","is","it","its","just",
    "least","let","like","likely","may","me","might","most","must","my","neither","no","nor",
    "not","of","off","often","on","only","or","other","our","own","rather","said","say","says",
    "she","should","since","so","some","than","that","the","their","them","then","there","these",
    "they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where",
    "which","while","who","whom","why","will","with","would","yet","you","your"
        ]

ALLOWED_CHARS = string.ascii_lowercase + ' '

'''
Contains one reuters entry and metadata
'''
class ReutersEntry:
    def __init__(self, entry):
        self.id = int(entry['newid'])
        self.lewis_split = entry['lewissplit']
        self.has_topics = True if entry['topics'] == 'YES' else False
        self.topics = []
        for topic in entry.topics.find_all('d'):
            self.topics.append(topic.get_text())
        if entry.text is not None:
            self.clean_body = filter_body(entry.text)
        else:
            self.clean_body = ''
        # 3-grams definitions
        # "Stride"- like three gram
        self.stride_3_grams = None
        # Full 3 gram
        self.full_3_grams = None
        # Run init functions
        #We calculate contigous for the first 100 entries
        if self.id < 100:
            self.calc_stride_3_grams()

    def __str__(self):
        topics = ','.join(self.topics)
        raw_string = "ID:%d, Topics:%s, Lewis-split:%s, Clean body:%s..."
        return raw_string % (self.id, topics, self.lewis_split, self.clean_body[0:50])
    '''
    Calculates all continous 3-grams for this entry
    NOTE: Any 3-gram not in the text will not be found as a key
    in the dictionary. Use "dictionary.get('alg')" instead of
    dictionary['alg'] (".get" returns None if there is no such key)
    '''
    def calc_stride_3_grams(self):
        self.stride_3_grams = dict()
        for i in range(0, len(self.clean_body) - 3):
            if self.clean_body[i:i+3] not in self.stride_3_grams:
                self.stride_3_grams[self.clean_body[i:i+3]] = 0
            self.stride_3_grams[self.clean_body[i:i+3]] += 1

'''
Reads a file an returns an BeautifulSoup object
'''
def parse_file(filename):
    file_contents = None
    with open(DATA_FOLDER + "/" + filename, 'r') as f:
        file_contents = f.readlines()
    return BeautifulSoup(' '.join(file_contents), 'html.parser')
    # return BeautifulSoup(' '.join(file_contents), 'lxml')

'''
Filters a CLEAN body and returns a string with lowercase chars
Only the chars in ALLOWED_CHARS and words not in STOP_WORDS
are kept
'''
def filter_body(body):
    body = body.lower()
    cleansed = ''.join([ x for x in body if x in ALLOWED_CHARS ])
    filtered = ' '.join([ x for x in cleansed.split() if x not in STOP_WORDS])
    return filtered

'''
Creates all entries for the reuters list

Note: Will involve calculating 3-grams, MIGHT TAKE TIME
'''
def create_all_entries_list():
    all_entries = []
    for file_index in range(DATA_NO_OF_FILES):
        #Fix for leading zeros
        print("Parsing file %d..." % file_index)
        file_index = "{:02d}".format(file_index)
        file_soup = parse_file(DATA_FILENAME % file_index)
        no_file = 1
        for entry in file_soup.find_all('reuters'):
            if entry['lewissplit'] == 'NOT-USED' or entry['topics'] != 'YES':
                continue
            print("\tParsing entry %d     \r" % no_file, end='')
            all_entries.append(ReutersEntry(entry))
            no_file += 1
        print("\nDone")
    print("Total of %d files" % len(all_entries))
    return all_entries

'''
Returns a list with <Reuter> objects representing the whole
dataset in order

If there is a pickled file, it loads from that, otherwise
it creates it and saves it
'''
def load_all_entries():
    #Check if we have the pickle
    if os.path.isfile("%s/%s" % (SAVE_FOLDER, SAVE_ALL_ENTRIES_FILE)):
        with open("%s/%s" % (SAVE_FOLDER, SAVE_ALL_ENTRIES_FILE), 'rb') as f:
            all_entries = pickle.load(f)
        return all_entries
    #We need to create it
    print("Could not load pickle file, creating it...")
    all_entries = create_all_entries_list()
    #save it
    if not os.path.isdir(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    f = open("%s/%s" % (SAVE_FOLDER, SAVE_ALL_ENTRIES_FILE), 'wb')
    pickle.dump(all_entries, f)
    f.close()
    #Done here
    return all_entries


#NOTE: For debugging purposes
if __name__ == '__main__':
    all_entries = load_all_entries()
    for e in all_entries[0:20]:
        print(e)
