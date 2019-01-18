import sys
import string
import os
import pickle
#from bs4 import BeautifulSoup
from pprint import pprint
from operator import itemgetter

DATA_FOLDER = 'reuters-dataset'
# All data files are named reut2-0[0-21].sgm
DATA_FILENAME = 'reut2-0%s.sgm'
DATA_NO_OF_FILES = 22

#Path for saving data
SAVE_FOLDER = 'pickels'
SAVE_ALL_ENTRIES_FILE = 'all_parsed_entries.pkl'

TOP_3000_K_3 = 'top_3000_sorted_k_3'
TOP_3000_K_4 = 'top_3000_sorted_k_4'
TOP_3000_K_5 = 'top_3000_sorted_k_5'

ALL_3GRAM_K_3 = 'all_grams_sorted_k_3'

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
        self.unfiltered_body = entry.text
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
Extract the top 3000 ngrams with continous grams
for the first 100 files of what is being passed
and saves the mapping
'''
def _create_top_3000_ngrams(k, all_data):
    counter_grams = dict()
    for entry in [ x for x in all_data if x.lewis_split == 'TRAIN' ]:
        for i in range(len(entry.clean_body) - k):
            if entry.clean_body[i:i+k] not in counter_grams:
                counter_grams[entry.clean_body[i:i+k]] = 0
            counter_grams[entry.clean_body[i:i+k]] += 1
    tuples_to_sort = [ (k,v) for k,v in counter_grams.items() ]
    sorted_tuples = sorted(tuples_to_sort, key=itemgetter(1), reverse=True)
    # Extract top 3000 and create a mapping
    top_list = [ x[0] for x in sorted_tuples[:3000] ]
    #save it
    path = None
    if k == 3:
        path = TOP_3000_K_3
    elif k == 4:
        path = TOP_3000_K_4
    elif k == 5:
        path = TOP_3000_K_5
    else:
        print("INVALID K IN DATA")
        quit()
    path = "%s/%s.pkl" % (SAVE_FOLDER, path)
    f = open(path, 'wb')
    pickle.dump(top_list, f)
    f.close()
    print("Saved for k = %d" % k)


def load_top_3000(k):
    path = None
    if k == 3:
        path = TOP_3000_K_3
    elif k == 4:
        path = TOP_3000_K_4
    elif k == 5:
        path = TOP_3000_K_5
    else:
        print("INVALID K IN DATA")
        quit()
    path = "%s/%s.pkl" % (SAVE_FOLDER, path)
    f = open(path, 'rb')
    top_list = pickle.load(f)
    f.close()
    return top_list

def _save_all_3grams(all_data):
    k = 3
    counter_grams = dict()
    for entry in [ x for x in all_data if x.lewis_split == 'TRAIN' ]:
        for i in range(len(entry.clean_body) - k):
            if entry.clean_body[i:i+k] not in counter_grams:
                counter_grams[entry.clean_body[i:i+k]] = 0
            counter_grams[entry.clean_body[i:i+k]] += 1
    tuples_to_sort = [ (k,v) for k,v in counter_grams.items() ]
    sorted_tuples = sorted(tuples_to_sort, key=itemgetter(1), reverse=True)
    all_sorted = [ x[0] for x in sorted_tuples ] 
    path = "%s/%s" % (SAVE_FOLDER, ALL_3GRAM_K_3)
    f = open(path, 'wb')
    pickle.dump(all_sorted, f)
    f.close()

def load_all_3grams():
    path = "%s/%s" % (SAVE_FOLDER, ALL_3GRAM_K_3)
    f = open(path, 'rb')
    top = pickle.load(f)
    f.close()
    return top


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
    # for e in all_entries[0:20]:
        # print(e)
    # _save_all_3grams(all_entries)
    top = load_all_3grams()
    print(top[:50])
    print(len(top))
    # _create_top_3000_ngrams(3, all_entries)
    # _create_top_3000_ngrams(4, all_entries)
    # _create_top_3000_ngrams(5, all_entries)
    # top_k_3 = load_top_3000(3)
    # print(top_k_3[:50])
    # top_k_4 = load_top_3000(4)
    # print(top_k_4[:50])
    # top_k_5 = load_top_3000(5)
    # print(top_k_5[:50])

