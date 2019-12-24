import os
import re
import utils
import numpy as np
import json
from keras.utils import to_categorical


RAW_DATA_PATH = '../downloads/'         # Path of McGill Billboard Dataset
DATASET_PATH = '../data/'               # Path to output parsed dataset
CHORD_FILENAME = 'salami_chords.txt'    # Name of Billboard annotation files
ALLOW_ROOT_ANNOTATIONS = False          # Whether to allow
ALLOW_SECTION_ANNOTATIONS = False       # Whether to consider section types (e.g. verse, chorus) as different tokens
ALLOW_BRACKET_EXTENSIONS = False        # Whether to consider bracketed chord extensions
SEQ_LENGTH = 8
token_dict = {}                         # Tokens for the chords and annotations
chord_freq = {}
reverse_token_dict = {}


def parse_key_line(line):
    '''
    Parses the key of the song given the appropriate line from a song's text file
    :param line: The line of text to be parsed
    :return: The key of the song
    '''
    return (line.split(':')[-1]).strip()


def parse_metre_line(line):
    '''
    Parses the metre of the song given the appropriate line from a song's text file
    :param line: The line of text to be parsed
    :return: The metre of the song
    '''
    return int((line.split(':')[-1]).strip()[0])


def parse_bars(bars, key, metre):
    '''
    Given a list of bars containing chords, parse out a list of chords
    :param bars: A list of strings containing chord annotations
    :param key: The key of the song
    :param metre: The metre of the song
    :return: A list of chords transposed to C, corresponding to the bars passed
    '''
    sequence = []
    for bar in bars:
        chords = bar.split(' ')[1:-1]
        if(len(chords) == 1): # Repeat chord for # beats in metre
            chords = [chords[0]] * metre
        elif(len(chords) == 2): # Repeat chords for half of the beats in the metre each
            chords = ([chords[0]] * int(metre / 2)) + ([chords[1]] * int(metre / 2))
        for i in range(len(chords)):
            if(chords[i] == '.'): # Repeat last chord
                chords[i] = chords[i-1]
        for i in range(len(chords)):
            note = chords[i].split(':')[0]
            transposed_note = utils.transpose_to_C(key, note)  # Transpose chord to key of C
            if transposed_note != None:
                chords[i] = transposed_note + ':' + chords[i].split(':')[1]
                if not ALLOW_ROOT_ANNOTATIONS: # If alternative root annotations not allowed, remove them
                    if '/' in chords[i]:
                        chords[i] = chords[i][0:chords[i].find('/')]
                if not ALLOW_BRACKET_EXTENSIONS:
                    if '(' in chords[i]:
                        chords[i] = chords[i][0:chords[i].find('(')]
            else:
                chords[i] = 'N/C' # A special token standing for "no chord"
            if not chords[i] in token_dict:  # If we don't have a token for this chord, add one
                token_dict[chords[i]] = len(token_dict)
                chord_freq[chords[i]] = 0
            sequence.append(token_dict[chords[i]])  # Add the section annotation to the list of chords
            chord_freq[chords[i]] += 1
    return sequence


def parse_chords_line(line, key, metre):
    '''
    Parses a line from the annotation file that contains chords
    :param line: The line to be parsed
    :param key: The key of the song
    :param metre: The metre of the song
    :return: A list of chords and structural annotations from the current line
    '''
    phrase = []
    split = line.split('|')
    meta = split[0]
    meta_split = meta.split(',')
    if(len(meta_split) > 1): # If we are at the start of a new section of a song
        if ALLOW_SECTION_ANNOTATIONS:
            section = meta_split[1].strip()
        else:
            section = "Section" # General indicator for new section
        if not section in token_dict: # If we don't have a token for this section definition, add one
            token_dict[section] = len(token_dict)
            chord_freq[section] = 0
        chord_freq[section] += 1
        phrase.append(token_dict[section]) # Add the section annotation to the list of chords

    # Get the bars for the line
    bars = split[1:-1]

    # Determine how many times this line is repeated
    repeats = 1
    repeat_section = split[-1]
    if repeat_section.find('x') != -1:
        repeat_str = repeat_section.split(' ')[1]
        repeat_str = re.sub("\D", "", repeat_str)
        if len(repeat_str) > 0:
            repeats = int(repeat_str)

    # Parse the bars for this line
    for i in range(repeats):
        phrase += parse_bars(bars, key, metre)
    return phrase


def read_chord_seq(dir_path):
    '''
    Reads the chords from a song annotation within a directory.
    :param dir_path: Path of the directory that contains the song's annotation
    :return: A list of chords and annotations corresponding to this song
    '''
    file = open(dir_path + "/" + CHORD_FILENAME, 'r')
    key = 'C'
    sequence = []
    for line in file: # Read the file line-by-line
        if '|' in line: # Line contains bars
            line_chords = parse_chords_line(line, key, metre)
            sequence += line_chords
        elif 'tonic' in line: # Line contains key
            key = parse_key_line(line)
        elif 'metre' in line: # Line contains metre
            metre = parse_metre_line(line)
    if not 'end' in token_dict:  # If we don't have a token for this chord, add one
        token_dict['end'] = len(token_dict)
    sequence.append(token_dict['end'])
    return sequence


def preprocess_data():
    '''
    Parses the data in the McGill Billboard dataset to get a sequence of chords and structural annotations for each song
    '''
    chord_seqs = []
    max_len = 0
    for f in os.scandir(RAW_DATA_PATH): # Iterate over all directories
        if f.is_dir() and os.path.exists(f.path + "/" + CHORD_FILENAME):
            chord_seq = read_chord_seq(f.path)
            chord_seqs.append(chord_seq)
            if len(chord_seq) > max_len:
                max_len = len(chord_seq)

    n_vocab = len(token_dict)
    n_seqs = len(chord_seqs)
    seqs = []
    outputs = [] # Output predicted character
    for i in range(len(chord_seqs)):
        for j in range(0, len(chord_seqs[i]) - SEQ_LENGTH, 2):
            x = chord_seqs[i][j : j + SEQ_LENGTH]
            y = chord_seqs[i][j + SEQ_LENGTH]
            seqs.append(np.expand_dims(np.asarray(x), axis=1))
            outputs.append(y)
    X = np.stack(seqs)
    print(X.shape)
    X = X / float(n_vocab) # Normalize inputs
    print(np.max(outputs))

    Y = to_categorical(outputs, num_classes=len(token_dict))

    chord_freq_sorted = sorted(chord_freq, key=chord_freq.get, reverse=True)
    reverse_token_dict = dict((v, k) for k, v in token_dict.items())
    np.save('../data/X.npy', X)
    np.save('../data/Y.npy', Y)
    with open('./config/tokens.json', 'w') as f:
        json.dump(token_dict, f)
    with open('./config/tokens_reverse.json', 'w') as f:
        json.dump(reverse_token_dict, f)
    with open('./config/chord_freq.json', 'w') as f:
        json.dump(chord_freq, f)

if __name__=="__main__":
    preprocess_data()