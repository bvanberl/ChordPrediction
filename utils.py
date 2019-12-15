
CHROMATIC_DICT = {'B#': 0, 'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 2, 'Eb': 3, 'E': 4, 'E#': 5, 'Fb': 4, 'F': 5,
                  'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11}

REVERSE_CHROMATIC_DICT = {0: 'C', 1: 'C#-Db', 2: 'D', 3: 'D#-Eb', 4: 'E', 5: 'F', 6: 'F#-Gb', 7: 'G',
                 8: 'G#-Ab', 9: 'A', 10: 'A#-Bb', 11: 'B'}

def transpose_to_C(key, note):
    if note in CHROMATIC_DICT:
        transposed_note = (CHROMATIC_DICT[note] - CHROMATIC_DICT[key]) % 12
        return REVERSE_CHROMATIC_DICT[transposed_note]
    else:
        return None