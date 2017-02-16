
import os
import sys

'''Returns generator of tuples: word image path and its transcription.'''
def loadEsposalles(directory):
    for record in os.listdir(directory):
        record_dir = os.path.join(directory, record, 'words')
        if not os.path.isdir(record_dir):
            # Happens for README.txt
            continue

        filename = record + '_transcription.txt'
        transcription_file = os.path.join(record_dir, filename)
        if not os.path.isfile(transcription_file):
            print('ERROR:', transcription_file)
            continue

        with open(transcription_file) as file:
            for line in file:
                linesplit = str.split(line, ':')
                if len(linesplit) != 2:
                    print('ERROR: ', line)
                imgpath = os.path.join(record_dir, linesplit[0] + '.png')
                yield (imgpath, linesplit[1])


if __name__ == '__main__':
    paths = sys.argv[1:]
    print(paths)
    for path in paths:
        gen = loadEsposalles(path)
        for i in range(20):
            print(gen.__next__())
