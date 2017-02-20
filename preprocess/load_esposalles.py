
import os
import sys

'''Returns generator of tuples: word image path and its transcription.'''
def loadEsposalles(directory):
    images = []
    labels = []

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
                images.append(imgpath)
                labels.append(linesplit[1])
                # yield (imgpath, linesplit[1])
    return (images, labels)


if __name__ == '__main__':
    paths = sys.argv[1:]
    print(paths)
    for path in paths:
        images, labels = loadEsposalles(path)
        print(len(labels))
