
import xml.etree.ElementTree as et

import csv
import os
import sys


# def inputNames(directory):
#     '''Returns pairnames for xml and jpgs in this IRIS directory.'''
#     transcription_dir = os.path.join(directory, 'transcription')
#     files = os.listdir(transcription_dir)
#     return [os.path.splitext(f)[0] for f in files if f.endswith('.xml')]
#
# def inputNamesGen(directory):
#     '''Generates pairnames for xml and jpgs in this IRIS directory.'''
#     transcription_dir = os.path.join(directory, 'transcription')
#     for filename in os.listdir(transcription_dir):
#         pairname, ext = os.path.splitext(filename)
#         if ext == '.xml':
#             yield pairname

def xmlFiles(directory):
    transcription_dir = os.path.join(directory, 'transcription')
    files = os.listdir(transcription_dir)
    return [os.path.join(transcription_dir, f) for f in files if f.endswith('.xml')]

# def jpgFiles(directory):
#     images_dir = os.path.join(directory, 'images')
#     files = os.listdir(images_dir)
#     return [os.path.join(images_dir, f) for f in files if f.endswith('.jpg')]


def parseHeader(header):
    '''Returns greatest year found or -1.'''
    maxyear = -1
    for item in header:
        if item.attrib['name'] == 'EVENT_YEAR' and len(item) == 0 and item.text:
            maxyear = max(maxyear, int(item.text))
    return maxyear


def parseXmlFile(filepath):
    ''' Parses all the relevant data in the indexed file.

        Returns name of corresponding image as well as
        list of unique years in the image.
    '''
    tree = et.parse(filepath)
    root = tree.getroot()

    jpg_name = root.attrib['raw-file']
    result = -1
    for child in root:
        if child.tag == 'header':
            result = parseHeader(child)
            break
    return jpg_name, result

# def parseXmlFiles(directory, pairnames):
#     xml_paths = map(lambda p: os.path.join(directory, 'transcription', p + '.xml'), pairnames)
#     return map(parseXmlFile, xml_paths)
#
# def getJpgPaths(directory, pairnames):
#     return map(lambda p: os.path.join(directory, 'images', p + '.jpg'), pairnames)

def loadTranscriptions(filepath):
    with open(filepath, 'r', newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data_path = reader.__next__()[0]
        jpgs = []
        years = []
        for row in reader:
            jpgs.append(os.path.join(data_path, 'images', row[0]))
            years.append(int(row[1]))
            # yield os.path.join(data_path, jpg_name), int(year)
        return jpgs, years

def writeTranscriptions(data_path, filepath):
    ''' Writes imagename and parsed labels into csv under data/.'''
    filepath = os.path.join('data', csvname)
    with open(filepath, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([data_path])
        for xml_file in xmlFiles(data_path):
            jpg_file, year = parseXmlFile(xml_file)
            writer.writerow([jpg_file, year])

if __name__ == '__main__':
    writeTranscriptions(sys.argv[1], sys.argv[2])
    # print(list(loadTranscriptions(sys.argv[2])))
