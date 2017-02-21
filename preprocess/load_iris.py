
import xml.etree.ElementTree as et

import os
import sys

def transcriptionFiles(directory):
    '''Generator for path to all xml with transcriptions files'''
    transcription_dir = os.path.join(directory, 'transcription')
    for filename in os.listdir(transcription_dir):
        _, ext = os.path.splitext(filename)
        if ext == '.xml':
            filepath = os.path.join(transcription_dir, filename)
            yield filepath

def parseHeader(header):
    '''Returns list of years found.'''
    years = set()
    for item in header:
        if item.attrib['name'] == 'EVENT_YEAR' and len(item) == 0:
            years.add(item.text)
    return list(years)


def parseFile(filepath):
    ''' Parses all the relevant data in the indexed file.

        Returns name of corresponding image as well as
        list of unique years in the image.
    '''
    tree = et.parse(filepath)
    root = tree.getroot()

    jpg_name = root.attrib['raw-file']
    result = None
    for child in root:
        if child.tag == 'header':
            result = parseHeader(child)
            break
    return jpg_name, result


def irisInput(directory):
    ''' Generator for reading IRIS data.'''
    for xmlpath in transcriptionFiles(directory):
        jpg_name, indexed_data = parseFile(xmlpath)
        jpg_path = os.path.join(directory, 'images', jpg_name)
        yield (jpg_path, indexed_data)


if __name__ == '__main__':
    for xmlpath in transcriptionFiles(sys.argv[1]):
        jpg_name, indexed = parseFile(xmlpath)
        print(jpg_name, indexed)
