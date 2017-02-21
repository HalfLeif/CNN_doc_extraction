
import xml.etree.ElementTree as et

import os
import sys

def inputNames(directory):
    '''Returns basenames for xml and jpgs in this IRIS directory.'''
    transcription_dir = os.path.join(directory, 'transcription')
    for filename in os.listdir(transcription_dir):
        basename, ext = os.path.splitext(filename)
        if ext == '.xml':
            yield basename

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

    # jpg_name = root.attrib['raw-file']
    result = None
    for child in root:
        if child.tag == 'header':
            result = parseHeader(child)
            break
    return result

def loadPair(directory, basename):
    xml_path = os.path.join(directory, 'transcription', basename + '.xml')
    label = parseXmlFile(xml_path)
    jpg_path = os.path.join(directory, 'images', basename + '.jpg')
    return jpg_path, label


if __name__ == '__main__':
    for xmlpath in transcriptionFiles(sys.argv[1]):
        jpg_name, indexed = parseXmlFile(xmlpath)
        print(jpg_name, indexed)
