
import xml.etree.ElementTree as et

import os
import sys

'''Generator for path to all xml with transcriptions files'''
def transcriptionFiles(directory):
    transcription_dir = os.path.join(directory, 'transcription')
    for filename in os.listdir(transcription_dir):
        basename, ext = os.path.splitext(filename)
        if ext == '.xml':
            filepath = os.path.join(transcription_dir, filename)
            yield (filepath, basename)

def parseHeader(header):
    years = set()
    for item in header:
        if item.attrib['name'] == 'EVENT_YEAR' and len(item) == 0:
            years.add(item.text)
    return list(years)


def parseFile(filepath):
    tree = et.parse(filepath)
    root = tree.getroot()

    jpg_name = root.attrib['raw-file']
    result = None
    for child in root:
        if child.tag == 'header':
            result = parseHeader(child)
            break
    return jpg_name, result

if __name__ == '__main__':
    for xmlpath, name in transcriptionFiles(sys.argv[1]):
        jpg_name, indexed = parseFile(xmlpath)
        print(jpg_name, indexed)
