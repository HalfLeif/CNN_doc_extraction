
import os

def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == '__main__':
    mkdirs('~/foo/bar')
    mkdirs('~/foo/bar')
