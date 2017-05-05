
def traverseSorted(dictionary):
    for key in sorted(dictionary.keys()):
        yield dictionary[key]

def increment(distribution, key, incr):
    if key in distribution:
        distribution[key] += incr
    else:
        distribution[key] = incr

def printDict(dictionary):
    for key in dictionary:
        print(key, dictionary[key])

def printDictSorted(dictionary):
    for key in sorted(dictionary.keys()):
        print(key, dictionary[key])
