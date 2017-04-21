
def traverseSorted(dictionary):
    for key in sorted(dictionary.keys()):
        yield dictionary[key]

def increment(distribution, key, incr):
    if key in distribution:
        distribution[key] += incr
    else:
        distribution[key] = incr
