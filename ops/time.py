
import time

time_start = [None]

def reset():
    time_start[0] = time.process_time()

def start():
    print('Clock started.')
    reset()

def lazyStart():
    ''' Returns a lazy operation to start the clock.'''
    return lambda _: start()

def clock():
    ''' Resets the clock. Returns the number of cpu seconds since last reset.'''
    cpu_seconds = time.process_time() - time_start[0]
    print('CPU time: ', cpu_seconds)
    reset()
    return cpu_seconds

def lazyClock():
    ''' Returns a lazy operation to print the cpu time in seconds
        since start or clock was executed last time.'''
    return lambda _: clock()
