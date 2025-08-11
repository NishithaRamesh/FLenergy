import threading

def start_function():
    thread = threading.Thread(target=my_function, args=(counter,))
    thread.start()
    return thread

def stop_function(thread):
    thread.join()

