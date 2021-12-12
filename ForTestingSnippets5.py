# from multiprocessing.shared_memory import SharedMemory
# from multiprocessing.managers import SharedMemoryManager
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import current_process, cpu_count, Process
# from datetime import datetime
# import numpy as np
# import pandas as pd
# import tracemalloc
# import time
#
#
# def work_with_shared_memory(shm_name, shape, dtype):
#     print(f'With SharedMemory: {current_process()=}')
#     # Locate the shared memory by its name
#     shm = SharedMemory(shm_name)
#     # Create the np.recarray from the buffer of the shared memory
#     np_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
#     return np.nansum(np_array.val)
#
#
# def work_no_shared_memory(np_array: np.recarray):
#     print(f'No SharedMemory: {current_process()=}')
#     # Without shared memory, the np_array is copied into the child process
#     return np.nansum(np_array.val)
#
#
# if __name__ == "__main__":
#     # Make a large data frame with date, float and character columns
#     a = [
#         (datetime.today(), 1, 'string'),
#         (datetime.today(), np.nan, 'abc'),
#     ] * 20000000
#     df = pd.DataFrame(a, columns=['date', 'val', 'character_col'])
#     # Convert into numpy recarray to preserve the dtypes
#     np_array = df.to_records(index=False)
#     del df
#     shape, dtype = np_array.shape, np_array.dtype
#     print(f"np_array's size={np_array.nbytes/1e6}MB")
#
#     # With shared memory
#     # Start tracking memory usage
#     tracemalloc.start()
#     start_time = time.time()
#     with SharedMemoryManager() as smm:
#         # Create a shared memory of size np_arry.nbytes
#         shm = smm.SharedMemory(np_array.nbytes)
#         # Create a np.recarray using the buffer of shm
#         shm_np_array = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
#         # Copy the data into the shared memory
#         np.copyto(shm_np_array, np_array)
#         # Spawn some processes to do some work
#         with ProcessPoolExecutor(cpu_count()) as exe:
#             fs = [exe.submit(work_with_shared_memory, shm.name, shape, dtype)
#                   for _ in range(cpu_count())]
#             for _ in as_completed(fs):
#                 pass
#     # Check memory usage
#     current, peak = tracemalloc.get_traced_memory()
#     print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
#     print(f'Time elapsed: {time.time()-start_time:.2f}s')
#     tracemalloc.stop()
#
#     # Without shared memory
#     tracemalloc.start()
#     start_time = time.time()
#     with ProcessPoolExecutor(cpu_count()) as exe:
#         fs = [exe.submit(work_no_shared_memory, np_array)
#               for _ in range(cpu_count())]
#         for _ in as_completed(fs):
#             pass
#     # Check memory usage
#     current, peak = tracemalloc.get_traced_memory()
#     print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
#     print(f'Time elapsed: {time.time()-start_time:.2f}s')
#     tracemalloc.stop()



import numpy as np

class RingBuffer(object):
    def __init__(self, size, padding=None):
        self.size = size
        self.padding = size if padding is None else padding
        self.buffer = np.zeros((self.size+self.padding, 3))
        self.counter = 0

        self.full = False

    def append(self, data):
        """this is an O(n) operation"""
        data = data[-self.padding:]
        n = len(data)
        print("n:", n)
        print("self.remaining:", self.remaining())
        if self.remaining() < n: self.compact()
        self.buffer[self.counter+self.size:][:n] = data
        print("buffer after append:", self.buffer)
        self.counter += n

    #@property
    def remaining(self):
        return self.padding-self.counter
    #@property
    def view(self):
        """this is always an O(1) operation"""
        print("in view:")
        print("self.buffer: ", self.buffer)
        print("self.buffer[self.counter:]: ", self.buffer[self.counter:])
        print("self.buffer[self.counter:][:self.size]: ", self.buffer[self.counter:][:self.size])
        print("counter", self.counter)
        test = self.buffer[self.counter:][:self.size][-self.counter:]
        print("test: ", test)
        print("test.base is self.buffer: ", test.base is self.buffer)
        return self.buffer[self.counter:][:self.size]

    def view_buffer_elements(self, buffer):
        if self.full:
            return buffer[self.counter:][:self.size]
        else:
            return buffer[self.counter:][:self.size][-self.counter:]

    def view_recent_pings(self, buffer, pings):
        # return buffer[self.counter:][:self.size][-self.counter:][-pings:]
        return buffer[self.counter:][:self.size][-pings:]

    def compact(self):
        """
        note: only when this function is called, is an O(size) performance hit incurred,
        and this cost is amortized over the whole padding space
        """
        print('compacting')
        self.full = True
        self.buffer[:self.size] = self.view()
        print("buffer after compact:", self.buffer)
        self.counter = 0

if __name__ == "__main__":

    rb = RingBuffer(10)
    for i in range(4):
        rb.append([[7,7,7],[8,8,8],[9,9,9]])
        print("View: ", rb.view())
        print("Test view_recent_pings:", rb.view_recent_pings(rb.buffer, 4))
        print(rb.counter)

    print("Test view_buffer_elements:", rb.view_buffer_elements(rb.buffer))
    print("Test view_recent_pings:", rb.view_recent_pings(rb.buffer, 4))

    # rb.append(np.arange(15))
    # print(rb.view)  #test overflow