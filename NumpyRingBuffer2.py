class NumpyRingBuffer2:
    def __init__(self, buffer, ring_buffer_indices):
        self.buffer = buffer
        # multiprocessing.Array containing two values: [0] = 'left' index; [1] = 'right' index
        self.ring_buffer_indices = ring_buffer_indices
        self.capacity = self.buffer.shape[0]

    def len(self):
        with self.ring_buffer_indices.get_lock():
            len = self.ring_buffer_indices[1] - self.ring_buffer_indices[0]  # right index - left index
        return len

    def is_full(self):
        return self.len() == self.capacity

    def _fix_indices(self):
        """
        Enforce our invariant that 0 <= self._left_index < self._capacity
        """
        with self.ring_buffer_indices.get_lock():
            if self.ring_buffer_indices[0] >= self.capacity:
                self.ring_buffer_indices[0] = self.ring_buffer_indices[0] - self.capacity
                self.ring_buffer_indices[1] = self.ring_buffer_indices[1] - self.capacity
            elif self.ring_buffer_indices[0] < 0:
                self.ring_buffer_indices[0] = self.ring_buffer_indices[0] + self.capacity
                self.ring_buffer_indices[1] = self.ring_buffer_indices[1] + self.capacity

    def append(self, value):
        with self.ring_buffer_indices.get_lock():
            if self.is_full():
                if not self.len():  # If len == 0
                    return
                else:
                    # Increment 'left' index
                    self.ring_buffer_indices[0] = self.ring_buffer_indices[0] + 1

            self.buffer[self.ring_buffer_indices[1] % self.capacity] = value
            # Increment 'right' index
            self.ring_buffer_indices[1] = self.ring_buffer_indices[1] + 1
            self.fix_indices()

    def pop(self):
        if self.len() == 0:
            raise IndexError("Pop from an empty RingBuffer.")
        with self.ring_buffer_indices.get_lock():
            self.ring_buffer_indices[1] = self.ring_buffer_indices[1] - 1
            self._fix_indices()
            res = self.buffer[self.ring_buffer_indices[1] % self.capacity]
        return res

    def peek(self):
        if self.len() == 0:
            raise IndexError("Peek from an empty RingBuffer.")
        with self.ring_buffer_indices.get_lock():
            res = self.buffer[(self.ring_buffer_indices[1] - 1) % self.capacity]
        return res


