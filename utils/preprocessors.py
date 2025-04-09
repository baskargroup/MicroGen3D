import numpy as np
import warnings

class Slice:
    """
    Slice a NumPy array
    :param arr: the input array
    :param start: the start index of the slice
    :param end: the end index of the slice

    :return: the sliced array

    :Example:
    >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> s = Slice(arr=arr, start=[1, 1], end=[2, 2])
    >>> s.sliced_arr
    array([[5]])
    """
    def __init__(self, **kwargs):
        arr = kwargs.get('arr', None)
        self.arr = arr
        self.start = kwargs.get('start', [0]*len(self.arr.shape))
        self.end = kwargs.get('end', list(self.arr.shape))
        self.sliced_arr = self.slice()
    
    @property
    def arr(self):
        return self._arr
    
    @arr.setter
    def arr(self, arr):
        assert isinstance(arr, np.ndarray), "Input array must be a NumPy array"
        self._arr = arr 
    
    @property
    def start(self):
        return self._start
    
    @start.setter
    def start(self, start):
        self._check_start_end(start)
        self._start = self._pad(start)
    
    @property
    def end(self):
        return self._end
    
    @end.setter
    def end(self, end):
        self._check_start_end(end)
        self._end = self._pad(end)

    def _check_start_end(self, input):
        assert isinstance(input, list), "The value of start and end must be a list"
        assert all(isinstance(x, int) for x in input), "The value of start and must be a list of integers"
        assert len(input) <= len(self._arr.shape), f"your input array is {len(self._arr.shape)} dimensional,but you have provided {len(input)} values for start"
        assert all(value <= self._arr.shape[id] for id, value in enumerate(input)), "The value of start or end is out of bounds of the array"

    def _pad(self, input):
        return input + [0]*(len(self._arr.shape) - len(input))
    
    def slice(self):
        return self._arr[tuple([slice(s, e) for s, e in zip(self.start, self.end)])]


class TransportLayer:
    """
    Add a transport layer to a NumPy array
    :param arr: the input array
    :param transport_layer_percent: the thickness of the transport layer in percentage

    :return: the array with the transport layer          
    """
    def __init__(self, arr, transport_layer_percent=5):
        self.arr = arr
        self.transport_layer_percent = int(transport_layer_percent)
        self._ndim = len(self.arr.shape)
        self._thickness = int(np.ceil(self.arr.shape[-1] * self.transport_layer_percent / 100))

    @property
    def arr(self):
        return self._arr
    
    @arr.setter
    def arr(self, arr):
        assert isinstance(arr, np.ndarray), "Input array must be a NumPy array"
        if len(set(arr.flatten())) > 2:
            warnings.warn("Input array is not Binary. More than 2 unique values", UserWarning)
        if arr.flatten().max() != 1 or arr.flatten().min() != 0:
            warnings.warn("Input array is not Binary. The values should be 0 and 1", UserWarning)
        self._arr = arr
    
    @property
    def transport_layer_percent(self):
        return self._transport_layer_percent
    
    @transport_layer_percent.setter
    def transport_layer_percent(self, transport_layer_percent):
        assert transport_layer_percent > 0 and transport_layer_percent <= 50, "Transport layer percent must be between 0 and 50"
        self._transport_layer_percent = transport_layer_percent
    
    def _layers(self):
        self._layer_shape = self.arr.shape[:-1] + (self._thickness,)
        self._donor_layer = np.zeros(self._layer_shape)
        self._acceptor_layer = np.ones(self._layer_shape)
    
    def _concat(self):
        return np.concatenate((self._acceptor_layer, self.arr, self._donor_layer), axis=(self._ndim-1))
    
    def add_transport_layer(self):
        self._layers()
        return self._concat()


