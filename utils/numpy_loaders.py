import numpy as np
from abc import ABC, abstractmethod
import warnings
import os
import xml.etree.ElementTree as ET
import pandas as pd

class Loader(ABC):
    def __init__(self, **kwargs):
        self.path = kwargs.get('path', None)
        self.file_name = kwargs.get('file_name', None)
        self.data = kwargs.get('data', None)
        self.dimensions = kwargs.get('dimensions', None)
        self.order = kwargs.get('order', 'F')
    
    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, path):
        if path is not None:
            if not os.path.exists(path):
                warnings.warn(f"Path {path} does not exist.")
        if path is None:
            warnings.warn("Path is not provided. Using current working directory.")
            path = os.getcwd()
        self._path = path
    
    @property
    def file_name(self):
        return self._file_name
    
    @file_name.setter
    def file_name(self, file_name):
        if file_name is None:
            warnings.warn("File name is not provided. Checking if data and dimensions are provided.")
        self._file_name = file_name

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        if self.file_name is None:
            if data is None:
                raise ValueError("No data source is provided. Either provide a file name or provide both data and dimension.")
        
            assert len(data) > 0, "Data cannot be empty."
        self._data = data
    
    @property
    def dimensions(self):
        return self._dimensions
    
    @dimensions.setter
    def dimensions(self, dimensions):
        self._dimensions = dimensions


class FromGraspi(Loader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _read_from_file(self):
        with open(os.path.join(self.path, self.file_name), 'r') as f:
            self.dimensions = [int(x) for x in f.readline().split()]
            self.dimensions = [d for d in self.dimensions if d != 0]
            self.data = [float(x) for x in f.readline().split()]

    def load(self):
        if self.file_name is not None:
            self._read_from_file()

        self.array = np.array(self.data).reshape(self.dimensions, order=self.order)    


class FromVti(Loader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _read_dimensions_from_file(self, root):
        piece_extent = root.find('.//Piece').get('Extent')
        self.dimensions = [int(x) + 1 for x in piece_extent.split() if x != '0']

    def _read_data_from_file(self, root):
        self.data = root.find('.//DataArray').text
        self.data = [float(x) for x in self.data.split()]

    def _read_from_file(self):
        with open(self.path + self.file_name, 'r') as f:
            lines = f.read()
            root = ET.fromstring(lines)
            self._read_dimensions_from_file(root)
            self._read_data_from_file(root)
                
    def load(self):
        if self.file_name is not None:
            self._read_from_file()
        self.array = np.array(self.data).reshape(self.dimensions, order=self.order)


class FromXdd(Loader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _read_dimensions(self, f):
        first_line = f.readline().strip().split()
        self.dimensions = [int(x) for x in first_line[1:]]
    
    def _read_data(self, f):
        # Skip the first line, which contains the dimensions
        next(f)
        df = pd.read_csv(f, header=None, sep='\t')
        self.data = df.iloc[:, -2]
    
    def _read_from_file(self):
        with open(self.path + self.file_name, 'r') as f:
            self._read_dimensions(f)
            f.seek(0)  # Reset file pointer to beginning
            self._read_data(f)

    def load(self):
        if self.file_name is not None:
            self._read_from_file()
        self.array = np.array(self.data).reshape(self.dimensions, order=self.order)