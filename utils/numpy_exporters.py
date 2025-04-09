import numpy as np
import os
from abc import ABC, abstractmethod
from PIL import Image
from scipy import ndimage as nd

class ToImage:
    '''needs modification. Nirmal April 8 2023'''
    def __init__(self, array, mode='L'):
        self.array = array
        self.mode = mode

    def to_image(self, file_path):
        # convert to PIL Image
        img = Image.fromarray(self.array.astype('uint8'), mode=self.mode)

        # save the image
        img.save(file_path)

    def __repr__(self):
        return str(self.array)

class Exporter(ABC):
    '''
    Abstract class to export a numpy array to a file.
    
    Parameters
    ----------
    arr : numpy.ndarray
        The array to be exported.
    path : str, optional
        The path to save the file. The default is '/home/nirmal/Research/ch_2d/data/raw/'.
    file_name : str, optional
        The name of the file. The default is 'test'.

    Returns
    -------
    None
    '''
    def __init__(self, **kwargs):
        self.arr = kwargs.get('arr', None)
        self._ndim = len(self.arr.shape)
        self.order = kwargs.get('order', 'F')

        self.path = kwargs.get('path', '/home/nirmal/Research/ch_2d/data/raw/')
        self.file_name = kwargs.get('file_name', 'test')
        self._create_path()
    
    @property
    def arr(self):
        return self._arr
    
    @arr.setter
    def arr(self, arr):
        if not isinstance(arr, np.ndarray):
            raise TypeError('arr must be a numpy array')
        if arr.size == 0:
            raise ValueError('arr must not be empty')
        self._arr = arr

    @property
    def order(self):
        return self._order
    
    @order.setter
    def order(self, order):
        assert isinstance(order, str), 'order must be a string'
        assert order in ('F', 'C'), 'order must be either F or C'
        self._order = order
    
    @abstractmethod
    def export(self):
        pass

    def _create_path(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)


class ToVti(Exporter):
    '''
    Class to export a numpy array to a VTI image data file.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to be exported.
    origin : list or tuple, optional
        The origin of the array. The default is [0, 0, 0].
    path : str, optional
        The path to save the file. The default is '/home/nirmal/Research/ch_2d/data/raw/'.
    file_name : str, optional
        The name of the file. The default is 'test'.
    order : str, optional
        The order of the array. The default is 'F'.

    Returns
    -------
    None
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._origin = kwargs.get('origin', [0]*self._ndim)
    
    @property
    def origin(self):
        return self._origin
    
    @origin.setter
    def origin(self, origin):
        assert isinstance(origin, (list, tuple)), 'origin must be a list or tuple' 
        assert len(origin) == self._ndim, 'origin dimension must match array dimension'
        self._origin = origin
    
    def _get_org_str(self):
            return ' '.join(str(item) for item in self.origin)
    
    def _get_whole_extent_str(self):
        whole_extent = [0]*6
        for i in range(self._ndim):
            whole_extent[2*i + 1] = self.arr.shape[i] - 1 
        return ' '.join(str(item) for item in whole_extent)
    
    def _write_header(self, f):
        print('<?xml version="1.0"?>',file=f)
        print(' <VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">',file=f)
    
    def _write_image_data(self, f):
        org_str = self._get_org_str()
        extent_str = self._get_whole_extent_str()
        print(f' <ImageData WholeExtent="{extent_str}" Origin="{org_str}" Spacing="1 1 1">', file=f) 
    
    def _write_piece(self, f):
        extent_str = self._get_whole_extent_str()
        print(f' <Piece Extent="{extent_str}">', file=f) 
    
    def _write_point_data(self, f):
        print('<PointData Scalars="u">', file=f)
        print('<DataArray type="Float64" Name="phi" format="ascii">', file=f)
        np.savetxt(f,self.arr.flatten(order=self.order),fmt='%g', delimiter=' ', newline=' ') 
        print('</DataArray>\n</PointData>\n</Piece>\n </ImageData>\n</VTKFile>', file=f)
    
    def export(self):
        with open(os.path.join(self.path, self.file_name+'.vti'), 'w') as f:
            self._write_header(f)
            self._write_image_data(f)
            self._write_piece(f)
            self._write_point_data(f)

class ToGraspi(Exporter):
    """
    Class to export a numpy array to a GRASPI input file.
    Parameters
    ----------
    arr : numpy.ndarray
        The array to be exported.
    path : str, optional
        The path to save the file. The default is '/home/nirmal/Research/ch_2d/data/raw/'.
    file_name : str, optional
        The name of the file. The default is 'test'.
    order : str, optional
        The order of the array. The default is 'F'.
    Returns
    -------
    None
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _get_dimensions(self):
        return ' '.join(str(item) for item in self.arr.shape)

    def export(self):
        with open(os.path.join(self.path, self.file_name+'.txt'), 'w') as f:
            print(self._get_dimensions(), file=f)
            np.savetxt(f,self.arr.flatten(order=self.order),fmt='%g', delimiter=' ', newline=' ')
    
class ToXdd(Exporter):
    """
    Class to export a numpy array to a XDD input file.
    Parameters
    ----------
    arr : numpy.ndarray
        The array to be exported.
    path : str, optional
        The path to save the file. The default is '/home/nirmal/Research/ch_2d/data/raw/'.
    file_name : str, optional
        The name of the file. The default is 'test'.
    order : str, optional
        The order of the array. The default is 'F'.
    total_height : float, optional
        The total height of the morph. The default is 100e-9.
    Returns
    -------
    None
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # total height of the morph is default 100 nm, 
        # irrespective of how many pixels the actual thickness is, z axis for 3D, y axis for 2D 
        self.total_height = kwargs.get('total_height', 100e-9)
        self._pixel_to_meter = self.total_height / (self.arr.shape[-1] - 1)
        self._compute_distance_interface()
    
    def _compute_distance_interface(self):
        self.distance_interface = nd.distance_transform_edt(self.arr) # for 1s
        self.distance_interface = self.distance_interface - nd.distance_transform_edt((1 - self.arr)) # for 0s
        self.distance_interface = self.distance_interface * self._pixel_to_meter # scaling to meters

    def _get_first_line(self):
        return ' '.join(str(item) for item in [self.arr.size, *self.arr.shape])

    def export(self):
        with open(os.path.join(self.path, self.file_name+'.txt'), 'w') as f:
            print(self._get_first_line(), file=f)
            with np.nditer(self.arr, flags=['multi_index'], order=self.order) as it:
                for x in it:
                    print(*it.multi_index, self.arr[it.multi_index], self.distance_interface[it.multi_index], sep='\t', file=f)





