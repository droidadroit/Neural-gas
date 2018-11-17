# Neural-gas
A lenna image is compressed by vector quantization using Neural gas. The image is divided into blocks of size `4 x 4` and the corresponding vectors are fed to the Neural gas. This generates a codebook of a predetermined size which is used to generate the reconstructed image.  
## Getting Started
### Prerequisites
1. **Anaconda for Python 2.7**  
2. **OpenCV for Python 2.7** 
### Installing
1. **Anaconda for Python 2.7**  
Go to the [downloads page of Anaconda](https://www.anaconda.com/download/) and select the installer for Python 2.7. Once downloaded, installing it should be a straightforward process. Anaconda has along with it most of the packages we need.  

2. **OpenCV for Python 2.7**   
This [page](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html) explains it quite well. 
## Running   
Before running `NG.py`, a few parameters are to be set.    
```python
image_location
bits_per_codevector
block_width
block_height
epochs
tau_i
tau_f
epsilon_i
epsilon_f
```  
`image_location` is set to the relative location of the image from the current directory.  
`bits_per_codevector` is set based on the size of the codebook you desire. For e.g., for a 256-vector codebook, this value should be `8` as `2^8 = 256`.    
`block_width` and `block_height` are set to the size of the blocks the image is divided into. Make sure the blocks cover the the entire image.   
`epochs` is the number of epochs this algorithm is to be run.  

Once the parameters are decided, enter the following command to run the script.  
`python [name of the script] [image_location] [bits_per_codevector] [block_width] [block_height] [epochs] [tau_i] [tau_f] [epsilon_i] [epsilon_f]`    

**Please read the [wiki](https://github.com/droidadroit/Neural-gas/wiki/Neural-gas) for an understanding of the above terms.**  
## Results  
For the image compressed using `NG.py`, click [here](https://github.com/droidadroit/Neural-gas/tree/master/Results).  
The following parameters are used.
```python
block_width = 4
block_height = 4
epochs = 3
tau_i = 0.5
tau_f = 0.005
epsilon_i = 10
epsilon_f = 0.01
```  
`bits_per_codevector` ranged from `1` to `10`.  
