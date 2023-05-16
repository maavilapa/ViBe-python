import numpy as np
from numba import jit
import numpy as np

class ViBe():
  def __init__(self, N, _min, R, phi ):
      self.N=N
      self._min=_min
      self.R=R
      self.phi=phi
  def set_background(self, I_gray):
    self.samples=set_background(I_gray, self.N)
    return self.samples
  def update(self, I_gray, samples):
    segMap, samples=update(I_gray,samples, self.N, self._min, self.R, self.phi)
    return segMap, samples

@jit(nopython=False)
def set_background(I_gray, N):
    I_pad = np.pad(I_gray, 8, 'symmetric')
    height = I_pad.shape[0]
    width = I_pad.shape[1]
    samples = np.zeros((height,width,N))
    for i in range(8, height - 8):
        for j in range(8, width - 8):
            for n in range(N):
                x, y = 0, 0
                while(x == 0 and y == 0):
                    x = np.random.randint(-8, 9)
                    y = np.random.randint(-8, 9)
                ri = i + x
                rj = j + y
                samples[i, j, n] = I_pad[ri, rj]
    samples = samples[8:height-8, 8:width-8]
    return samples

@jit(nopython=False)
def update(I_gray, samples, N, _min, R, phi):
    height = I_gray.shape[0]
    width = I_gray.shape[1]
    segMap = np.zeros((height, width)).astype(np.uint8)
    for i in range(height):
        for j in range(width):
            count, index, dist = 0, 0, 0
            while count < _min and index < N:
                dist = np.abs(I_gray[i,j] - samples[i,j,index])
                if dist < R:
                    count += 1
                index += 1
            if count >= _min:
                segMap[i,j] = 0
                r = np.random.randint(0, phi-1+1)
                if r == 0:
                    r = np.random.randint(0, N-1+1)
                    samples[i,j,r] = I_gray[i,j]
                r = np.random.randint(0, phi-1+1)
                if r == 0:
                    x, y = 0, 0
                    while(x == 0 and y == 0):
                        x = np.random.randint(-8, 9)
                        y = np.random.randint(-8, 9)
                    r = np.random.randint(0, N-1+1)
                    ri = i + x
                    rj = j + y
                    try:
                        samples[ri, rj, r] = I_gray[i, j]
                    except:
                        pass
            else:
                segMap[i, j] = 255
    return segMap, samples
