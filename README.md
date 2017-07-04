csalgs: Compressed sensing algorithms in Python
===============================================

This is a small collection of compressed sensing/low rank matrix recovery algorithms in Python. It's neither complete nor very elaborate -- it's mainly just for learning exisiting algorithms or for testing purposes. Use at your own risk :)

## Content

- `csalg.tt`: Low-rank tensor recovery for the tensor train format
  - `iht.py`: Iterative hard thresholding (projected gradient descent)
  - `altmin.py`: Alternating Least Squares
  - `_altmin_gpu.py`: A CUDA implementation of alternating least squares 
- `csalgs.lowrank`: Low-rank matrix recovery
  - `iht.py`: Iterative hard thresholding (projected gradient descent)
  - `convex.py`: Convex optimization methods (nuclear norm minimization)
  - `altmin.py`: Alternating Least Squares
- `csalg.cs`: Compressed Sensing (Recovery of sparse vectors)
  - `iht.py`: Iterative hard thresholding (projected gradient descent)


## LICENSE

Distributed under the terms of the GPLv3 license (see [LICENSE](LICENSE)).
