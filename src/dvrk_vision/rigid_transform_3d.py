import numpy as np
from math import sqrt

# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigidTransform3D(A, B):
  assert len(A) == len(B)
  assert A.shape[1] == 3
  N = A.shape[0] # total points

  centroid_A = np.mean(A, axis=0)
  centroid_B = np.mean(B, axis=0)
  
  # centre the points
  AA = A - np.tile(centroid_A, (N, 1))
  BB = B - np.tile(centroid_B, (N, 1))

  # dot is matrix multiplication for array
  H = np.transpose(AA) * BB

  U, S, Vt = np.linalg.svd(H)

  R = Vt.T * U.T

  # special reflection case
  if np.linalg.det(R) < 0:
     print "Reflection detected"
     Vt[2,:] *= -1
     R = Vt.T * U.T

  t = -R*centroid_A.T + centroid_B.T

  print t

  return R, t

def calculateRMSE(A, B, rotation, translation):
  assert len(A) == len(B)
  assert A.shape[1] == 3
  n = A.shape[0]
  A2 = (rotation*A.T) + np.tile(translation, (1, A.shape[0]))
  A2 = A2.T

  # Find the error
  err = A2 - B

  err = np.multiply(err, err)
  err = np.sum(err)
  rmse = np.sqrt(err/n);

  print "Points A"
  print A
  print ""

  print "Points B"
  print B
  print ""

  print "Rotation"
  print rotation
  print ""

  print "Translation"
  print translation
  print ""

  print "RMSE:", rmse
  print "If RMSE is near zero, the function is correct!"
  return rmse

if __name__ == '__main__':

  # Test with random data

  # Random rotation and translation
  R = np.mat(np.random.rand(3,3))
  t = np.mat(np.random.rand(3,1))

  # make R a proper rotation matrix, force orthonormal
  U, S, Vt = np.linalg.svd(R)
  R = U*Vt

  # remove reflection
  if np.linalg.det(R) < 0:
     Vt[2,:] *= -1
     R = U*Vt

  # number of points
  n = 10

  A = np.mat(np.random.rand(n,3))
  B = R*A.T + np.tile(t, (1, n))
  B = B.T

  # recover the transformation
  ret_R, ret_t = rigidTransform3D(A, B)
  calculateRMSE(A,B,ret_R, ret_t)

