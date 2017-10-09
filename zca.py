def whitening(X):
  print "whitening"
  [D,V]=np.linalg.eig(np.cov(X,rowvar=False))
  P = V.dot(np.diag(np.sqrt(1/(D + 1e-9)))).dot(V.T)
  X = X.dot(P)
  return X, P
