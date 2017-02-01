from balu.ImagesAndData import balu_load
from balu.Classification import Bcl_lda
import matplotlib
matplotlib.use('TkAgg')
from balu.InputOutput import Bio_plotfeatures
from balu.PerformanceEvaluation import Bev_performance
data = balu_load('datagauss')           #simulated data (2 classes, 2 features)
X = data['X']
d = data['d']
Xt = data['Xt']
dt = data['dt']
Bio_plotfeatures(X, d)                  # plot feature space
op = {'p': []}
ds, options = Bcl_lda(X, d, Xt, op)     # LDA classifier
p = Bev_performance(ds, dt)             # performance on test data
print p
