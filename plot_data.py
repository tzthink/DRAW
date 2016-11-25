import matplotlib
import sys
import numpy as np

interactive = False   # set to False if you want to write images to file

if not interactive:
    matplotlib.use('Agg')   # force matplotlib to not use any xwindows backend.

import matplotlib.pyplot as plt

def xrecons_grid(X, B, A):
    """
    plots canvas for single time step
    X is x_recons, (batch_size x img_size)
    assumes features = B x A images
    batch is assumed to be a square number
    """
    padsize = 1
    padval = .5
    ph = B + 2*padsize
    pw = A + 2*padsize
    batch_size = X.shape[0]
    N = int(np.sqrt(batch_size))
    X = X.reshape((N, N, B, A))
    img = np.ones((N*ph, N*pw))*padval
    for i in range(N):
        for j in range(N):
            startr = i*ph + padsize
            endr = startr + B
            startc = j*pw + padsize
            endc = startc + A
            img[startr:endr, startc:endc]=X[i,j,:,:]
    return img

if __name__ == '__main__':
    prefix=sys.argv[1]
    out_file=sys.argv[2]
    [C, Lxs, Lzs] = np.load(out_file)
    T, batch_size, img_size=C.shape
    X = 1.0 / (1.0+np.exp(-C))   # x_recons = sigmoid(canvas)
    B = A = int(np.sqrt(img_size))
    if interactive:
        f, arr = plt.subplots(1, T)
    for t in range(T):
        img = xrecons_grid(X[t, :, :], B, A)
        if interactive:
            f, arr = plt.subplots(1, T)
        for t in range(T):
            img = xrecons_grid(X[t, :, :], B, A)
            if interactive:
                arr[t].matshow(img, cmap=plt.cm.gray)
                arr[t].set_xtricks([])
                arr[t].set_ytricks([])
            else:
                plt.matshow(img, cmap=plt.cm.gray)
                imgname='pic/%s_%d.png' % (prefix, t)
                plt.savefig(imgname)
                print(imgname)

    f=plt.figure()
    plt.plot(Lxs, label='Reconstruction Loss Lx')
    plt.plot(Lzs, label='Latent Loss Lz')
    plt.xlabel('iterations')
    plt.legend()
    if interactive:
        plt.show()
    else:
        plt.savefig('pic/%s_loss.png' % (prefix))