"""Test the DLT implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import dlt

def test_exceptions():
    #TODO make some error cases here
    pass


def test_3d():

    # 3D (x, y, z) coordinates (in cm) of the corner of a cube (the measurement error is at least 0.2 cm)
    xyz = [[0, 0, 0],
           [0, 12.3, 0],
           [14.5, 12.3, 0],
           [14.5, 0, 0],
           [0, 0, 14.5],
           [0, 12.3, 14.5],
           [14.5, 12.3, 14.5],
           [14.5, 0, 14.5]]

    # 2D (u, v) coordinates (in pixels) of 4 different views of the cube
    uv1 = [[1302, 1147],
           [1110, 976],
           [1411, 863],
           [1618, 1012],
           [1324, 812],
           [1127, 658],
           [1433, 564],
           [1645, 704]]

    uv2 = [[1094, 1187],
           [1130, 956],
           [1514, 968],
           [1532, 1187],
           [1076, 854],
           [1109, 647],
           [1514, 659],
           [1523, 860]]

    uv3 = [[1073, 866],
           [1319, 761],
           [1580, 896],
           [1352, 1016],
           [1064, 545],
           [1304, 449],
           [1568, 557],
           [1313, 668]]

    uv4 = [[1205, 1511],
           [1193, 1142],
           [1601, 1121],
           [1631, 1487],
           [1157, 1550],
           [1139, 1124],
           [1628, 1100],
           [1661, 1520]]

    # calibration results
    err1_blessed = 2.57965902006
    err2_blessed = 3.04214261951
    err3_blessed = 6.16791729721
    err4_blessed = 2.79210779319
    L1_blessed = np.array([  2.95265206e+01, -8.97370130e+00, -6.96531802e-01,  1.30358419e+03,
                            -4.06246400e+00, -8.02186056e+00, -2.44358618e+01,  1.14686150e+03,
                             4.94180751e-03,  6.18568358e-03, -1.68242140e-03,  1.00000000e+00])
    L2_blessed = np.array([  3.19407422e+01,  1.26911035e+01, -4.63671185e+00,  1.09701804e+03,
                             1.86877074e+00, -9.99243817e+00, -2.56231471e+01,  1.18695817e+03,
                             1.43560285e-03,  9.01401595e-03, -2.88449313e-03,  1.00000000e+00])
    L3_blessed = np.array([  1.16209215e+01,  2.44307350e+01, -8.06307139e-01,  1.07849968e+03,
                             5.33446749e+00, -5.99924577e+00, -2.22602954e+01,  8.68588147e+02,
                            -4.81341554e-03,  3.71965408e-03,  3.40587076e-04,  1.00000000e+00])
    L4_blessed = np.array([  3.04486953e+01,  2.06678879e+00, -1.52883726e+01,  1.20481687e+03,
                            -7.87459694e-01, -2.66125606e+01, -1.32016005e+01,  1.50953468e+03,
                             6.16247151e-04,  2.74227353e-03, -1.03889378e-02,  1.00000000e+00])

    # reconstruction results
    error_cm_blessed = 0.108730880148
    xyz1234_blessed = np.array([[ -8.09297218e-02, -7.60766130e-02,  8.18317612e-02],
                                [  6.14967987e-02,  1.23308395e+01, -3.34614720e-02],
                                [  1.43971386e+01,  1.22842067e+01, -1.01040774e-01],
                                [  1.46310434e+01,  6.92701815e-02,  5.15954438e-02],
                                [  9.68520833e-03,  6.59756252e-02,  1.44007915e+01],
                                [  1.07361971e-02,  1.22785425e+01,  1.45588380e+01],
                                [  1.45309228e+01,  1.23050727e+01,  1.45759737e+01],
                                [  1.44428869e+01, -6.01772394e-02,  1.44702910e+01]])


    nd = 3  # number of dimensions
    nc = 4  # number of cameras
    npts = 8  # number of data points in each image

    # perform the calibrations
    L1, err1 = dlt.calibrate(nd, xyz, uv1)
    L2, err2 = dlt.calibrate(nd, xyz, uv2)
    L3, err3 = dlt.calibrate(nd, xyz, uv3)
    L4, err4 = dlt.calibrate(nd, xyz, uv4)

    # perform reconstruction
    xyz1234 = np.zeros((len(xyz), 3))
    L1234 = [L1, L2, L3, L4]
    for i in range(npts):
        xyz1234[i, :] = dlt.reconstruct(nd, nc, L1234, [uv1[i], uv2[i], uv3[i], uv4[i]])
    xyz = np.array(xyz)
    error_cm = np.mean(np.sqrt(((xyz1234 - xyz)**2).sum(axis=1)))

    # check datatypes
    assert(isinstance(xyz1234, np.ndarray))
    assert(isinstance(L1, np.ndarray))

    # check calibration values
    assert(np.allclose(L1, L1_blessed))
    assert(np.allclose(L2, L2_blessed))
    assert(np.allclose(L3, L3_blessed))
    assert(np.allclose(L4, L4_blessed))

    # check calibration errors
    assert(np.allclose(err1, err1_blessed))
    assert(np.allclose(err2, err2_blessed))
    assert(np.allclose(err3, err3_blessed))
    assert(np.allclose(err4, err4_blessed))

    # check reconstruction values
    assert(np.allclose(xyz1234_blessed, xyz1234))

    # check reconstruction error
    assert(np.allclose(error_cm, error_cm_blessed))

    # plot the images and reconstruction
    uv1 = np.asarray(uv1)
    uv2 = np.asarray(uv2)
    uv3 = np.asarray(uv3)
    uv4 = np.asarray(uv4)

    #TODO: make this its own module function
    # plot image points
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    ax[0, 0].plot(uv1[:, 0], uv1[:, 1], 'o')
    ax[0, 1].plot(uv2[:, 0], uv2[:, 1], 'o')
    ax[1, 0].plot(uv3[:, 0], uv3[:, 1], 'o')
    ax[1, 1].plot(uv4[:, 0], uv4[:, 1], 'o')
    fig.tight_layout()

    for _ax in ax.flatten():
        _ax.set_xticklabels([])
        _ax.set_yticklabels([])


    #TODO: make this its own module
    # plot reconstruction
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o')
    ax.plot(xyz1234[:, 0], xyz1234[:, 1], xyz1234[:, 2], 'x')

    plt.show()


def t():


    L1, err1 = DLTcalib(nd, xyz, uv1)
    print 'Camera calibration parameters based on view #1:'
    print L1
    print 'Error of the calibration of view #1 (in pixels):'
    print err1
    L2, err2 = DLTcalib(nd, xyz, uv2)
    print 'Camera calibration parameters based on view #2:'
    print L2
    print 'Error of the calibration of view #2 (in pixels):'
    print err2
    L3, err3 = DLTcalib(nd, xyz, uv3)
    print 'Camera calibration parameters based on view #3:'
    print L3
    print 'Error of the calibration of view #3 (in pixels):'
    print err3
    L4, err4 = DLTcalib(nd, xyz, uv4)
    print 'Camera calibration parameters based on view #4:'
    print L4
    print 'Error of the calibration of view #4 (in pixels):'
    print err4
    xyz1234 = np.zeros((len(xyz),3))
    L1234 = [L1,L2,L3,L4]
    for i in range(len(uv1)):
        xyz1234[i,:] = DLTrecon( nd, nc, L1234, [uv1[i],uv2[i],uv3[i],uv4[i]] )
    print 'Reconstruction of the same 8 points based on 4 views and the camera calibration parameters:'
    print xyz1234
    print 'Mean error of the point reconstruction using the DLT (error in cm):'
    print np.mean(np.sqrt(np.sum((np.array(xyz1234)-np.array(xyz))**2,1)))

    print ''
    print 'Test of the 2D DLT'
    print '2D (x, y) coordinates (in cm) of the corner of a square (the measurement error is at least 0.2 cm):'
    xy = [[0,0], [0,12.3], [14.5,12.3], [14.5,0]]
    print np.asarray(xy)
    print '2D (u, v) coordinates (in pixels) of 2 different views of the square:'
    uv1 = [[1302,1147],[1110,976],[1411,863],[1618,1012]]
    uv2 = [[1094,1187],[1130,956],[1514,968],[1532,1187]]
    print 'uv1:'
    print np.asarray(uv1)
    print 'uv2:'
    print np.asarray(uv2)
    print ''
    print 'Use 2 views to perform a 2D calibration of the camera with 4 points of the square:'
    nd=2
    nc=2
    L1, err1 = DLTcalib(nd, xy, uv1)
    print 'Camera calibration parameters based on view #1:'
    print L1
    print 'Error of the calibration of view #1 (in pixels):'
    print err1
    L2, err2 = DLTcalib(nd, xy, uv2)
    print 'Camera calibration parameters based on view #2:'
    print L2
    print 'Error of the calibration of view #2 (in pixels):'
    print err2
    xy12 = np.zeros((len(xy),2))
    L12 = [L1,L2]
    for i in range(len(uv1)):
        xy12[i,:] = DLTrecon( nd, nc, L12, [uv1[i],uv2[i]] )
    print 'Reconstruction of the same 4 points based on 2 views and the camera calibration parameters:'
    print xy12
    print 'Mean error of the point reconstruction using the DLT (error in cm):'
    print np.mean(np.sqrt(np.sum((np.array(xy12)-np.array(xy))**2,1)))

    print ''
    print 'Use only one view to perform a 2D calibration of the camera with 4 points of the square:'
    nd=2
    nc=1
    L1, err1 = DLTcalib(nd, xy, uv1)
    print 'Camera calibration parameters based on view #1:'
    print L1
    print 'Error of the calibration of view #1 (in pixels):'
    print err1
    xy1 = np.zeros((len(xy),2))
    for i in range(len(uv1)):
        xy1[i,:] = DLTrecon( nd, nc, L1, uv1[i] )
    print 'Reconstruction of the same 4 points based on one view and the camera calibration parameters:'
    print xy1
    print 'Mean error of the point reconstruction using the DLT (error in cm):'
    print np.mean(np.sqrt(np.sum((np.array(xy1)-np.array(xy))**2,1)))
