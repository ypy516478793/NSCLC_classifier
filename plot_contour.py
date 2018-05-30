import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pydicom
import os
import numpy

Contour_file = "/home/cougarnet.uh.edu/pyuan2/Downloads/data/Lung_Data/NSCLC-Radiomics/LUNG1-001/09-18-2008-StudyID-69331/0-95085/000000.dcm"
Contour = pydicom.read_file(Contour_file)
hights = len(Contour.ROIContourSequence[0].ContourSequence)

fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
ax = Axes3D(fig)
for h in range(hights):
    d = [float(i) for i in Contour.ROIContourSequence[0].ContourSequence[h].ContourData]
    coords = []
    temp = []
    for i in range(len(d)):
        l = i % 3
        if l == 2:
            temp.append(d[i])
            coords.append(temp)
            temp = []
        else:
            temp.append(d[i])

    x, y, z = zip(*coords)
    ax.plot(x, y, z, c = 'b')

plt.show()

# middle = hights // 2
# d = [float(i) for i in Contour.ROIContourSequence[0].ContourSequence[middle].ContourData]
# coords = []
# temp = []
# for i in range(len(d)):
#     l = i % 3
#     if l == 2:
#         temp.append(d[i])
#         coords.append(temp)
#         temp = []
#     else:
#         temp.append(d[i])
#
# x, y, z = zip(*coords)
# ax.plot(x, y, z, c='b')
# ax.scatter(x, y, z, c='b')
# plt.show()

print("")