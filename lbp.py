import numpy as np

def lbp_3D(image):
    z = image.shape[0]
    w = image.shape[1]
    h = image.shape[2]
    texture_matrix = np.zeros([z, w, h])
    for k in range(1, z-1):
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                lbp = 0
                if image[k + 1, i - 1, j - 1] < image[k, i, j]:
                    a1 = 0
                else:
                    a1 = 1
                lbp = lbp + a1 * 2 ** 0
                if image[k + 1, i - 1, j + 1] < image[k, i, j]:
                    a2 = 0
                else:
                    a2 = 1
                lbp = lbp + a2 * 2 ** 1
                if image[k - 1, i - 1, j + 1] < image[k, i, j]:
                    a3 = 0
                else:
                    a3 = 1
                lbp = lbp + a3 * 2 ** 2
                if image[k - 1, i - 1, j - 1] < image[k, i, j]:
                    a4 = 0
                else:
                    a4 = 1
                lbp = lbp + a4 * 2 ** 3
                if image[k + 1, i + 1, j - 1] < image[k, i, j]:
                    a5 = 0
                else:
                    a5 = 1
                lbp = lbp + a5 * 2 ** 4
                if image[k + 1, i + 1, j + 1] < image[k, i, j]:
                    a6 = 0
                else:
                    a6 = 1
                lbp = lbp + a6 * 2 ** 5
                if image[k - 1, i + 1, j + 1] < image[k, i, j]:
                    a7 = 0
                else:
                    a7 = 1
                lbp = lbp + a7 * 2 ** 6
                if image[k - 1, i + 1, j - 1] < image[k, i, j]:
                    a8 = 0
                else:
                    a8 = 1
                lbp = lbp + a8 * 2 ** 7
                texture_matrix[k, i, j] = lbp
    return texture_matrix

def lbp_2D(image):
    w = image.shape[0]
    h = image.shape[1]
    texture_matrix=np.zeros([w, h])
    for i in range(1, w-1):
        for j in range(1, h-1):
            lbp=0
            if image[i-1,j-1]<image[i,j]:
                a1=0
            else:
                a1=1
            lbp = lbp + a1*2**0
            if image[i-1,j]<image[i,j]:
                a2=0
            else:
                a2=1
            lbp = lbp + a2*2**1
            if image[i-1,j+1]<image[i,j]:
                a3=0
            else:
                a3=1
            lbp = lbp + a3*2**2
            if image[i,j+1]<image[i,j]:
                a4=0
            else:
                a4=1
            lbp = lbp + a4*2**3
            if image[i+1,j+1]<image[i,j]:
                a5=0
            else:
                a5=1
            lbp = lbp + a5*2**4
            if image[i+1,j]<image[i,j]:
                a6=0
            else:
                a6=1
            lbp = lbp + a6*2**5
            if image[i+1,j-1]<image[i,j]:
                a7=0
            else:
                a7=1
            lbp = lbp + a7*2**6
            if image[i,j-1]<image[i,j]:
                a8=0
            else:
                a8=1
            lbp = lbp + a8*2**7
            texture_matrix[i,j]=lbp
    return texture_matrix

def lbp_3D_MV(image, s):
    z = image.shape[0]
    w = image.shape[1]
    h = image.shape[2]
    texture_matrix = np.zeros([z, w, h])
    for k in range(s, z-s):
        for i in range(s, w - s):
            for j in range(s, h - s):
                lbp = 0
                if image[k + s, i - s, j - s] < image[k, i, j]:
                    a1 = 0
                else:
                    a1 = 1
                lbp = lbp + a1 * 2 ** 0
                if image[k + s, i - s, j + s] < image[k, i, j]:
                    a2 = 0
                else:
                    a2 = 1
                lbp = lbp + a2 * 2 ** 1
                if image[k - s, i - s, j + s] < image[k, i, j]:
                    a3 = 0
                else:
                    a3 = 1
                lbp = lbp + a3 * 2 ** 2
                if image[k - s, i - s, j - s] < image[k, i, j]:
                    a4 = 0
                else:
                    a4 = 1
                lbp = lbp + a4 * 2 ** 3
                if image[k + s, i + s, j - s] < image[k, i, j]:
                    a5 = 0
                else:
                    a5 = 1
                lbp = lbp + a5 * 2 ** 4
                if image[k + s, i + s, j + s] < image[k, i, j]:
                    a6 = 0
                else:
                    a6 = 1
                lbp = lbp + a6 * 2 ** 5
                if image[k - s, i + s, j + s] < image[k, i, j]:
                    a7 = 0
                else:
                    a7 = 1
                lbp = lbp + a7 * 2 ** 6
                if image[k - s, i + s, j - s] < image[k, i, j]:
                    a8 = 0
                else:
                    a8 = 1
                lbp = lbp + a8 * 2 ** 7
                texture_matrix[k, i, j] = lbp
    return texture_matrix

def lbp_3D_ri(image):
    z, w, h = image.shape
    texture_matrix = np.zeros([z, w, h])
    image2 = np.zeros([z + 2, w + 2, h + 2])
    image2[1:z + 1, 1:w + 1, 1:h + 1] = image
    image = image2
    for k in range(1, z+1):
        for i in range(1, w+1):
            for j in range(1, h+1):
                lbp = 0
                code = np.zeros([26])
                if image[k - 1, i - 1, j - 1] > image[k, i, j]:
                    code[0] = 1
                if image[k - 1, i - 1, j ] > image[k, i, j]:
                    code[1] = 1
                if image[k - 1, i - 1, j + 1] > image[k, i, j]:
                    code[2] = 1
                if image[k - 1, i, j + 1] > image[k, i, j]:
                    code[3] = 1
                if image[k - 1, i + 1, j + 1] > image[k, i, j]:
                    code[4] = 1
                if image[k - 1, i + 1, j] > image[k, i, j]:
                    code[5] = 1
                if image[k - 1, i + 1, j - 1] > image[k, i, j]:
                    code[6] = 1
                if image[k - 1, i, j - 1] > image[k, i, j]:
                    code[7] = 1
                if k - 2 >= 0:
                    if image[k - 2, i, j] > image[k, i, j]:
                        code[8] = 1
                if image[k, i - 1, j - 1] > image[k, i, j]:
                    code[9] = 1
                if image[k, i - 1, j ] > image[k, i, j]:
                    code[10] = 1
                if image[k, i - 1, j + 1] > image[k, i, j]:
                    code[11] = 1
                if image[k, i, j + 1] > image[k, i, j]:
                    code[12] = 1
                if image[k, i + 1, j + 1] > image[k, i, j]:
                    code[13] = 1
                if image[k, i + 1, j] > image[k, i, j]:
                    code[14] = 1
                if image[k, i + 1, j - 1] > image[k, i, j]:
                    code[15] = 1
                if image[k, i, j - 1] > image[k, i, j]:
                    code[16] = 1
                if image[k + 1, i - 1, j - 1] > image[k, i, j]:
                    code[17] = 1
                if image[k + 1, i - 1, j ] > image[k, i, j]:
                    code[18] = 1
                if image[k + 1, i - 1, j + 1] > image[k, i, j]:
                    code[19] = 1
                if image[k + 1, i, j + 1] > image[k, i, j]:
                    code[20] = 1
                if image[k + 1, i + 1, j + 1] > image[k, i, j]:
                    code[21] = 1
                if image[k + 1, i + 1, j] > image[k, i, j]:
                    code[22] = 1
                if image[k + 1, i + 1, j - 1] > image[k, i, j]:
                    code[23] = 1
                if image[k + 1, i, j - 1] > image[k, i, j]:
                    code[24] = 1
                if k+2<=65:
                    if image[k + 2, i, j] > image[k, i, j]:
                        code[25] = 1
                        lbp = lbp + 1
                t = 0
                prev = code[-1]
                for p in range(0, len(code)):
                    cur = code[p]
                    if cur != prev:
                        t += 1
                    prev = cur
                if t<=3:
                    A = list(code[0:8])
                    B = list(code[9:17])
                    C = list(code[17:25])
                    for x in (A, B, C):
                        value = []
                        temp = x[0] * 1 + x[1] * 2 + x[2] * 4 + x[3] * 8 + x[4] * 16 + x[5] * 32 + x[6] * 64 + x[7] * 128
                        value.append(temp)
                        x.insert(0, x.pop())
                        temp = x[0] * 1 + x[1] * 2 + x[2] * 4 + x[3] * 8 + x[4] * 16 + x[5] * 32 + x[6] * 64 + x[7] * 128
                        value.append(temp)
                        x.insert(0, x.pop())
                        temp = x[0] * 1 + x[1] * 2 + x[2] * 4 + x[3] * 8 + x[4] * 16 + x[5] * 32 + x[6] * 64 + x[7] * 128
                        value.append(temp)
                        x.insert(0, x.pop())
                        temp = x[0] * 1 + x[1] * 2 + x[2] * 4 + x[3] * 8 + x[4] * 16 + x[5] * 32 + x[6] * 64 + x[7] * 128
                        value.append(temp)
                        x.insert(0, x.pop())
                        temp = x[0] * 1 + x[1] * 2 + x[2] * 4 + x[3] * 8 + x[4] * 16 + x[5] * 32 + x[6] * 64 + x[7] * 128
                        value.append(temp)
                        x.insert(0, x.pop())
                        temp = x[0] * 1 + x[1] * 2 + x[2] * 4 + x[3] * 8 + x[4] * 16 + x[5] * 32 + x[6] * 64 + x[7] * 128
                        value.append(temp)
                        x.insert(0, x.pop())
                        temp = x[0] * 1 + x[1] * 2 + x[2] * 4 + x[3] * 8 + x[4] * 16 + x[5] * 32 + x[6] * 64 + x[7] * 128
                        value.append(temp)
                        x.insert(0, x.pop())
                        temp = x[0] * 1 + x[1] * 2 + x[2] * 4 + x[3] * 8 + x[4] * 16 + x[5] * 32 + x[6] * 64 + x[7] * 128
                        value.append(temp)
                        value = min(value)
                        lbp = lbp + value
                    lbp = code[8]+code[25]
                    texture_matrix[k - 1, i - 1, j - 1] = lbp
                else:
                    texture_matrix[k-1, i-1, j-1] = 9
    return texture_matrix

def lbp_3D_new1(image):
    z, w, h = image.shape
    texture_matrix = np.zeros([z, w, h])
    for k in range(1, z - 1):
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                lbp = 0
                if image[k - 1, i, j] < image[k, i, j]:
                    a1 = 0
                else:
                    a1 = 1
                    lbp = lbp + a1 * 1
                if image[k, i - 1, j] < image[k, i, j]:
                    a2 = 0
                else:
                    a2 = 1
                    lbp = lbp + a2 * 2
                if image[k, i, j + 1] < image[k, i, j]:
                    a3 = 0
                else:
                    a3 = 1
                    lbp = lbp + a3 * 4
                if image[k, i + 1, j] < image[k, i, j]:
                    a4 = 0
                else:
                    a4 = 1
                    lbp = lbp + a4 * 8
                if image[k, i, j - 1] < image[k, i, j]:
                    a5 = 0
                else:
                    a5 = 1
                    lbp = lbp + a5 * 16
                if image[k + 1, i, j] < image[k, i, j]:
                    a6 = 0
                else:
                    a6 = 1
                    lbp = lbp + a6 * 32
                texture_matrix[k, i, j] = lbp
    return texture_matrix

def lbp_3D_sphere1(image):
    z = image.shape[0]
    w = image.shape[1]
    h = image.shape[2]
    texture_matrix = np.zeros([z, w, h])
    for k in range(1, z - 1):
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                lbp = 0
                if image[k - 1, i, j] < image[k, i, j]:
                    a1 = 0
                else:
                    a1 = 1
                lbp = lbp + a1 * 5
                if image[k, i - 1, j - 1] < image[k, i, j]:
                    a2 = 0
                else:
                    a2 = 1
                lbp = lbp + a2 * 10
                if image[k, i - 1, j ] < image[k, i, j]:
                    a3 = 0
                else:
                    a3 = 1
                lbp = lbp + a3 * 15
                if image[k, i - 1, j + 1] < image[k, i, j]:
                    a4 = 0
                else:
                    a4 = 1
                lbp = lbp + a4 * 20
                if image[k, i, j + 1] < image[k, i, j]:
                    a5 = 0
                else:
                    a5 = 1
                lbp = lbp + a5 * 25
                if image[k, i + 1, j + 1] < image[k, i, j]:
                    a6 = 0
                else:
                    a6 = 1
                lbp = lbp + a6 * 30
                if image[k, i + 1, j] < image[k, i, j]:
                    a7 = 0
                else:
                    a7 = 1
                lbp = lbp + a7 * 35
                if image[k, i + 1, j - 1] < image[k, i, j]:
                    a8 = 0
                else:
                    a8 = 1
                lbp = lbp + a8 * 40
                if image[k, i, j - 1] < image[k, i, j]:
                    a9 = 0
                else:
                    a9 = 1
                lbp = lbp + a9 * 45
                if image[k + 1, i, j] < image[k, i, j]:
                    a10 = 0
                else:
                    a10 = 1
                lbp = lbp + a10 * 50
                texture_matrix[k, i, j] = lbp
    return texture_matrix

def lbp_3D_sphere2(image):
    z = image.shape[0]
    w = image.shape[1]
    h = image.shape[2]
    texture_matrix = np.zeros([z, w, h])
    for k in range(2, z - 2):
        for i in range(2, w - 2):
            for j in range(2, h - 2):
                lbp = 0
                if image[k - 2, i , j ] < image[k, i, j]:
                    a1 = 0
                else:
                    a1 = 1
                lbp = lbp + a1 * 1

                if image[k - 1, i - 1, j - 1] < image[k, i, j]:
                    a2 = 0
                else:
                    a2 = 1
                lbp = lbp + a2 * 2
                if image[k - 1, i - 1, j ] < image[k, i, j]:
                    a3 = 0
                else:
                    a3 = 1
                lbp = lbp + a3 * 3
                if image[k - 1, i - 1, j + 1] < image[k, i, j]:
                    a4 = 0
                else:
                    a4 = 1
                lbp = lbp + a4 * 4
                if image[k - 1, i, j + 1] < image[k, i, j]:
                    a5 = 0
                else:
                    a5 = 1
                lbp = lbp + a5 * 5
                if image[k - 1, i + 1, j + 1] < image[k, i, j]:
                    a6 = 0
                else:
                    a6 = 1
                lbp = lbp + a6 * 6
                if image[k - 1, i + 1, j] < image[k, i, j]:
                    a7 = 0
                else:
                    a7 = 1
                lbp = lbp + a7 * 7
                if image[k - 1, i + 1, j - 1] < image[k, i, j]:
                    a8 = 0
                else:
                    a8 = 1
                lbp = lbp + a8 * 8
                if image[k - 1, i, j - 1] < image[k, i, j]:
                    a9 = 0
                else:
                    a9 = 1
                lbp = lbp + a9 * 9

                if image[k, i - 2, j - 2] < image[k, i, j]:
                    a10 = 0
                else:
                    a10 = 1
                lbp = lbp + a10 * 10
                if image[k, i - 2, j - 1] < image[k, i, j]:
                    a11 = 0
                else:
                    a11 = 1
                lbp = lbp + a11 * 11
                if image[k, i - 2, j] < image[k, i, j]:
                    a12 = 0
                else:
                    a12 = 1
                lbp = lbp + a12 * 12
                if image[k, i - 2, j + 1] < image[k, i, j]:
                    a13 = 0
                else:
                    a13 = 1
                lbp = lbp + a13 * 13
                if image[k, i - 2, j + 2] < image[k, i, j]:
                    a14 = 0
                else:
                    a14 = 1
                lbp = lbp + a14 * 14
                if image[k, i - 1, j - 2] < image[k, i, j]:
                    a15 = 0
                else:
                    a15 = 1
                lbp = lbp + a15 * 15
                if image[k, i - 1, j + 2] < image[k, i, j]:
                    a16 = 0
                else:
                    a16 = 1
                lbp = lbp + a16 * 16
                if image[k, i , j - 2] < image[k, i, j]:
                    a17 = 0
                else:
                    a17 = 1
                lbp = lbp + a17 * 17
                if image[k, i , j + 2] < image[k, i, j]:
                    a18 = 0
                else:
                    a18 = 1
                lbp = lbp + a18 * 18
                if image[k, i + 1, j - 2] < image[k, i, j]:
                    a19 = 0
                else:
                    a19 = 1
                lbp = lbp + a19 * 19
                if image[k, i + 1, j + 2] < image[k, i, j]:
                    a20 = 0
                else:
                    a20 = 1
                lbp = lbp + a20 * 20
                if image[k, i + 2, j - 2 ] < image[k, i, j]:
                    a21 = 0
                else:
                    a21 = 1
                lbp = lbp + a21 * 21
                if image[k, i + 2, j - 1] < image[k, i, j]:
                    a22 = 0
                else:
                    a22 = 1
                lbp = lbp + a22 * 22
                if image[k, i + 2 , j] < image[k, i, j]:
                    a23 = 0
                else:
                    a23 = 1
                lbp = lbp + a23 * 23
                if image[k, i + 2 , j + 1] < image[k, i, j]:
                    a24 = 0
                else:
                    a24 = 1
                lbp = lbp + a24 * 24
                if image[k, i + 2 , j + 2] < image[k, i, j]:
                    a25 = 0
                else:
                    a25 = 1
                lbp = lbp + a25 * 25

                if image[k + 1, i - 1, j - 1] < image[k, i, j]:
                    a26 = 0
                else:
                    a26 = 1
                lbp = lbp + a26 * 26
                if image[k + 1, i - 1, j ] < image[k, i, j]:
                    a27 = 0
                else:
                    a27 = 1
                lbp = lbp + a27 * 27
                if image[k + 1, i - 1, j + 1] < image[k, i, j]:
                    a28 = 0
                else:
                    a28 = 1
                lbp = lbp + a28 * 28
                if image[k + 1, i, j + 1] < image[k, i, j]:
                    a29 = 0
                else:
                    a29 = 1
                lbp = lbp + a29 * 29
                if image[k + 1, i + 1, j + 1] < image[k, i, j]:
                    a30 = 0
                else:
                    a30 = 1
                lbp = lbp + a30 * 30
                if image[k + 1, i + 1, j] < image[k, i, j]:
                    a31 = 0
                else:
                    a31 = 1
                lbp = lbp + a31 * 31
                if image[k + 1, i + 1, j - 1] < image[k, i, j]:
                    a32 = 0
                else:
                    a32 = 1
                lbp = lbp + a32 * 32
                if image[k + 1, i, j - 1] < image[k, i, j]:
                    a33 = 0
                else:
                    a33 = 1
                lbp = lbp + a33 * 33

                if image[k + 2, i, j] < image[k, i, j]:
                    a34 = 0
                else:
                    a34 = 1
                lbp = lbp + a34 * 34
                texture_matrix[k, i, j] = lbp
    return texture_matrix


def lbp_3D_fast(image):
    z, w, h = image.shape
    texture_matrix = np.zeros([z, w, h])
    for k in range(1, z - 1):
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                lbp = 0
                if image[k - 1, i - 1, j - 1] >= image[k, i, j]:
                    lbp = lbp + 1
                if image[k - 1, i - 1, j ] >= image[k, i, j]:
                    lbp = lbp + 2
                if image[k - 1, i - 1, j + 1] >= image[k, i, j]:
                    lbp = lbp + 3
                if image[k - 1, i, j + 1] >= image[k, i, j]:
                    lbp = lbp + 4
                if image[k - 1, i + 1, j + 1] >= image[k, i, j]:
                    lbp = lbp + 5
                if image[k - 1, i + 1, j] >= image[k, i, j]:
                    lbp = lbp + 6
                if image[k - 1, i + 1, j - 1] >= image[k, i, j]:
                    lbp = lbp + 7
                if image[k - 1, i, j - 1] >= image[k, i, j]:
                    lbp = lbp + 8
                if image[k - 1, i, j] >= image[k, i, j]:
                    lbp = lbp + 9
                if image[k, i - 1, j - 1] >= image[k, i, j]:
                    lbp = lbp + 10
                if image[k, i - 1, j ] >= image[k, i, j]:
                    lbp = lbp + 11
                if image[k, i - 1, j + 1] >= image[k, i, j]:
                    lbp = lbp + 12
                if image[k, i, j + 1] >= image[k, i, j]:
                    lbp = lbp + 13
                if image[k, i + 1, j + 1] >= image[k, i, j]:
                    lbp = lbp + 14
                if image[k, i + 1, j] >= image[k, i, j]:
                    lbp = lbp + 15
                if image[k, i + 1, j - 1] >= image[k, i, j]:
                    lbp = lbp + 16
                if image[k, i, j - 1] >= image[k, i, j]:
                    lbp = lbp + 17
                if image[k + 1, i - 1, j - 1] >= image[k, i, j]:
                    lbp = lbp + 18
                if image[k + 1, i - 1, j ] >= image[k, i, j]:
                    lbp = lbp + 19
                if image[k + 1, i - 1, j + 1] >= image[k, i, j]:
                    lbp = lbp + 20
                if image[k + 1, i, j + 1] >= image[k, i, j]:
                    lbp = lbp + 21
                if image[k + 1, i + 1, j + 1] >= image[k, i, j]:
                    lbp = lbp + 22
                if image[k + 1, i + 1, j] >= image[k, i, j]:
                    lbp = lbp + 23
                if image[k + 1, i + 1, j - 1] >= image[k, i, j]:
                    lbp = lbp + 24
                if image[k + 1, i, j - 1] >= image[k, i, j]:
                    lbp = lbp + 25
                if image[k + 1, i, j] >= image[k, i, j]:
                    lbp = lbp + 26
                texture_matrix[k, i, j] = lbp
    return texture_matrix

def lbp_3D_robust(image):
    z, w, h = image.shape
    texture_matrix = np.zeros([z, w, h])
    for k in range(1, z - 1):
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                lbp = 0
                averge=(image[k, i, j]*6+image[k - 1, i, j]+image[k, i - 1, j]+image[k, i, j + 1]+image[k, i + 1, j]+image[k, i, j - 1]+image[k + 1, i, j])/12
                averge=np.uint8(averge)
                if image[k - 1, i, j] >= averge:
                    lbp = lbp + 1
                if image[k, i - 1, j] >= averge:
                    lbp = lbp + 2
                if image[k, i, j + 1] >= averge:
                    lbp = lbp + 4
                if image[k, i + 1, j] >= averge:
                    lbp = lbp + 8
                if image[k, i, j - 1] >= averge:
                    lbp = lbp + 16
                if image[k + 1, i, j] >= averge:
                    lbp = lbp + 32
                texture_matrix[k, i, j] = lbp
    return texture_matrix

def lbp_3D_robust_M2(image):
    z, w, h = image.shape
    texture_matrix = np.zeros([z, w, h])
    for k in range(1, z - 2):
        for i in range(1, w - 2):
            for j in range(1, h - 2):
                lbp = 0
                averge=(image[k, i, j]*6+image[k - 1, i, j]+image[k, i - 1, j]+image[k, i, j + 1]+image[k, i + 1, j]+image[k, i, j - 1]+image[k + 1, i, j])/12
                averge=np.uint8(averge)
                p1=(image[k - 1, i, j]*6+image[k - 2, i, j]+image[k, i, j]+image[k-1, i, j + 1]+image[k-1, i, j-1]+image[k-1, i+1, j]+image[k-1, i-1, j])/12
                p1=np.uint8(p1)
                if p1 >= averge:
                    lbp = lbp + 1
                p2 = (image[k, i - 1, j] * 6 + image[k - 1, i-1, j] + image[k+1, i-1, j] + image[k, i, j] + image[
                    k, i-2, j] + image[k, i-1, j+1] + image[k, i - 1, j-1]) / 12
                p2 = np.uint8(p2)
                if p2 >= averge:
                    lbp = lbp + 2
                p3 = (image[k, i, j + 1] * 6 + image[k+1, i, j + 1] + image[k-1, i, j + 1] + image[k, i+1, j + 1] + image[
                    k, i - 1, j+1] + image[k, i, j + 2] + image[k, i, j]) / 12
                p3 = np.uint8(p3)
                if p3 >= averge:
                    lbp = lbp + 4
                p4 = (image[k, i + 1, j] * 6 + image[k+1, i + 1, j] + image[k-1, i + 1, j] + image[k, i + 2, j] + image[k, i, j] + image[k, i + 1, j+1] + image[k, i + 1, j-1]) / 12
                p4 = np.uint8(p4)
                if p4 >= averge:
                    lbp = lbp + 8
                p5 = (image[k, i, j - 1] * 6 + image[k+1, i, j - 1] + image[k-1, i, j - 1] + image[k, i+1, j - 1] +
                      image[k, i-1, j - 1] + image[k, i, j] + image[k, i, j - 2]) / 12
                p5 = np.uint8(p5)
                if p5 >= averge:
                    lbp = lbp + 16
                p6 = (image[k + 1, i, j] * 6 + image[k + 2, i, j] + image[k, i, j] + image[k + 1, i+1, j] +
                      image[k + 1, i-1, j] + image[k + 1, i, j+1] + image[k + 1, i, j]-1) / 12
                p6 = np.uint8(p6)
                if p6 >= averge:
                    lbp = lbp + 32
                texture_matrix[k, i, j] = lbp
    return texture_matrix

def lbp_3D_robust_HR(image, s):
    z, w, h = image.shape
    texture_matrix = np.zeros([z, w, h])
    image2 = np.zeros([z + 4 * s, w + 4 * s, h + 4 * s])
    image2[2 * s:z + 2 * s, 2 * s:w + 2 * s, 2 * s:h + 2 * s] = image
    for k in range(2*s, z + 2*s):
        for i in range(2*s, w + 2*s):
            for j in range(2*s, h + 2*s):
                lbp = 0
                averge=(image2[k, i, j]*6+image2[k - s, i, j]+image2[k, i - s, j]+image2[k, i, j + s]+image2[k, i + s, j]+image2[k, i, j - s]+image2[k + s, i, j])/12
                averge=np.uint8(averge)
                p1=(image2[k - s, i, j]*6+image2[k - 2*s, i, j]+image2[k, i, j]+image2[k-s, i, j + s]+image2[k-s, i, j-s]+image2[k-s, i+s, j]+image2[k-s, i-s, j])/12
                p1=np.uint8(p1)
                if p1 >= averge:
                    lbp = lbp + 1
                p2 = (image2[k, i - s, j] * 6 + image2[k - s, i-s, j] + image2[k+s, i-s, j] + image2[k, i, j] + image2[
                    k, i-2*s, j] + image2[k, i-s, j+s] + image2[k, i - s, j-s]) / 12
                p2 = np.uint8(p2)
                if p2 >= averge:
                    lbp = lbp + 2
                p3 = (image2[k, i, j + s] * 6 + image2[k+s, i, j + s] + image2[k-s, i, j + s] + image2[k, i+s, j + s] + image2[
                    k, i - s, j+s] + image2[k, i, j + 2*s] + image2[k, i, j]) / 12
                p3 = np.uint8(p3)
                if p3 >= averge:
                    lbp = lbp + 4
                p4 = (image2[k, i + s, j] * 6 + image2[k+s, i + s, j] + image2[k-s, i + s, j] + image2[k, i + 2*s, j] + image2[k, i, j] + image2[k, i + s, j+s] + image2[k, i + s, j-s]) / 12
                p4 = np.uint8(p4)
                if p4 >= averge:
                    lbp = lbp + 8
                p5 = (image2[k, i, j - s] * 6 + image2[k+s, i, j - s] + image2[k-s, i, j - s] + image2[k, i+s, j - s] +
                      image2[k, i-s, j - s] + image2[k, i, j] + image2[k, i, j - 2*s]) / 12
                p5 = np.uint8(p5)
                if p5 >= averge:
                    lbp = lbp + 16
                p6 = (image2[k + s, i, j] * 6 + image2[k + 2*s, i, j] + image2[k, i, j] + image2[k + s, i+s, j] +
                      image2[k + s, i-s, j] + image2[k + s, i, j+s] + image2[k + s, i, j-s]) / 12
                p6 = np.uint8(p6)
                if p6 >= averge:
                    lbp = lbp + 32
                texture_matrix[k-2*s, i-2*s, j-2*s] = lbp

    return texture_matrix

def lbp_3D_robust_C(image):
    z, w, h = image.shape
    texture_matrix = np.zeros([z, w, h])
    aver_grey = np.mean(image)
    for k in range(1, z - 1):
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                averge=(image[k, i, j]*6+image[k - 1, i, j]+image[k, i - 1, j]+image[k, i, j + 1]+image[k, i + 1, j]+image[k, i, j - 1]+image[k + 1, i, j])/12
                averge=np.uint8(averge)
                if aver_grey >= averge:
                    texture_matrix[k, i, j] = 0
                else:
                    texture_matrix[k, i, j] = 1
    return texture_matrix

def MS_lbp_3D_robust(image, s):
    z, w, h = image.shape
    texture_matrix = np.zeros([z, w, h])
    for k in range(1, z - s):
        for i in range(1, w - s):
            for j in range(1, h - s):
                lbp = 0
                averge=(image[k, i, j]*6+image[k - s, i, j]+image[k, i - s, j]+image[k, i, j + s]+image[k, i + s, j]+image[k, i, j - s]+image[k + s, i, j])/12
                averge=np.uint8(averge)
                if image[k - s, i, j] >= averge:
                    lbp = lbp + 1
                if image[k, i - s, j] >= averge:
                    lbp = lbp + 2
                if image[k, i, j + s] >= averge:
                    lbp = lbp + 4
                if image[k, i + s, j] >= averge:
                    lbp = lbp + 8
                if image[k, i, j - s] >= averge:
                    lbp = lbp + 16
                if image[k + s, i, j] >= averge:
                    lbp = lbp + 32
                texture_matrix[k, i, j] = lbp
    return texture_matrix

def MS_lbp_3D_robust_U(image, s):
    z, w, h = image.shape
    texture_matrix = np.zeros([z, w, h])
    image2 = np.zeros([z+2*s, w+2*s, h+2*s])
    image2[s:z+s, s:w+s, s:h+s] = image
    for k in range(s, z+s):
        for i in range(s, w+s):
            for j in range(s, h+s):
                code = np.array([0, 0, 0, 0, 0, 0])
                averge=(image2[k, i, j]*6+image2[k - s, i, j]+image2[k, i - s, j]+image2[k, i, j + s]+image2[k, i + s, j]+image2[k, i, j - s]+image2[k + s, i, j])/12
                averge=np.uint8(averge)
                if image2[k - s, i, j] >= averge:
                    code[0] = 1
                if image2[k, i - s, j] >= averge:
                    code[1] = 1
                if image2[k, i, j + s] >= averge:
                    code[2] = 1
                if image2[k, i + s, j] >= averge:
                    code[3] = 1
                if image2[k, i, j - s] >= averge:
                    code[4] = 1
                if image2[k + s, i, j] >= averge:
                    code[5] = 1
                texture_matrix[k-s, i-s, j-s] = 1*code[0]+2*code[1]+4*code[2]+8*code[3]+16*code[4]+32*code[5]

    return texture_matrix

def CLBP_3D_robust(image, s):

    z, w, h = image.shape
    aver_grey = np.mean(image)
    magnitude_matrix = np.zeros([z-2, w-2, h-2, 6])
    texture_matrix = np.zeros([z, w, h])
    texture_matrix2 = np.zeros([z, w, h])
    texture_matrix3 = np.zeros([z, w, h])

    for k in range(s, z - s):
        for i in range(s, w - s):
            for j in range(s, h - s):
                total = np.sum(image[k - 1:k + 1, i - 1:i + 1, j - 1:j + 1])
                texture_matrix[k, i, j] = (total + image[k, i, j] * 25) / 52
    image[s:z-s, s:w-s, s:h-s] = texture_matrix[s:z-s, s:w-s, s:h-s]
    texture_matrix = np.zeros([z, w, h])

    for k in range(s, z - s):
        for i in range(s, w - s):
            for j in range(s, h - s):
                magnitude_matrix[k - s, i - s, j - s, 0] = abs(image[k - s, i, j] - image[k, i, j])
                magnitude_matrix[k - s, i - s, j - s, 1] = abs(image[k, i - s, j] - image[k, i, j])
                magnitude_matrix[k - s, i - s, j - s, 2] = abs(image[k, i, j + s] - image[k, i, j])
                magnitude_matrix[k - s, i - s, j - s, 3] = abs(image[k, i + s, j] - image[k, i, j])
                magnitude_matrix[k - s, i - s, j - s, 4] = abs(image[k, i, j - s] - image[k, i, j])
                magnitude_matrix[k - s, i - s, j - s, 5] = abs(image[k + s, i, j] - image[k, i, j])
    mp_aver = np.mean(magnitude_matrix)

    for k in range(s, z - s):
        for i in range(s, w - s):
            for j in range(s, h - s):
                lbp = 0
                if magnitude_matrix[k - s, i - s, j - s, 0] >= mp_aver:
                    lbp = lbp + 1
                if magnitude_matrix[k - s, i - s, j - s, 1] >= mp_aver:
                    lbp = lbp + 2
                if magnitude_matrix[k - s, i - s, j - s, 2] >= mp_aver:
                    lbp = lbp + 4
                if magnitude_matrix[k - s, i - s, j - s, 3] >= mp_aver:
                    lbp = lbp + 8
                if magnitude_matrix[k - s, i - s, j - s, 4] >= mp_aver:
                    lbp = lbp + 16
                if magnitude_matrix[k - s, i - s, j - s, 5] >= mp_aver:
                    lbp = lbp + 32
                texture_matrix[k, i, j] = lbp

                lbp = 0
                averge = (image[k, i, j] * 6 + image[k - s, i, j] + image[k, i - s, j] + image[k, i, j + s] + image[
                    k, i + s, j] + image[k, i, j - s] + image[k + s, i, j]) / 12
                if image[k - s, i, j] >= averge:
                    lbp = lbp + 1
                if image[k, i - s, j] >= averge:
                    lbp = lbp + 2
                if image[k, i, j + s] >= averge:
                    lbp = lbp + 4
                if image[k, i + s, j] >= averge:
                    lbp = lbp + 8
                if image[k, i, j - s] >= averge:
                    lbp = lbp + 16
                if image[k + s, i, j] >= averge:
                    lbp = lbp + 32
                texture_matrix2[k, i, j] = lbp

                if image[k, i, j] >= aver_grey:
                    texture_matrix3[k, i, j] = 1

    lbp1 = texture_matrix.flatten()
    max_bins = 64
    hist1, bins = np.histogram(lbp1, normed=True, bins=max_bins, range=(0, max_bins))

    lbp2 = texture_matrix2.flatten()
    hist2, bins = np.histogram(lbp2, normed=True, bins=max_bins, range=(0, max_bins))

    lbp3 = texture_matrix3.flatten()
    max_bins = 2
    hist3, bins = np.histogram(lbp3, normed=True, bins=max_bins, range=(0, max_bins))

    hist = np.hstack((hist1, hist2))
    hist = np.hstack((hist, hist3))

    return hist

def MS_ELBP_3D_robust(image, s):
    z, w, h = image.shape
    texture_matrix = np.zeros([z, w, h])
    image2 = np.zeros([z + 2 * s, w + 2 * s, h + 2 * s])
    image2[s:z + s, s:w + s, s:h + s] = image
    for k in range(s, z + s):
        for i in range(s, w + s):
            for j in range(s, h + s):
                averge = (image2[k, i, j] * 6 + image2[k - s, i, j] + image2[k, i - s, j] + image2[k, i, j + s] +
                          image2[k, i + s, j] + image2[k, i, j - s] + image2[k + s, i, j]) / 12
                texture_matrix[k - s, i - s, j - s] = np.uint8(averge)
    mean_voxel = np.mean(texture_matrix)

    texture_matrix1 = np.zeros([z, w, h])
    texture_matrix2 = np.zeros([z, w, h])
    texture_matrix3 = np.zeros([z, w, h])
    image2 = np.zeros([z+2*s, w+2*s, h+2*s])
    image2[s:z+s, s:w+s, s:h+s] = texture_matrix
    for k in range(s, z+s):
        for i in range(s, w+s):
            for j in range(s, h+s):
                if image2[k, i, j] >= mean_voxel:
                    texture_matrix1[k - s, i - s, j - s] = 1
                code = np.array([0, 0, 0, 0, 0, 0])
                averge=(image2[k - s, i, j]+image2[k, i - s, j]+image2[k, i, j + s]+image2[k, i + s, j]+image2[k, i, j - s]+image2[k + s, i, j])/6
                averge=np.uint8(averge)
                if image2[k - s, i, j] >= averge:
                    code[0] = 1
                if image2[k, i - s, j] >= averge:
                    code[1] = 1
                if image2[k, i, j + s] >= averge:
                    code[2] = 1
                if image2[k, i + s, j] >= averge:
                    code[3] = 1
                if image2[k, i, j - s] >= averge:
                    code[4] = 1
                if image2[k + s, i, j] >= averge:
                    code[5] = 1
                texture_matrix2[k-s, i-s, j-s] = 1*code[0]+2*code[1]+4*code[2]+8*code[3]+16*code[4]+32*code[5]

                if k>s and k<z and i>s and i<w and j>s and j<h:
                    if image2[k - 2*s, i, j] >= image2[k - s, i, j]:
                        code[0] = 1
                    if image2[k, i - 2*s, j] >= image2[k, i - s, j]:
                        code[1] = 1
                    if image2[k, i, j + 2*s] >= image2[k, i, j + s]:
                        code[2] = 1
                    if image2[k, i + 2*s, j] >= image2[k, i + s, j]:
                        code[3] = 1
                    if image2[k, i, j - 2*s] >= image2[k, i, j - s]:
                        code[4] = 1
                    if image2[k + 2*s, i, j] >= image2[k + s, i, j]:
                        code[5] = 1
                    texture_matrix3[k-s, i-s, j-s] = 1*code[0]+2*code[1]+4*code[2]+8*code[3]+16*code[4]+32*code[5]

    lbp1 = texture_matrix1.flatten()
    max_bins = 2
    hist1, bins = np.histogram(lbp1, normed=True, bins=max_bins, range=(0, max_bins))

    lbp2 = texture_matrix2.flatten()
    max_bins = 64
    hist2, bins = np.histogram(lbp2, normed=True, bins=max_bins, range=(0, max_bins))

    lbp3 = texture_matrix3.flatten()
    max_bins = 64
    hist3, bins = np.histogram(lbp3, normed=True, bins=max_bins, range=(0, max_bins))

    hist = np.hstack((hist1, hist2))
    hist = np.hstack((hist, hist3))

    return hist

