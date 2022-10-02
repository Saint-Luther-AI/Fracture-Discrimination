import numpy as np

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
