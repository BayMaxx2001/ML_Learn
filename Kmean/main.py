import cv2
from sklearn.cluster import KMeans
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

c_data_path = "data/Busy/car"

def get_feature(img):
    intensity = img.sum(axis=1)
    intensity = intensity.sum(axis=0)/(255*img.shape[0] * img.shape[1])
    return intensity

def load_data(data_path=c_data_path):

    try:
        with open('data.pickle', 'rb') as handle:
            X = pickle.load(handle)
        with open('label.pickle', 'rb') as handle:
            L = pickle.load(handle)
        return X,L
    except:
        X = []
        L = []
        for file in os.listdir(data_path):
            print(file)
            c_x = get_feature(cv2.imread(os.path.join(data_path, file)))
            X.append(c_x)
            L.append(file)

        X = np.array(X)
        L = np.array(L)
        with open('data.pickle', 'wb') as handle:
            pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('label.pickle', 'wb') as handle:
            pickle.dump(L, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return X,L


#print shape of img
X,L = load_data()
plt.imshow(X)
plt.show()
print(X.shape)

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
#
#
#
#
# ### kmean
kmeans = KMeans(n_clusters=4).fit(X)
for i in range(len(kmeans.labels_)):
    print(kmeans.labels_[i]," - ", L[i])
print(kmeans.cluster_centers_)

n_row = 6
n_col = 6
for i in range(4):
    _, axs = plt.subplots(n_row, n_col, figsize=(7, 7))
    axs = axs.flatten()
    for img, ax in zip(L[ kmeans.labels_ == i][:36], axs):
        ax.imshow(mpimg.imread(os.path.join(c_data_path, img)))
    plt.tight_layout()
    plt.show()

