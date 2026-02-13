import numpy as np
import pandas as pd


def native_guide_retrieval(query, predicted_label, distance, n_neighbors, X_train, y_train):
    df = pd.DataFrame(y_train, columns = ['label'])
    df.index.name = 'index'
    df[df['label'] == 1].index.values, df[df['label'] != 1].index.values
    ts_length = X_train.shape[1]

    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric = distance)
    knn.fit(X_train[list(df[df['label'] != predicted_label].index.values)])
    dist ,ind = knn.kneighbors(query.reshape(1 ,ts_length), return_distance=True)

    return dist[0], df[df['label'] != predicted_label].index[ind[0][:]]


def findSubarray(a, k): #used to find the maximum contigious subarray of length k in the explanation weight vector
    n = len(a)
    vec=[]

    # Iterate to find all the sub-arrays
    for i in range(n-k+1):
        temp=[]
        # Store the sub-array elements in the array
        for j in range(i,i+k):
            temp.append(a[j])
        # Push the vector in the container
        vec.append(temp)

    sum_arr = []
    for v in vec:
        sum_arr.append(np.sum(v))

    return (vec[np.argmax(sum_arr)])