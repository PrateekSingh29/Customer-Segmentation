import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset =  pd.read_csv("Mall_Customers.csv")
dataset.head()
X = dataset[["Spending Score (1-100)","Age"]].iloc[:,:].values
dataset.describe()

dataset.isnull().sum()

sns.countplot(y = 'Gender', data = dataset)

sns.set(style="darkgrid")
#tips = sns.load_dataset("Age")
g = sns.jointplot("Spending Score (1-100)", "Annual Income (k$)", data=dataset, kind="reg",
                   color="m", height=7)

n=0
for cols in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n +=1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.violinplot(x = cols , y = 'Gender' , data = dataset , palette = 'vlag')


from sklearn.preprocessing import LabelEncoder
LabelEncoder = LabelEncoder()
dataset["Gender"] = LabelEncoder.fit_transform(dataset["Gender"])

sns.set(style = 'whitegrid')
sns.distplot(dataset['Annual Income (k$)'])
plt.title('Distribution of Annual Income', fontsize = 20)
plt.xlabel('Range of Annual Income')
plt.ylabel('Count')
plt.show()

dataset['Age'].value_counts().plot.bar(figsize = (9, 9))


from sklearn.cluster import KMeans
# =============================================================================
# Spending Score According to Age
# =============================================================================
WCSS = []
intertia = []
for i in range(1,11):
    Kmeans = (KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10,
                    random_state = 0))
    Kmeans.fit(X)
    WCSS.append(Kmeans.inertia_)
plt.plot(np.arange(1,11),WCSS)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


Kmeans = KMeans(n_clusters = 4, init = 'k-means++',max_iter = 300, n_init = 10,
                    random_state = 0)
centroids = Kmeans.fit_predict(X)

plt.scatter(X[centroids == 0, 0], X[centroids == 0,1], c = 'red' , s = 200 )
plt.scatter(X[centroids == 1, 0], X[centroids == 1,1], s = 100, c = 'blue')
plt.scatter(X[centroids == 2, 0], X[centroids == 2,1], s = 100, c = 'cyan')
plt.scatter(X[centroids == 3, 0], X[centroids == 3,1], s = 100, c = 'green')
plt.scatter(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1],s=300, c = 'yellow')
plt.title("Spending score according to age")
plt.xlabel("Spending Score")
plt.ylabel("Age")

# =============================================================================
# Spending Score according to annual Income
# =============================================================================
X1 = dataset[["Spending Score (1-100)","Annual Income (k$)"]].iloc[:,:].values
intertia = []
WCSS = []
for i in range(1,11):
    Kmeans = (KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10,
                    random_state = 0))
    Kmeans.fit(X1)
    WCSS.append(Kmeans.inertia_)
plt.plot(np.arange(1,11),WCSS)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


Kmeans = KMeans(n_clusters = 5, init = 'k-means++',max_iter = 300, n_init = 10,
                    random_state = 0)
centroids = Kmeans.fit_predict(X1)

plt.scatter(X1[centroids == 0, 0], X1[centroids == 0,1], c = 'red' , s = 200 )
plt.scatter(X1[centroids == 1, 0], X1[centroids == 1,1], s = 100, c = 'blue')
plt.scatter(X1[centroids == 2, 0], X1[centroids == 2,1], s = 100, c = 'cyan')
plt.scatter(X1[centroids == 3, 0], X1[centroids == 3,1], s = 100, c = 'green')
plt.scatter(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1],s=300, c = 'yellow')
plt.title("Spending score according to Annual Income")
plt.xlabel("Spending Score")
plt.ylabel("Annual Income")

