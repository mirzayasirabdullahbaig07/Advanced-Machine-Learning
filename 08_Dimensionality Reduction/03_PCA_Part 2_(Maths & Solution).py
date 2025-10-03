# Import necessary libraries
import numpy as np
import pandas as pd

# Fix random seed for reproducibility (so results are same every run)
np.random.seed(23) 

# ---------------------------
# STEP 0: Create synthetic 3D dataset with 2 classes
# ---------------------------

# Mean vector for class 1
mu_vec1 = np.array([0,0,0])
# Covariance matrix (identity => no correlation, equal variance)
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
# Generate 20 samples from class 1
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20)

# Put class 1 into DataFrame
df = pd.DataFrame(class1_sample,columns=['feature1','feature2','feature3'])
df['target'] = 1   # Label class 1 as "1"

# Mean vector for class 2
mu_vec2 = np.array([1,1,1])
# Same covariance matrix
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
# Generate 20 samples from class 2
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20)

# Put class 2 into DataFrame
df1 = pd.DataFrame(class2_sample,columns=['feature1','feature2','feature3'])
df1['target'] = 0   # Label class 2 as "0"

# Combine both classes
df = df.append(df1,ignore_index=True)

# Shuffle dataset
df = df.sample(40)

# View first rows
df.head()

# ---------------------------
# STEP 1: Visualize data in 3D
# ---------------------------
import plotly.express as px

# 3D scatter plot of 3 features colored by target
fig = px.scatter_3d(df, x=df['feature1'], y=df['feature2'], z=df['feature3'],
              color=df['target'].astype('str'))   # target as string for color labels

# Styling markers
fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()

# ---------------------------
# STEP 2: Standardize data (mean=0, variance=1)
# ---------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Scale only feature columns
df.iloc[:,0:3] = scaler.fit_transform(df.iloc[:,0:3])

# ---------------------------
# STEP 3: Covariance Matrix
# ---------------------------
# Compute covariance between feature1, feature2, feature3
covariance_matrix = np.cov([df.iloc[:,0],df.iloc[:,1],df.iloc[:,2]])
print('Covariance Matrix:\n', covariance_matrix)

# ---------------------------
# STEP 4: Eigen Decomposition
# ---------------------------
# Find eigenvalues and eigenvectors of covariance matrix
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

print("Eigenvalues:", eigen_values)
print("Eigenvectors:\n", eigen_vectors)

# ---------------------------
# STEP 5: Visualize Eigenvectors in 3D
# ---------------------------
%pylab inline

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

# Helper class to draw 3D arrows for eigenvectors
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

# Plot 3D scatter with eigenvectors
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

# Plot data points
ax.plot(df['feature1'], df['feature2'], df['feature3'], 'o', markersize=8, color='blue', alpha=0.2)

# Plot mean point in red
ax.plot([df['feature1'].mean()], [df['feature2'].mean()], [df['feature3'].mean()], 
        'o', markersize=10, color='red', alpha=0.5)

# Plot each eigenvector as arrow
for v in eigen_vectors.T:
    a = Arrow3D([df['feature1'].mean(), v[0]], 
                [df['feature2'].mean(), v[1]], 
                [df['feature3'].mean(), v[2]], 
                mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)

ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')
plt.title('Eigenvectors')
plt.show()

# ---------------------------
# STEP 6: Project data onto principal components
# ---------------------------
# Select top 2 eigenvectors (PC1 and PC2)
pc = eigen_vectors[0:2]
print(pc)

# Transform original data into new subspace (dot product with PCs)
transformed_df = np.dot(df.iloc[:,0:3],pc.T)

# Create new dataframe with PCs
new_df = pd.DataFrame(transformed_df,columns=['PC1','PC2'])
new_df['target'] = df['target'].values
new_df.head()

# ---------------------------
# STEP 7: Visualize PCA results in 2D
# ---------------------------
new_df['target'] = new_df['target'].astype('str')

# 2D scatter plot of new PCA features
fig = px.scatter(x=new_df['PC1'],
                 y=new_df['PC2'],
                 color=new_df['target'],
                 color_discrete_sequence=px.colors.qualitative.G10)

# Styling markers
fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()
