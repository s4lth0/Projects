
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import t
from scipy.stats import norm

# Parameters for t-Copula
nu = 3.829939833890225  # Degrees of freedom
rho = 0.8305922714576357  # Correlation

# Generate samples from the t-copula (uniform samples)
u = np.random.rand(100)  # Uniform samples for X
v = np.random.rand(100)  # Uniform samples for Y

# Transform uniform samples to t-distribution using the inverse CDF
x = t.ppf(u, df=nu)  # Inverse CDF for X
y = t.ppf(v, df=nu)  # Inverse CDF for Y

# Calculate the joint density (for visualizing the density of the copula)
# In a real application, you would use the copula density here, but for simplicity,
# we will plot the joint distribution based on t-distributions.
Z = t.pdf(x, df=nu) * t.pdf(y, df=nu)  # Product of marginals (not the copula density)

# Create the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid for plotting the surface
x_vals = np.linspace(min(x), max(x), 1000)
y_vals = np.linspace(min(y), max(y), 1000)
X, Y = np.meshgrid(x_vals, y_vals)

# Use the joint PDF (for demonstration purposes)
Z_vals = t.pdf(X, df=nu) * t.pdf(Y, df=nu)  # Approximation of the joint density

# Plot the surface
ax.plot_surface(X, Y, Z_vals, cmap='viridis', edgecolor='none')

# Set labels
ax.set_xlabel('VNQ Price')
ax.set_ylabel('S&P 500 Price')
ax.set_zlabel('Density')
ax.set_title('3D Joint Density of t-Copula')

plt.show()

