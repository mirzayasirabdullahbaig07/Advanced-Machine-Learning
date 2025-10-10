"""
# Gradient Descent (Scratch + Animation + Variants)
# Author: Yasir Abdullah

📘 Concept Overview:
-------------------
Gradient Descent (GD) is an optimization algorithm used to minimize a **cost function** 
by iteratively moving in the direction of the **negative gradient** of that function.

It’s widely used in:
- Linear Regression
- Logistic Regression
- Deep Learning (to train neural networks)
- Optimization tasks across AI/ML

---

## Why Gradient Descent is Used:
Because many ML models involve parameters (like weights and biases) that 
must be optimized to minimize error (loss). Exact analytical solutions 
are not always possible — GD finds the optimal parameters iteratively.

---

## Formula:
For any parameter θ (could be slope `m` or intercept `b`):
θ_new = θ_old - (learning_rate) * (∂J/∂θ)

Where:
- J(θ) → cost function (e.g., MSE)
- ∂J/∂θ → gradient (slope) of cost function wrt parameter
- learning_rate → controls step size

---

## Mathematical Behind Linear Regression Gradient Descent:
For simple linear regression:
    ŷ = m*x + b
Cost function (MSE):
    J = (1/N) * Σ(y - ŷ)²

Gradients:
    ∂J/∂m = (-2/N) * Σ x * (y - ŷ)
    ∂J/∂b = (-2/N) * Σ (y - ŷ)

We update both:
    m_new = m_old - α * ∂J/∂m
    b_new = b_old - α * ∂J/∂b

---

## Benefits:
✅ Works even when analytical solution (like OLS) is hard to compute  
✅ Can scale to millions of parameters (used in deep learning)  
✅ Works on any differentiable cost function  

---

## Disadvantages:
❌ Can converge to local minima (for non-convex functions)  
❌ Requires careful tuning of learning rate  
❌ Slower than direct solutions like OLS for small problems  

"""

# ===============================================================
# 🧠 IMPORTS
# ===============================================================
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ===============================================================
# 🧮 1. DATA GENERATION
# ===============================================================
X, y = make_regression(
    n_samples=4, n_features=1, n_informative=1, n_targets=1,
    noise=80, random_state=13
)

plt.scatter(X, y)
plt.title("Sample Data for Regression")
plt.show()

# ===============================================================
# 🧾 2. APPLY ORDINARY LEAST SQUARES (OLS)
# ===============================================================
reg = LinearRegression()
reg.fit(X, y)

print("OLS Coefficient (m):", reg.coef_)
print("OLS Intercept (b):", reg.intercept_)

plt.scatter(X, y)
plt.plot(X, reg.predict(X), color='red', label='OLS Fit')
plt.legend()
plt.show()

# ===============================================================
# 🔁 3. APPLY GRADIENT DESCENT FOR INTERCEPT (b)
# ===============================================================
m = 78.35  # constant slope (from OLS)
b = 100     # initial intercept
lr = 0.1    # learning rate

for i in range(3):
    loss_slope = -2 * np.sum(y - m * X.ravel() - b)
    b = b - lr * loss_slope
    print(f"Iteration {i+1}: b = {b}")

    plt.scatter(X, y)
    plt.plot(X, reg.predict(X), color='red', label='OLS')
    plt.plot(X, (m * X + b), color='#00a65a', label=f'b={b:.2f}')
    plt.legend()
    plt.show()

# ===============================================================
# 🔄 4. ITERATIVE UPDATE (b only)
# ===============================================================
b = -100
m = 78.35
lr = 0.01
epochs = 100

for i in range(epochs):
    loss_slope = -2 * np.sum(y - m * X.ravel() - b)
    b = b - (lr * loss_slope)
    y_pred = m * X + b
    plt.plot(X, y_pred, alpha=0.2, color='green')

plt.scatter(X, y)
plt.title("Gradient Descent Convergence (b only)")
plt.show()

# ===============================================================
# 🎞️ 5. ANIMATION OF GRADIENT DESCENT (b only)
# ===============================================================
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=13)
reg = LinearRegression()
reg.fit(X, y)
print("OLS → m:", reg.coef_, " | b:", reg.intercept_)

b = -150
m = 27.82
lr = 0.001
epochs = 30
all_b, all_cost = [], []

for i in range(epochs):
    slope, cost = 0, 0
    for j in range(X.shape[0]):
        slope = slope - 2 * (y[j] - (m * X[j]) - b)
        cost = cost + (y[j] - m * X[j] - b) ** 2
    b = b - (lr * slope)
    all_b.append(b)
    all_cost.append(cost)
    plt.plot(X, m * X + b, alpha=0.3)
plt.scatter(X, y)
plt.title("Gradient Descent Line Updates (b only)")
plt.show()

# ===============================================================
# 📈 6. ANIMATE COST FUNCTION
# ===============================================================
num_epochs = list(range(1, 31))
fig = plt.figure(figsize=(9, 5))
axis = plt.axes(xlim=(0, 31), ylim=(0, 2500000))
line, = axis.plot([], [], lw=2)
xdata, ydata = [], []

def animate(i):
    label = f'epoch {i+1}'
    xdata.append(num_epochs[i])
    ydata.append(all_cost[i])
    line.set_data(xdata, ydata)
    axis.set_xlabel(label)
    return line,

anim = animation.FuncAnimation(fig, animate, frames=30, repeat=False, interval=500)
plt.title("Cost Function Decrease Over Epochs")
plt.show()

# ===============================================================
# ⚙️ 7. GRADIENT DESCENT FOR m AND b (FULL)
# ===============================================================
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=13)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Scikit-learn comparison
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("Sklearn Coef (m):", lr_model.coef_)
print("Sklearn Intercept (b):", lr_model.intercept_)
print("R² Score (OLS):", r2_score(y_test, lr_model.predict(X_test)))

# ---------------------------------------------------------------
# Custom Gradient Descent Implementation
# ---------------------------------------------------------------
class GDRegressor:
    def __init__(self, learning_rate=0.001, epochs=50):
        self.m = 100
        self.b = -120
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        for i in range(self.epochs):
            loss_slope_b = -2 * np.sum(y - self.m * X.ravel() - self.b)
            loss_slope_m = -2 * np.sum((y - self.m * X.ravel() - self.b) * X.ravel())
            self.b -= self.lr * loss_slope_b
            self.m -= self.lr * loss_slope_m
        print(f"Final Parameters → m = {self.m:.3f}, b = {self.b:.3f}")

    def predict(self, X):
        return self.m * X + self.b

gd = GDRegressor(learning_rate=0.001, epochs=50)
gd.fit(X_train, y_train)
y_pred = gd.predict(X_test)

print("R² Score (GD from scratch):", r2_score(y_test, y_pred))
