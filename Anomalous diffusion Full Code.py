import numpy as np
import pandas as pd

np.random.seed(327)

R_values = [10, 100, 1000, 10000, 100000]

results = []

for R in R_values:
    # Generate R pseudo-random numbers
    u_r = np.random.random(R)
    
    # Compute mean, expectation, and standard deviation
    u_mean = np.mean(u_r)
    u_square_mean = np.mean(u_r**2)
    u_std = np.sqrt(u_square_mean - u_mean**2)
    
    results.append([R, u_mean, u_square_mean, u_std])

# Convert results to DataFrame
df_results = pd.DataFrame(results, columns=["R", "Mean (u_R)", "Expectation (u²_R)", "Std dev (σ_R)"])

print(df_results)


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(327)

R_values = [10, 100, 1000, 10000, 100000]

mu = 0.5

var_results = []

# Compute the average of each R with 100 experiments
for R in R_values:
    squared_diffs = []
    for _ in range(100): 
        u_r = np.random.random(R) 
        u_mean = np.mean(u_r)  
        squared_diffs.append((u_mean - mu) ** 2)  
    
    expected_value = np.mean(squared_diffs)  
    var_results.append([R, expected_value, R * expected_value]) 

var_results = np.array(var_results)

R_values = var_results[:, 0]
expected_variance = var_results[:, 1]
scaled_variance = var_results[:, 2]

# Draw the figure：R × ⟨(ū_R - μ)²⟩ vs. R
plt.figure(figsize=(8, 6))
plt.plot(R_values, scaled_variance, 'o-', label=r'$R \times \langle (\bar{u}_R - \mu)^2 \rangle$', color='orange')
plt.axhline(y=1/12, color='r', linestyle='--', label="σ² (1/12)")
plt.xscale('log') 
plt.xlabel("R")
plt.ylim(0.00, 0.17)
plt.ylabel(r"$R \times \langle (\bar{u}_R - \mu)^2 \rangle$")
plt.title("Convergence of Variance Estimation")
plt.legend()
plt.grid()
plt.show()


import numpy as np

np.random.seed(327)

R = 1000000

u_samples = np.random.random(R)

x_samples = np.arccos(1 - 2 * u_samples) / np.pi

mean_x = np.mean(x_samples)
std_x = np.std(x_samples)

# Round to 5 significant figures for comparison
mean_x_rounded = round(mean_x, 5)
std_x_rounded = round(std_x, 5)

print("Numerical Mean (μ):", mean_x_rounded)
print("Numerical Std Dev (σ):", std_x_rounded)


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(327)

R = 1000000

u_samples = np.random.random(R)

x_samples = np.arccos(1 - 2 * u_samples) / np.pi

# Define p(x)
def p_x_analytical(x):
    return (np.pi / 2) * np.sin(np.pi * x)

# Create histogram of the sampled x values
plt.figure(figsize=(8, 6))
bins = 51 
plt.hist(x_samples, bins=bins, density=True, alpha=0.6, color='b', label="Numerical Results")


x_vals = np.linspace(0, 1, 500) 
plt.plot(x_vals, p_x_analytical(x_vals), 'r-', label=r"Exact $p(x) = \frac{\pi}{2} \sin(\pi x)$", linewidth=2)

plt.xlabel("x")
plt.ylabel("Density")
plt.title("Sampled x  vs. Exact p(x)")
plt.legend()
plt.grid()
plt.show()


import numpy as np

np.random.seed(327)

N = 100  
R_values = [10, 100, 1000, 10000, 100000] 

def generate_random_steps(size):
    u_samples = np.random.random(size)  
    return np.arccos(1 - 2 * u_samples) / np.pi  

mean_results = []
std_results = []

for R in R_values:
    # Generate R 100-step random walks
    walks = np.array([np.sum(generate_random_steps(N)) for _ in range(R)])
    
    # Compute mean and standard deviation
    mean_X_R = np.mean(walks)
    std_X_R = np.sqrt(np.mean(walks**2) - mean_X_R**2) 
    
    mean_results.append(round(mean_X_R, 5))
    std_results.append(round(std_X_R, 5))

mean_results = np.array(mean_results)
std_results = np.array(std_results)

# Predictions
mu_exact = N / 2  
sigma_exact = np.sqrt(1/2 - 2/np.pi**2) * np.sqrt(N) 

print("Numerical Means:", mean_results)
print("Numerical Standard Deviations:", std_results)
print("Mean:", round(mu_exact, 5))
print("Standard Deviation:", round(sigma_exact, 5))


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(327)

R = 10000  
N_values = np.arange(1, 501)  

def generate_random_steps(size):
    u_samples = np.random.random(size)  
    return np.arccos(1 - 2 * u_samples) / np.pi  

diffusion_lengths = []

# Compute ℓ₂(N)_R for each N
for N in N_values:
    walks = np.array([np.sum(generate_random_steps(N)) for _ in range(R)])
    
    mean_X_R = np.mean(walks)
    std_X_R = np.sqrt(np.mean(walks**2) - mean_X_R**2) 
    
    diffusion_lengths.append(std_X_R)

diffusion_lengths = np.array(diffusion_lengths)

plt.figure(figsize=(8, 6))
plt.plot(N_values, diffusion_lengths, label=r"$ℓ2(N)_R$", color='b', linewidth=2)
plt.plot(N_values, np.sqrt(1/2 - 2/np.pi**2) * np.sqrt(N_values), 'r--', label=r"$0.2176/Sqrt[N]$")

plt.xlabel("N (Number of Steps)")
plt.ylabel(r"$ℓ2(N)_R$")
plt.title("Diffusion Constant: $ℓ2(N)_R$ vs. N")
plt.legend()
plt.grid()
plt.show()


# Use log transformation to fit ℓ₂(N)_R
log_N = np.log(N_values)
log_L2 = np.log(diffusion_lengths)

fit_output = np.polyfit(log_N, log_L2, 1)
D_fit = fit_output[-2]  
C_fit_log = fit_output[-1] 

C_fit = np.exp(C_fit_log)

print(f"Fitted D: {C_fit:.6f}")
print(f"Fitted C: {D_fit:.6f}")

fit_curve = C_fit * N_values**D_fit

plt.figure(figsize=(8, 6))
plt.plot(N_values, diffusion_lengths, linestyle='None', marker='.', label="Numerical Results", color='b')
plt.plot(N_values, fit_curve, 'r-', label="Fitted Curve", linewidth=2)
plt.plot(N_values, 0.2176 * np.sqrt(N_values), 'g--', label=r"$0.2176\sqrt{N}$")

plt.xlabel("N (Number of Steps)")
plt.ylabel(r"$\ell_2(N)_R$")
plt.title("Power-Law Fit: $\ell_2(N)_R$ vs. N")
plt.legend(loc="lower right")
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(327)

R = 1000000
u_samples = np.random.random(R)  
x_samples = np.tan(np.pi * (u_samples - 0.5)) / 2  # Inverse transform for Cauchy-Lorentz distribution

bins = np.arange(-4.0, 4.0, 8.0 / 201.0)

# Compute the Cauchy-Lorentz pdf
x_values = np.linspace(-4, 4, 500)
pdf_values = 2 / (np.pi * (1 + 4 * x_values**2))  

plt.figure(figsize=(8, 6))
plt.hist(x_samples, bins=bins, density=True, alpha=0.6, color='b', label="{x_r}")

plt.plot(x_values, pdf_values, 'r-', label="Cauchy-Lorentz pdf")

plt.xlabel(r"$x$")
plt.ylabel(r"$p_C(x)$")
plt.title("Histogram of {x_r} ")
plt.legend()
plt.grid()
plt.show()


import numpy as np
import pandas as pd

np.random.seed(327)

N = 100  
R_values = [100, 1000, 10000, 100000]  

# Inverse transform sampling to generate random step sizes of the Cauchy-Lorentz distribution
def inverse_transform_cauchy(u):
    return 0.5 * np.tan(np.pi * (u - 0.5)) # F(u) in Exercise 4

walks_results = {}

# Generate R N-step random walks
for R in R_values:
    Xr = np.zeros(R)  # Initialize the start position of each random walks
    for step in range(N):
        u_samples = np.random.random(R)  
        step_sizes = inverse_transform_cauchy(u_samples)  # Transform to Cauchy-Lorentz distribution step sizes
        Xr += step_sizes 

    walks_results[R] = Xr

# Display the final positions of random walks for different values of R separately
for R in R_values:
    df_walks = pd.DataFrame(walks_results[R], columns=[f"Xr(N) for R={R}"])
    print(f"Random walk final position Xr(N) (R={R}):")
    print(df_walks.head(11))  # too long, only display first 11
    print("\n" + "="*50 + "\n")


    theta_values = [0.1, 0.5, 0.9] 
diffusion_results = []

for R in R_values:
    Xr = walks_results[R] 
    
    theta_results = []
    for theta in theta_values:
        len_theta = np.power(np.mean(np.power(np.abs(Xr), theta)), 1 / theta)  # Compute the difussion length
        theta_results.append(len_theta)
    
    diffusion_results.append([R] + theta_results)


df_diffusion = pd.DataFrame(diffusion_results, columns=["R", "θ=0.1", "θ=0.5", "θ=0.9"])
print(df_diffusion)


import matplotlib.pyplot as plt

np.random.seed(327)

R = 10000  
N_values = np.arange(1, 251) 
theta_values = [0.1, 0.5, 0.9]  

def inverse_transform_cauchy(u):
    return 0.5 * np.tan(np.pi * (u - 0.5))

# Store the difussion length ℓθ(N)_R
diffusion_results = {theta: [] for theta in theta_values}

# Compute ℓθ(N)_R
for N in N_values:
    Xr = np.zeros(R)  
    for step in range(N):
        u_samples = np.random.random(R)  
        step_sizes = inverse_transform_cauchy(u_samples)
        Xr += step_sizes  
    
    # Compute difussion lengths under different θ values
    for theta in theta_values:
        len_theta = np.power(np.mean(np.power(np.abs(Xr), theta)), 1 / theta) 
        diffusion_results[theta].append(len_theta)

# Plot the figure
plt.figure(figsize=(8, 5))
for theta in theta_values:
    plt.plot(N_values, diffusion_results[theta], label=f"ℓθ(N)_R for θ={theta}")

plt.xlabel("N (Number of Steps)")
plt.ylabel("ℓθ(N)_R")
plt.title("Anomalous Diffusive Exponent: ℓθ(N)_R vs. N")
plt.legend()
plt.grid()
plt.show()


# Take the log of both sides to get back to a linear relation: log(ℓθ(N)_R) = log(D) + α * log(N)
fit_results_log = {}

for theta in theta_values:
    logN = np.log(N_values)  
    logL = np.log(diffusion_results[theta]) 
    
    output = np.polyfit(logN, logL, 1)
    alpha_fit = output[-2]  
    D_fit = np.exp(output[-1])  

    fit_results_log[theta] = {"D": D_fit, "alpha": alpha_fit}


    # Plot data points
    plt.plot(N_values, diffusion_results[theta], linestyle='None', marker=".", label=f"Data (θ={theta})")
    
    # Generate the fit line
    fit_N = np.linspace(1, 250, 100)
    fit_L = D_fit * np.power(fit_N, alpha_fit)
    plt.plot(fit_N, fit_L, 'r', label=f"Fit (θ={theta})")

plt.xlabel("N (Number of Steps)")
plt.ylabel("ℓθ(N)_R")
plt.title("Fitted Anomalous Diffusive Exponent: ℓθ(N)_R vs. N")
plt.legend()
plt.grid()
plt.show()


df_fit_log = pd.DataFrame.from_dict(fit_results_log, orient="index", columns=["D", "alpha"])
print(df_fit_log)