import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import fileinput
import os
import shutil
import math
import random as random
from scipy.stats import loguniform
import scienceplots
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy import interpolate as interp
from scipy.optimize import newton
from scipy.interpolate import interp1d

random.seed(1)  # set random seed.
pd.set_option('display.max_columns', None)

os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

fig_PATH = "~/Desktop/Research/Figures/"
data_PATH = "~/Desktop/Caltech/Data/"

triples_catalog = pd.read_csv(data_PATH + "triples_catalog.csv")
triple_100pc = triples_catalog.query("1000/inner_star1_parallax<100 and triple_type == 'MSMS-MS'")
distances_resolvedtrip = 1000 / triple_100pc['inner_star1_parallax']

#################################
#### FUNCTIONS USED LATER #######
#################################
columns = ['sur', 'sur2', 't', 'e1','e2','g1','g2','a1','i1','i2','i','spin1h','spintot',
           'beta','vp','spin1e','spin1q','spin2e','spin2q','spin2h','htot','m1','R1','m2',
           'R2','a2','m3','Roche1','R1','type1','type2','type3','beta2','gamma','gamma2','flag']

pecaut = pd.read_csv("./Data/pecaut_mamjek.txt",sep=r'\s+')
pecaut = pecaut[pd.to_numeric(pecaut['M_G'], errors='coerce').notnull()]
pecaut = pecaut[pd.to_numeric(pecaut['Msun'], errors='coerce').notnull()]
pecaut['M_G'] = pecaut['M_G'].astype('float')
pecaut['Msun'] = pd.to_numeric(pecaut['Msun'], errors='coerce')
pecaut['M_G'] = pd.to_numeric(pecaut['M_G'], errors='coerce')

interp_func = interp1d(pecaut['M_G'], pecaut['Msun'], kind='linear', fill_value='extrapolate')

# Drop rows with NaN values
pecaut = pecaut.dropna(subset=['Msun', 'M_G'])
interp_func_inv = interp1d(pecaut['Msun'], pecaut['M_G'], kind='linear', fill_value='extrapolate')


mass_triplefraction = [[0.10574792264157319, 2.1526418786692574],
                        [0.21624767972643996, 3.718199608610547],
                        [0.42762979914903076, 6.457925636007801],
                        [0.9778928306368192, 11.937377690802336],
                        [1.1435428088501094, 13.894324853228952],
                        [1.9775000577578266, 25.24461839530332],
                        [3.9105064784322034, 36.00782778864969],
                        [6.394774621219934, 45.20547945205479],
                        [11.693948181473157, 56.947162426614476],
                        [29.242831207597824, 67.90606653620351]]
# Extract masses and triple fractions
masses_trip = [item[0] for item in mass_triplefraction]
triple_fractions = [item[1] for item in mass_triplefraction]

# Create an interpolation function
def get_triple_fraction(mass):
    interpolation_function = interp1d(masses_trip, triple_fractions, kind='linear', fill_value="extrapolate")
    return interpolation_function(mass)/100

# Extract masses and triple fractions
mass_multiplefraction =   [[0.11622294, 0.00313641*100],
                            [0.15393248, 0.00836102*100],
                            [0.20387722, 0.01684225*100],
                            [0.27002695, 0.03184672*100],
                            [0.35763955, 0.05085594*100],
                            [0.47367882, 0.07607926*100],
                            [0.62736805, 0.10397445*100],
                            [0.83092308, 0.11769751*100],
                            [1.10052331, 0.13951625*100],
                            [1.45759769, 0.15402938*100],
                            [1.93052796, 0.15998134*100],
                            [3.9105064784322034, 81.01761252446184],
                            [6.394774621219934, 89.04109589041096],
                            [11.5639657341197, 92.95499021526419],
                            [29.242831207597824, 95.89041095890411]]

masses_bin = [item[0] for item in mass_multiplefraction]
binary_fractions = [item[1] for item in mass_multiplefraction]

def get_binary_fraction(mass):
    interpolation_function = interp1d(masses_bin, binary_fractions, kind='linear', fill_value="extrapolate")
    return interpolation_function(mass)/100

def get_first_line(file_path): #get first line 
    with open(file_path) as f:
        return f.readline()

def get_last_line(file_path): #faster way to get last line
    with open(file_path, 'rb') as file:
        try:
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR)
        except OSError:
            file.seek(0)
        last_line = file.readline().decode()
        return last_line
def get_second_to_last_line(file_path):
    with open(file_path, 'r') as file:
        # Read the last two lines from the file
        lines = file.readlines()[-2:]

        # Return the second to last line
        return lines[0] if len(lines) > 1 else None

def maybe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return s

def Kepler_3rdLaw_SMA(m1,m2,SMA ): #M_sun and AU units, output in yr
    return np.power( ( (SMA**3)/(m1+m2) ) ,1./2.)

def Kepler_3rdLaw(P,m1,m2,SMA = 0,output='a'):

    '''
    Function to get semi-major axis from period using K3L
    :param P: Period of orbit in **yr**
    :return: returns SMA (a) in **AU**
    '''
    G = 1.19e-19 #G in AU^3/M_sun*s^2
    if output == 'a':
        return np.power( ( (P**2)*(m1+m2) ) ,1./3.) #Kepler's Third Law
    elif output=='p' or output=='P' : 
        return np.power( ( (np.power(SMA,3))/(m1+m2) ) ,1./2.)
    
    
        mu,sigma = 4.8, 2.3 

def Roche_limit(q):
    '''
    Function to get Roche Limit of specified mass ratios (q)
    :param q: mass ratio
    :return: returns the Roche Limit (RHS of Eqn.1 from Naoz+2014)
    '''
    num1,num2=0.49,0.6;
    return num1*np.power(q,2./3.)/(0.6* np.power(q,2./3.)+np.log(1+np.power(q,1./3.)));

def get_epsilon(a1,a2,e2):
    return (a1/a2) * e2 / (1-e2**2)
def get_epsilonM(m1,m2,a1,a2,e2):
    factor =( m1-m2) / (m1+m2)
    return factor*(a1/a2) * e2 / (1-e2**2)

def Kepler_3rdLaw_Period(m1,m2,Period ): #M_sun and AU units, output in yr
    return np.power( ( (Period**2) * (m1+m2) ) , 1./3.)

def get_t_ekl(m1,m2,m3,e2,P1,P2):
    """
    Characteristic quadrupole timescale fro EKL (eq 27 from review)
    """
    return (8./(15*np.pi))*((m1+m2+m3)/m3)*(P2*P2/P1)*np.sqrt(1-e2*e2)*(1-e2*e2)
def pos_normal(mean, sigma): 
    '''
    Function to get only positive values from normal distribution recursively
    '''
    x = np.random.normal(mean,sigma)
    return (x if x>=0 else pos_normal(mean,sigma))
def random_normal_bounds(mean, sigma, lower, upper):
    val = np.random.normal(mean,sigma)
    if lower<val<upper:
        return val
    else:
        return random_normal_bounds(mean, sigma, lower, upper)

def rndm(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g) 
 
def miller_scalo_IMF(arr,lower=0.1,upper=8.0):

    if 0.1<arr<=1: return rndm(0.1,1.0,-0.25)[0]
    if 1.<arr<=2.: return rndm(1.0,2.0,-1.00)[0]
    if 2.<arr<=10: return rndm(2.0,8.0,-1.30)[0]
    if 10<arr<=25: return rndm(10.0,125.0,-2.30)[0]

def kroupa_IMF(arr,lower=0.08,upper=8.0):

    #if arr<0.08: return rndm(0.1,1.0,0.7)[0]
    if 0.08<=arr<0.5: return rndm(0.08,0.5,-0.3)[0]
    if 0.5<=arr<150: 
        if lower > 0.5:
            return rndm(lower,upper,-1.30)[0]
        else:
            return rndm(0.5,upper,-1.30)[0]

def open_fits(path):
    hdu = fits.open(path)
    table = hdu[1].data
    df = Table(table).to_pandas()
    return df


def sampleFromSalpeter(M_min : int, M_max : int,N :int = 1, alpha : float = 2.35):
    """Sample from a Salpeter, but can be used for any power law

    Args:
        M_min (int): lower bound of mass interval
        M_max (int): upper bound of mass interval
        N (int, optional): number of samples. Defaults to 1.
        alpha (float, optional): power-law index. Defaults to 2.35 for Salpeter.

    Returns:
        float: if you only wanted one sample
        list:  if you wanted multiple samples
    """
    
    # Convert limits from M to logM.
    log_M_Min = math.log(M_min)
    log_M_Max = math.log(M_max)
    # Since Salpeter SMF decays, maximum likelihood occurs at M_min
    maxlik = math.pow(M_min, 1.0 - alpha)

    # Prepare array for output masses.
    Masses = []
    # Fill in array.
    while (len(Masses) < N):
        # Draw candidate from logM interval.
        logM = random.uniform(log_M_Min,log_M_Max)
        M    = math.exp(logM)
        # Compute likelihood of candidate from Salpeter SMF.
        likelihood = math.pow(M, 1.0 - alpha)
        # Accept randomly.
        u = random.uniform(0.0,maxlik)
        if (u < likelihood):
            Masses.append(M)
            
    return Masses if len(Masses) > 1 else Masses[0]

def sample_power_law(alpha, xmin, xmax, size=1):
    """
    Sample from a power law distribution with a given exponent (alpha) over a specified range.
    
    Parameters:
    - alpha: float. The exponent of the power law.
    - xmin: float. The minimum value of the range to sample from.
    - xmax: float. The maximum value of the range to sample from.
    - size: int, optional. The number of samples to generate. Default is 1.
    
    Returns:
    - samples: ndarray. Samples from the power law distribution.
    """
    if alpha == -1:
        # Handle the alpha = -1 case separately to avoid division by zero
        samples = np.exp(np.random.uniform(np.log(xmin), np.log(xmax), size))
    else:
        # General case for alpha != -1
        r = np.random.uniform(0, 1, size)
        factor = (xmax**(alpha+1) - xmin**(alpha+1)) * r + xmin**(alpha+1)
        samples = factor**(1/(alpha+1))
    return samples

Rsun = (695500*6.68459e-9)  #sun radius in AU

Kroupa_IMF_gen1 = np.vectorize(lambda arr: kroupa_IMF(arr,lower=0.1)) #vectorize the function
Kroupa_IMF_gen2 = np.vectorize(lambda arr: kroupa_IMF(arr,lower=18,upper=60)) #vectorize the function
x1= np.linspace(0.08,8,int(1e6))
x2 = np.linspace(18,60,int(1e6))
Kroupa_IMF_samples1 = Kroupa_IMF_gen1(x1)
Kroupa_IMF_samples2 = Kroupa_IMF_gen2(x2)

n_Kroupa1, bins_Kroupa1, patches1 = plt.hist(Kroupa_IMF_samples1,bins=5000,cumulative=False,density=True) 
# n_Kroupa2, bins_Kroupa2, patches2 = plt.hist(Kroupa_IMF_samples2,bins=5000,cumulative=False,density=True,histtype='step') 
plt.clf()

# BOOLEANS to check that parameters are **STABLE**
def get_all_criteria(a1,a2,e1,e2,m1,m2,m3,R1,R2,i,q_out):
    #stability_criteria from Mardling & Aarseth (2001),**BOOL**
    stable = 2.8 * np.power(1.+ q_out,2./5.) * np.power(1.+e2, 2./5.)*np.power(1.-e2, -6./5.)*(1-(0.3*i/180.)) 
    epsilon = (a1/a2) * e2 / (1-e2**2) #epsilon criterion from Naoz+2014
    
    Roche1=Roche_limit(m1/m2)
    Roche2=Roche_limit(m2/m1)

    stability_criteria = (a2/a1) > stable
    epsilon_criteria = epsilon < 0.1
    Roche1_criteria = R1*Rsun < 2*(a1*(1-e1)*Roche1)
    Roche2_criteria = R2*Rsun < 2*(a1*(1-e1)*Roche2)
    mass_criteria =  m1>0 and m2>0 and m3>0 #this condition should already be met bc pos_normal function
    BH_NS_no_RLO = a2 <= 1e5
    all_criteria = (stability_criteria and mass_criteria and epsilon_criteria and (Roche1_criteria and Roche2_criteria) and BH_NS_no_RLO)
    
    return all_criteria

from numpy.random import random as rndm
from scipy import interpolate
import cProfile


# Define all the distributions
def f1(x):
    return 1 - 0.5 * x

def f2(x):
    return x**(-1.3)

def f3(x):
    return x**(-0.55)

def f4(x):
    return x**(-1.7)

def f5(x):
    beta = -0.4
    qlo = 0.01
    return x**beta * (1 + beta) / (1 - qlo**(1 + beta))

def f6(x):
    return x**(-1.6)

# Define bounds for each distribution
bounds = {
    1: (0, 3.5),
    2: (0.1, 3.0),
    3: (0.15, 8.0),
    4: (1.1, 150),
    5: (0.01, 1),
    6: (50, 5e4)
}

# Mapping from which -> function
functions = {
    1: f1,
    2: f2,
    3: f3,
    4: f4,
    5: f5,
    6: f6
}

def build_inverse_cdf(which):
    f = functions[which]
    xmin, xmax = bounds[which]
    x = np.linspace(xmin, xmax, 10000)
    y = f(x)
    y = np.clip(y, 0, None)  # Ensure no negative probabilities
    cdf = np.cumsum(y)
    cdf /= cdf[-1]  # Normalize to 1
    return interp1d(cdf, x, bounds_error=False, fill_value="extrapolate")

def return_samples(N=10, which=2):
    inverse_cdf = build_inverse_cdf(which)
    u = np.random.uniform(0, 1, size=N)
    return inverse_cdf(u)

def sample_thermal_eccentricity(n_samples):
    """
    Generate samples from a thermal eccentricity distribution.
    
    Parameters:
        n_samples (int): Number of samples to generate.
    
    Returns:
        numpy.ndarray: Array of sampled eccentricities.
    """
    # Generate uniform random numbers
    u = np.random.uniform(0, 1, n_samples)
    # Apply the inverse CDF transformation
    eccentricities = np.sqrt(u)
    return eccentricities


def sample_q(n_samples, a, b1, b2, x_break, x_values = np.linspace(0, 3, 10000)):
    """
    Sample q values from a broken power law distribution.

    Parameters:
    - n_samples: Number of samples to generate.
    - a, b1, b2, x_break: Parameters of the broken power law.

    Returns:
    - samples: Array of sampled q values.
    """
    # Define the broken power law function
    def broken_power_law(x, a, b1, b2, x_break):
        return np.piecewise(x, 
                            [x < x_break, x >= x_break], 
                            [lambda x: a * (x)**b1, 
                             lambda x: a * (x_break)**(b1 - b2) * (x)**b2])

    # Generate a large number of x values to create the CDF
    pdf_values = broken_power_law(x_values, a, b1, b2, x_break)
    cdf_values = np.cumsum(pdf_values)
    cdf_values /= cdf_values[-1]  # Normalize to range [0, 1]

    # Create an inverse function (CDF⁻¹)
    inverse_cdf = interp.interp1d(cdf_values, x_values, kind="linear")

    # Sample random values from uniform [0, 1] and map to q values
    random_samples = np.random.rand(n_samples)
    samples = inverse_cdf(random_samples)

    return samples


def double_broken_power_law(x, a, b1, b2, b3, x_break1, x_break2):
    return np.piecewise(x, 
                        [x < x_break1, (x >= x_break1) & (x < x_break2), x >= x_break2], 
                        [lambda x: a * (x + 0.01)**b1, 
                            lambda x: a * (x_break1 + 0.01)**(b1 - b2) * (x + 0.01)**b2,
                            lambda x: a * (x_break1 + 0.01)**(b1 - b2) * (x_break2 + 0.01)**(b2 - b3) * (x + 0.01)**b3])

def sample_from_doublebroken(n_samples, a, b1, b2, b3, x_break1, x_break2, x_values=np.linspace(0.01, 1, 10000)):
    """
    Sample values from a double broken power law distribution.

    Parameters:
    - n_samples: Number of samples to generate.
    - a, b1, b2, b3, x_break1, x_break2: Parameters of the double broken power law.
    - x_values: Range of x values to calculate the PDF and CDF.

    Returns:
    - samples: Array of sampled values.
    """
    # Define the double broken power law function
    def double_broken_power_law(x, a, b1, b2, b3, x_break1, x_break2):
        return np.piecewise(x, 
                            [x < x_break1, (x >= x_break1) & (x < x_break2), x >= x_break2], 
                            [lambda x: a * (x + 0.01)**b1, 
                             lambda x: a * (x_break1 + 0.01)**(b1 - b2) * (x + 0.01)**b2,
                             lambda x: a * (x_break1 + 0.01)**(b1 - b2) * (x_break2 + 0.01)**(b2 - b3) * (x + 0.01)**b3])

    # Generate PDF and CDF
    pdf_values = double_broken_power_law(x_values, a, b1, b2, b3, x_break1, x_break2)
    cdf_values = np.cumsum(pdf_values)
    cdf_values /= cdf_values[-1]  # Normalize to range [0, 1]

    # Create an inverse function (CDF⁻¹)
    inverse_cdf = interp.interp1d(cdf_values, x_values, kind="linear")

    # Sample random values from uniform [0, 1] and map to the distribution
    random_samples = np.random.rand(n_samples)
    # Clip random samples to ensure they are within the range of the CDF
    random_samples = np.clip(random_samples, cdf_values[0], cdf_values[-1])
    samples = inverse_cdf(random_samples)

    return samples

def sample_log_uniform(x1, x2, size=1):
    log_x1, log_x2 = np.log10(x1), np.log10(x2)  # Convert to log-space
    samples = 10**(np.random.uniform(log_x1, log_x2, size))  # Sample uniformly in log-space
    return samples

def uniform_twin():
    if np.random.rand() < 0.18:
        # 10% of the time, sample from [0.9, 1]
        return np.random.uniform(0.95, 1.0)
    else:
        # 90% of the time, sample from [0.01, 0.9)
        return np.random.uniform(0.0, 0.95)
# Vectorized version of uniform_twin
def uniform_twin_vectorized(size=1000):
    random_values = np.random.rand(size)
    result = np.where(random_values < 0.18, 
                      np.random.uniform(0.95, 1.0, size), 
                      np.random.uniform(0.0, 0.95, size))
    return result

def absolute_to_apparent_magnitude(absolute_magnitude, distance_pc):
    """
    Convert absolute magnitude to apparent magnitude given a distance in parsecs.

    Parameters:
    - absolute_magnitude (float): Absolute magnitude.
    - distance_pc (float): Distance in parsecs.

    Returns:
    - float: Apparent magnitude.
    """
    apparent_magnitude = absolute_magnitude + 5 * (np.log10(distance_pc) - 1)
    return apparent_magnitude

def check_resolved_triple_new(sep_inner, sep_outer, m1, m2, m3, gaia_distances = distances_resolvedtrip, num_iterations = 1):
    
    resolved_counts = 0
    for _ in range(num_iterations):
        sampled_distance = np.random.choice(gaia_distances)  # Sample a distance
        G1_sample = interp_func_inv(m1)
        G2_sample = interp_func_inv(m2)
        G3_sample = interp_func_inv(m3)
        G1_sample = absolute_to_apparent_magnitude(G1_sample, sampled_distance)
        G2_sample = absolute_to_apparent_magnitude(G2_sample, sampled_distance)
        G3_sample = absolute_to_apparent_magnitude(G3_sample, sampled_distance)

        # # Calculate delta G
        deltaG_sample = np.abs(G1_sample - G2_sample)
        deltaG_sample3 = np.abs(G1_sample - G3_sample)
        deltaG_sample2 = np.abs(G2_sample - G3_sample)
        # ang_res = theta_min(deltaG_sample)
        # ang_res2 = theta_min(deltaG_sample3)
        this_theta1 = sep_inner/sampled_distance
        this_theta2 = sep_outer/sampled_distance
        is_resolved12 = is_resolved(this_theta1,deltaG_sample)
        is_resolved13 = is_resolved(this_theta2,deltaG_sample3)
        is_resolved23 = is_resolved(this_theta2,deltaG_sample2)
        
        # if (sep_inner >=ang_res*sampled_distance) and (sep_outer > ang_res2*sampled_distance):
        if is_resolved12 and is_resolved13 and is_resolved23:
            resolved_counts +=1            

    if resolved_counts> int(num_iterations/2):
        return 'Y'
    else:
        return 'N'

def add_resolved_new(df, num_iterations=1):
    for i, row in df.iterrows():
        this_a1 = row.sep1_AU
        this_a2 = row.sep2_AU
        m1, m2, m3 = row.m1, row.m2, row.m3
        # this_a2k = row.a2_after_au_kick
        # resolved_counts = 0
        resolved = 'N'
        
        resolved = check_resolved_triple_new(sep_inner = this_a1,
                                        sep_outer = this_a2, 
                                        m1 = m1, m2 = m2, m3 = m3,
                                        gaia_distances = distances_resolvedtrip,num_iterations = num_iterations)

            
            
        df.at[i,'resolved_sep'] = resolved
    return df

def sample_true_anomaly(e, num_samples):
    """Correctly sample true anomaly for eccentric orbits."""
    M = np.random.uniform(0, 2*np.pi, num_samples)  # Mean anomaly is uniform

    # Solve Kepler's equation numerically to get E (Eccentric Anomaly)
    def kepler_eq(E, M, e): return E - e * np.sin(E) - M

    E_samples = np.array([newton(kepler_eq, M_i, args=(M_i, e)) for M_i in M])

    # Convert Eccentric Anomaly to True Anomaly
    nu = 2 * np.arctan2(np.sqrt(1+e) * np.sin(E_samples/2),
                         np.sqrt(1-e) * np.cos(E_samples/2))
    return nu


def simulate_projected_separations_with_keplerian(a, e, num_samples):
    """
    Simulate observed projected separations with correct Keplerian sampling of true anomaly.
    
    Parameters:
        a (float or array): Semi-major axis in AU (or other units).
        e (float or array): Eccentricity (0 <= e < 1).
        num_samples (int): Number of random realizations per binary.

    Returns:
        projected_separations (array): Simulated projected separations.
    """
    # Sample true anomaly (nu) using Keplerian PDF
    nu = sample_true_anomaly(e, num_samples)

    # Randomize inclination and argument of periapsis
    i = np.arccos(np.random.uniform(-1, 1, num_samples))  # Inclination (cos i uniform)
    omega = np.random.uniform(0, 2 * np.pi, num_samples)  # Argument of periapsis

    # Compute physical separation r
    r = (a * (1 - e**2)) / (1 + e * np.cos(nu))

    # Compute projected separation rho
    sep = r * np.sqrt(
        np.cos(nu + omega)**2 * np.sin(i)**2 + np.sin(nu + omega)**2
    )

    return sep

def simulate_projected_separations(a, e, num_samples):
    """
    Simulate the observed projected separations for a given semi-major axis (a)
    and eccentricity (e) using random orbital orientations and phases.

    Parameters:
        a (float or array): Semi-major axis in AU (or other units).
        e (float or array): Eccentricity (0 <= e < 1).
        num_samples (int): Number of random realizations per binary.

    Returns:
        projected_separations (array): Simulated projected separations.
    """
    # Step 1: Correctly Sample True Anomaly (nu)
    nu = sample_true_anomaly(e, num_samples)  # Corrected sampling

    # Step 2: Randomly Sample Orbital Angles
    i = np.arccos(np.random.uniform(-1, 1, num_samples))  # Inclination (cos(i) uniform)
    omega = np.random.uniform(0, 2 * np.pi, num_samples)  # Argument of periapsis

    # Step 3: Compute Physical Separation (r)
    r = (a * (1 - e**2)) / (1 + e * np.cos(nu))

    # Step 4: Compute Corrected Projected Separation (ρ)
    rho = r * np.sqrt(np.cos(nu + omega)**2 * np.sin(i)**2 + np.sin(nu + omega)**2)

    return rho
# Example: Simulating a population of wide binaries

def compute_projected_separations(a1, a2, e1, e2, num_samples):
    s1 = np.array([np.median(simulate_projected_separations(a, e, num_samples)) for a, e in zip(a1, e1)])
    s2 = np.array([np.median(simulate_projected_separations(a, e, num_samples)) for a, e in zip(a2, e2)])
    return s1, s2

# triple fraction as a function of mass
mass_triplefraction = [[0.10574792264157319, 2.1526418786692574],
                        [0.21624767972643996, 3.718199608610547],
                        [0.42762979914903076, 6.457925636007801],
                        [0.9778928306368192, 11.937377690802336],
                        [1.1435428088501094, 13.894324853228952],
                        [1.9775000577578266, 25.24461839530332],
                        [3.9105064784322034, 36.00782778864969],
                        [6.394774621219934, 45.20547945205479],
                        [11.693948181473157, 56.947162426614476],
                        [29.242831207597824, 67.90606653620351]]
# Extract masses and triple fractions
masses = [item[0] for item in mass_triplefraction]
triple_fractions = [item[1] for item in mass_triplefraction]

# Create the fraction of triples for stars of with given mass
def get_triple_fraction(mass):
    interpolation_function = interp1d(masses, triple_fractions, kind='linear', fill_value="extrapolate")
    return interpolation_function(mass)


def sample_all(this_IMF = 'cosmic_constant_SFR', this_Periods = 'DM91', this_IMRD='uniform'):
    m1s,m2s,m3s,a1s,a2s,e1s,e2s = [],[],[],[],[], [], []

    for _ in range(10000):
        m1,m2,m3,R1,R2,spin1P,spin2P,beta,beta2,gamma,gamma2,a1,a2,e1,e2,g1,g2,i,t_i,age,idum = initial_conditions(IMF = this_IMF, Periods = this_Periods, IMRD=this_IMRD)
        m1s.append(m1),m2s.append(m2),m3s.append(m3)
        a1s.append(a1),a2s.append(a2)
        e1s.append(e1),e2s.append(e2)

    m1s,m2s,m3s,a1s,a2s,e1s,e2s =  np.array(m1s).astype(float), np.array(m2s).astype(float), np.array(m3s).astype(float), np.array(a1s).astype(float), np.array(a2s).astype(float), np.array(e1s).astype(float), np.array(e2s).astype(float)

    s1s, s2s = compute_projected_separations(a1 = a1s, a2 = a2s, e1 = e1s, e2 = e2s, num_samples = 1)

    # Create the DataFrame
    results_df = pd.DataFrame({
        'ID': list(range(len(m1s))),
        'm1': m1s,
        'm2': m2s,
        'm3': m3s,
        'a1': s1s,
        'a2': s2s,
        's1': s1s,
        's2': s2s,
        'e2': e2s,
    })

    results_df['sep1_AU'] = results_df['s1']
    results_df['sep2_AU'] = results_df['s2']
    results_df = add_resolved_new(results_df, num_iterations=1)
    # results_df = add_resolved(results_df, num_iterations=1)

    return results_df

from scipy.interpolate import interp1d

# Gaia sensitivity to resolving a binary; as a function of theta (angular separation) and deltaG = abs(G1-G2), difference in apparent Gaia G magnitudes
# Adopted from El-Badry 2024

# bins of deltaG
deltaG_bins = [[0,1],[1,2],[2,3],[3,4],[4,6],[6,8],[8,np.inf]]

# discrete sensitivity as a function of theta for detltaG bins in order
bin1_discrete_xy = [
    [0.24193548387096797, -0.0011210762331836932],
    [1.290322580645161, 0.9069506726457399],
    [1.8064516129032255, 0.9899103139013452],
    [2.306451612903226, 1.0100896860986546],
    [14.741935483870964, 1.0011210762331837]
]
bin2_discrete_xy = [
    [0.274193548387097, -0.0011210762331836932],
    [1.290322580645161, 0.8531390134529149],
    [1.8064516129032255, 0.9742152466367713],
    [2.322580645161289, 1.0033632286995515],
    [14.741935483870964, 0.9988789237668161]
]
bin3_discrete_xy = [
    [0.25806451612903203, -0.0011210762331836932],
    [1.290322580645161, 0.803811659192825],
    [1.8064516129032255, 0.960762331838565],
    [2.322580645161289, 1.0033632286995515],
    [14.741935483870964, 0.9988789237668161]
]
bin4_discrete_xy = [
    [0.25806451612903203, -0.0011210762331836932],
    [0.7741935483870965, 0.11995515695067271],
    [1.290322580645161, 0.6782511210762332],
    [1.8064516129032255, 0.9316143497757847],
    [2.338709677419354, 0.9966367713004484],
    [14.532258064516125, 1.0033632286995515]
]
bin5_discrete_xy = [
    [0.274193548387097, 0.0011210762331839152],
    [0.7741935483870965, 0.005605381165919132],
    [1.8064516129032255, 0.7432735426008968],
    [2.322580645161289, 0.9114349775784754],
    [2.854838709677418, 0.960762331838565],
    [3.354838709677419, 0.9831838565022422],
    [3.854838709677419, 0.9921524663677129],
    [14.532258064516125, 1.0011210762331837]
]
bin6_discrete_xy = [
    [0.24193548387096797, -0.0033632286995515237],
    [1.2741935483870965, -0.0011210762331836932],
    [1.790322580645161, 0.12443946188340815],
    [2.838709677419355, 0.6760089686098655],
    [3.354838709677419, 0.8172645739910314],
    [3.887096774193547, 0.8800448430493273],
    [4.403225806451612, 0.9226457399103138],
    [4.903225806451612, 0.9786995515695067],
    [5.435483870967741, 0.9652466367713004],
    [5.935483870967741, 0.9831838565022422],
    [6.903225806451612, 0.976457399103139],
    [7.483870967741934, 0.9988789237668161],
    [14.58064516129032, 1.0011210762331837]
]
bin7_discrete_xy = [
    [0.274193548387097, -0.0011210762331836932],
    [1.6290322580645156, -0.0011210762331836932],
    [2.677419354838709, 0.0594170403587444],
    [4.806451612903224, 0.6199551569506726],
    [6.951612903225804, 0.8800448430493273],
    [8.016129032258062, 0.9024663677130045],
    [9.096774193548386, 0.9674887892376681],
    [10.129032258064514, 0.9473094170403586],
    [12.290322580645158, 0.9899103139013452],
    [13.2258064516129, 0.9966367713004484],
    [14.499999999999996, 1.0011210762331837],
]



# Create interpolation functions for each deltaG bin
bin1_interp = interp1d([point[0] for point in bin1_discrete_xy], [point[1] for point in bin1_discrete_xy], 
                        kind='linear', fill_value='extrapolate')
bin2_interp = interp1d([point[0] for point in bin2_discrete_xy], [point[1] for point in bin2_discrete_xy], 
                        kind='linear', fill_value='extrapolate')
bin3_interp = interp1d([point[0] for point in bin3_discrete_xy], [point[1] for point in bin3_discrete_xy], 
                        kind='linear', fill_value='extrapolate')
bin4_interp = interp1d([point[0] for point in bin4_discrete_xy], [point[1] for point in bin4_discrete_xy], 
                        kind='linear', fill_value='extrapolate')
bin5_interp = interp1d([point[0] for point in bin5_discrete_xy], [point[1] for point in bin5_discrete_xy], 
                        kind='linear', fill_value='extrapolate')
bin6_interp = interp1d([point[0] for point in bin6_discrete_xy], [point[1] for point in bin6_discrete_xy], 
                        kind='linear', fill_value='extrapolate')
bin7_interp = interp1d([point[0] for point in bin7_discrete_xy], [point[1] for point in bin7_discrete_xy], 
                        kind='linear', fill_value='extrapolate')

# Function to get the probability of resolution given theta (arcsec) and delta G (app mag)
def get_resolution_probability(theta, deltaG):
    deltaG = np.abs(deltaG)
    if deltaG_bins[0][0] <= deltaG < deltaG_bins[0][1]:
        return bin1_interp(theta)
    elif deltaG_bins[1][0] <= deltaG < deltaG_bins[1][1]:
        return bin2_interp(theta)
    elif deltaG_bins[2][0] <= deltaG < deltaG_bins[2][1]:
        return bin3_interp(theta)
    elif deltaG_bins[3][0] <= deltaG < deltaG_bins[3][1]:
        return bin4_interp(theta)
    elif deltaG_bins[4][0] <= deltaG < deltaG_bins[4][1]:
        return bin5_interp(theta)
    elif deltaG_bins[5][0] <= deltaG < deltaG_bins[5][1]:
        return bin6_interp(theta)
    elif deltaG_bins[6][0] <= deltaG < deltaG_bins[6][1]:
        return bin7_interp(theta)
    else:
        return 0  # Return 0 if deltaG is outside the defined bins
    
# Function to check whether a binary is resolved given their angular resoluition (theta) and contrast (deltaG)    
def is_resolved(theta, deltaG):
    probability = get_resolution_probability(theta, deltaG)
    return np.random.uniform() < probability