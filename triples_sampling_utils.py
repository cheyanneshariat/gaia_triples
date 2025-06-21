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

Rsun = (695500*6.68459e-9)  #sun radius in AU

columns = ['sur', 'sur2', 't', 'e1','e2','g1','g2','a1','i1','i2','i','spin1h','spintot',
           'beta','vp','spin1e','spin1q','spin2e','spin2q','spin2h','htot','m1','R1','m2',
           'R2','a2','m3','Roche1','R1','type1','type2','type3','beta2','gamma','gamma2','flag']

# End of basic Imports and Definitions
# -------------------------------

# mass-Mg interpolation from Pecaut & Mamjek empirical table
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

# Triple Fraction of Binaries from Offner+2023 Table 1
mass_triplefraction = [
       [0.1125, (2.2/19.)*100], # Winters+2019
       [0.225, (3.6/23.)*100], # Winters+2019
       [0.45, (6.3/30.)*100], # Winters+2019
       [1.0, (12./46.)*100], # Raghavan+2010
       [1.175, ((14.)/47.)*100], # Tokovinin 2014
       [2.0, ((25)/68.)*100], # Moe and Kratter 2021
       [4.0, (36./81.)*100], # Moe and Di Stefano 2017
       [6.5, (45./89.)*100], # Moe and Di Stefano 2017
       [12.0, (57./93.)*100], # Moe and Di Stefano 2017
       [33.0, (68./96.)*100]] # Sana et al (2012,2014)

# Extract masses and triple fractions
masses_trip = [item[0] for item in mass_triplefraction]
triple_fractions = [item[1] for item in mass_triplefraction]

# Create an interpolation function
def get_triple_fraction(mass):
    interpolation_function = interp1d(masses_trip, triple_fractions, kind='linear', fill_value="extrapolate")
    return interpolation_function(mass)/100

def get_outer_eccentricity(e_out):
    if e_out == 'uniform':
        e2  = np.random.uniform(0,1)
    elif e_out == 'thermal':
        e2  = sample_thermal_eccentricity(n_samples = 1)[0]
    return e2

def sample_log_uniform(x1, x2, size=1):
    log_x1, log_x2 = np.log10(x1), np.log10(x2)  # Convert to log-space
    samples = 10**(np.random.uniform(log_x1, log_x2, size))  # Sample uniformly in log-space
    return samples

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

def open_fits(path):
    hdu = fits.open(path)
    table = hdu[1].data
    df = Table(table).to_pandas()
    return df


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


# BOOLEANS to check that parameters are **STABLE**
def get_all_criteria(a1,a2,e1,e2,m1,m2,m3,R1,R2,i,q_out,simple=False):
    if simple:
        P1 = Kepler_3rdLaw_SMA(m1,m2,a1)
        P2 = Kepler_3rdLaw_SMA(m1+m2,m3,a2)
        all_criteria = ((P2/P1) > 5)
    else:
        #stability_criteria from Mardling & Aarseth (2001),**BOOL**
        stable = 2.8 * np.power(1.+ q_out,2./5.) * np.power(1.+e2, 2./5.)*np.power(1.-e2, -6./5.)*(1-(0.3*i/180.)) 
        epsilon = (a1/a2) * e2 / (1-e2**2) #epsilon criterion from Naoz+2014
        
        Roche1=Roche_limit(m1/m2)
        Roche2=Roche_limit(m2/m1)

        stability_criteria = (a2/a1) > stable
        epsilon_criteria = epsilon < 0.1 # hierarchical
        Roche1_criteria = R1*Rsun < 2*(a1*(1-e1)*Roche1) # not immediately near roche limit
        Roche2_criteria = R2*Rsun < 2*(a1*(1-e1)*Roche2) # not immediately near roche limit
        mass_criteria =  m1>0 and m2>0 and m3>0 #this condition should already be met bc pos_normal function
        BH_NS_no_RLO = a2 <= 1e5 #galactic tides super dominant
        all_criteria = (stability_criteria and mass_criteria and epsilon_criteria and (Roche1_criteria and Roche2_criteria) and BH_NS_no_RLO)
    
    
    return all_criteria


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

def check_resolved_triple_new(sep_inner, sep_outer, m1, m2, m3, gaia_distances, num_iterations = 1):
    
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

def add_resolved_new(df, gaia_distances, num_iterations=1):
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
                                        gaia_distances = gaia_distances,num_iterations = num_iterations)

            
            
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


from scipy.interpolate import interp1d

# Gaia sensitivity to resolving a binary; as a function of theta (angular separation) and deltaG = abs(G1-G2), difference in apparent Gaia G magnitudes
# Adopted from El-Badry 2024

# bins of deltaG
deltaG_bins = [[0,1],[1,2],[2,3],[3,4],[4,6],[6,8],[8,np.inf]]

# sensitivity as a function of theta for detltaG bins in order


# From El-Badry 2024, with no cuts  (Uncomment if using all triples, not just those with bp_rp colors)
# bin1_discrete_xy = [
#     [0.24193548387096797, 0.0011210762331836932],
#     [1.290322580645161, 0.9069506726457399],
#     [1.8064516129032255, 0.9899103139013452],
#     [2.306451612903226, 1.0100896860986546],
#     [14.741935483870964, 1.0011210762331837]
# ]
# bin2_discrete_xy = [
#     [0.274193548387097, 0.0011210762331836932],
#     [1.290322580645161, 0.8531390134529149],
#     [1.8064516129032255, 0.9742152466367713],
#     [2.322580645161289, 1.0033632286995515],
#     [14.741935483870964, 0.9988789237668161]
# ]
# bin3_discrete_xy = [
#     [0.25806451612903203, 0.0011210762331836932],
#     [1.290322580645161, 0.803811659192825],
#     [1.8064516129032255, 0.960762331838565],
#     [2.322580645161289, 1.0033632286995515],
#     [14.741935483870964, 0.9988789237668161]
# ]
# bin4_discrete_xy = [
#     [0.25806451612903203, 0.0011210762331836932],
#     [0.7741935483870965, 0.11995515695067271],
#     [1.290322580645161, 0.6782511210762332],
#     [1.8064516129032255, 0.9316143497757847],
#     [2.338709677419354, 0.9966367713004484],
#     [14.532258064516125, 1.0033632286995515]
# ]
# bin5_discrete_xy = [
#     [0.274193548387097, 0.0011210762331839152],
#     [0.7741935483870965, 0.005605381165919132],
#     [1.8064516129032255, 0.7432735426008968],
#     [2.322580645161289, 0.9114349775784754],
#     [2.854838709677418, 0.960762331838565],
#     [3.354838709677419, 0.9831838565022422],
#     [3.854838709677419, 0.9921524663677129],
#     [14.532258064516125, 1.0011210762331837]
# ]
# bin6_discrete_xy = [
#     [0.24193548387096797, 0.0033632286995515237],
#     [1.2741935483870965, 0.0011210762331836932],
#     [1.790322580645161, 0.12443946188340815],
#     [2.838709677419355, 0.6760089686098655],
#     [3.354838709677419, 0.8172645739910314],
#     [3.887096774193547, 0.8800448430493273],
#     [4.403225806451612, 0.9226457399103138],
#     [4.903225806451612, 0.9786995515695067],
#     [5.435483870967741, 0.9652466367713004],
#     [5.935483870967741, 0.9831838565022422],
#     [6.903225806451612, 0.976457399103139],
#     [7.483870967741934, 0.9988789237668161],
#     [14.58064516129032, 1.0011210762331837]
# ]
# bin7_discrete_xy = [
#     [0.274193548387097, 0.0011210762331836932],
#     [1.6290322580645156, 0.0011210762331836932],
#     [2.677419354838709, 0.0594170403587444],
#     [4.806451612903224, 0.6199551569506726],
#     [6.951612903225804, 0.8800448430493273],
#     [8.016129032258062, 0.9024663677130045],
#     [9.096774193548386, 0.9674887892376681],
#     [10.129032258064514, 0.9473094170403586],
#     [12.290322580645158, 0.9899103139013452],
#     [13.2258064516129, 0.9966367713004484],
#     [14.499999999999996, 1.0011210762331837],
# ]


# From El-Badry 2024, with BP_RP color cuts
bin1_discrete_xy = [
    [0.23972602739725968, 0.0011848341232227888],
    [0.7705479452054789, 0.2665876777251184],
    [1.2842465753424657, 0.3684834123222749],
    [1.8150684931506849, 0.4466824644549763],
    [2.3287671232876717, 0.9016587677725119],
    [2.825342465753425, 0.9656398104265402],
    [3.339041095890411, 0.9822274881516587],
    [3.8869863013698627, 1.0248815165876777],
    [4.417808219178082, 1.0154028436018958],
    [8.86986301369863, 1.0011848341232228], 
]

bin2_discrete_xy = [
    [0.2739726027397258, 0.0011848341232227888],
    [0.8047945205479454, 0.01540284360189581],
    [1.2842465753424657, 0.04383886255924163],
    [1.8150684931506849, 0.12677725118483418],
    [2.3116438356164384, 0.764218009478673],
    [2.825342465753425, 0.8803317535545023],
    [3.8869863013698627, 1.0035545023696681],
    [4.3493150684931505, 1.0082938388625593],
    [10.256849315068493, 1.0035545023696681]
]

bin3_discrete_xy = [
    [0.2910958904109586, 0.0011848341232227888],
    [0.8047945205479454, 0.010663507109004877],
    [1.301369863013699, 0.0390995260663507],
    [1.8150684931506849, 0.12677725118483418],
    [2.3116438356164384, 0.7405213270142179],
    [2.859589041095891, 0.8684834123222749],
    [3.339041095890411, 0.9182464454976303],
    [3.8698630136986303, 0.9964454976303317],
    [4.40068493150685, 1.0106635071090047],
    [7.859589041095892, 1.0035545023696681]
]

bin4_discrete_xy = [
    [0.2739726027397258, 0.0011848341232227888],
    [0.7876712328767121, 0.005924170616113722],
    [1.2842465753424657, 0.034360189573459765],
    [1.7979452054794525, 0.1244075829383886],
    [2.3287671232876717, 0.7381516587677726],
    [2.8424657534246576, 0.8708530805687204],
    [3.3732876712328768, 0.9206161137440758],
    [3.8869863013698627, 0.9988151658767772],
    [4.40068493150685, 1.0106635071090047],
    [8.287671232876715, 1.0059241706161137]
]

bin5_discrete_xy = [
    [0.2739726027397258, 0.0011848341232227888],
    [1.25, 0.010663507109004877],
    [1.7979452054794525, 0.07227488151658767],
    [2.3116438356164384, 0.6481042654028436],
    [2.8424657534246576, 0.8234597156398105],
    [3.8698630136986303, 0.9869668246445498],
    [4.417808219178082, 1.0082938388625593],
    [4.914383561643835, 1.0130331753554502],
    [10.325342465753426, 1.0059241706161137]
    ]

bin6_discrete_xy = [
    [0.25684931506849296, -0.0011848341232227888],
    [1.7979452054794525, 0.005924170616113722],
    [2.8424657534246576, 0.5627962085308057],
    [3.8869863013698627, 0.8732227488151658],
    [4.40068493150685, 0.9040284360189573],
    [4.914383561643835, 0.9845971563981043],
    [5.4452054794520555, 0.9656398104265402],
    [5.941780821917809, 0.9964454976303317],
    [6.523972602739728, 0.9964454976303317],
    [6.986301369863015, 0.9869668246445498],
    [7.5342465753424674, 1.0059241706161137],
    [12.825342465753426, 1.0011848341232228]
]

bin7_discrete_xy = [
    [0.25684931506849296, -0.0035545023696683664],
    [1.60958904109589, 0.0011848341232227888],
    [2.6883561643835616, 0.04383886255924163],
    [4.845890410958906, 0.6078199052132702],
    [6.969178082191781, 0.8708530805687204],
    [8.047945205479454, 0.8992890995260664],
    [9.126712328767123, 0.9727488151658767],
    [10.222602739726028, 0.9490521327014217],
    [11.3013698630137, 0.9751184834123223],
    [14.434931506849317, 1.0011848341232228]
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