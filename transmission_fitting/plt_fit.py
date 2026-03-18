import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.rm'] = 'Times New Roman'  
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
rcParams["figure.dpi"] = 300

def read_original_negf_data(filename, fit_E_min=-3.0, fit_E_max=4.0, max_y_gap=0.001):
    data = []
    with open(filename, 'r') as f:
        for _ in range(3):
            f.readline()
        for line_num, line in enumerate(f, 4):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    E_val = float(parts[0])  #  E
                    T_val = float(parts[1])  # NEGF transmission
                    data.append([E_val, T_val])
                except Exception as e:
                    print(f"NEGF data in the {line_num} row is wrong ({e}), pass")
    if not data:
        raise ValueError("no valid NEGF data")
    
    data = np.array(data, dtype=np.float32)
    E_raw, T_raw = data[:, 0], data[:, 1]
    mask = (E_raw >= fit_E_min) & (E_raw <= fit_E_max)
    E_original, T_original = E_raw[mask], T_raw[mask]
    print(f"Read NEGF data：{fit_E_min}~{fit_E_max} eV, there are {len(E_original)} points in total")
    
    E_dense, T_dense = dense_sample_by_y(E_original, T_original, max_y_gap)
    print(f"After densifing data：There are {len(E_dense)} points in total")
    return E_dense, T_dense
    
def dense_sample_by_y(E_original, T_original, max_y_gap=0.01):
    E_dense = []
    T_dense = []
    n_original = len(E_original)
    
    for i in range(n_original - 1):
        x1, y1 = E_original[i], T_original[i]
        x2, y2 = E_original[i+1], T_original[i+1]
        y_gap = abs(y2 - y1)
        
        if y_gap <= max_y_gap:
            E_dense.append(x1)
            T_dense.append(y1)
            continue
        
        n_insert = int(np.ceil(y_gap / max_y_gap)) - 1
        x_insert = np.linspace(x1, x2, n_insert + 2)[:-1]
        y_insert = np.linspace(y1, y2, n_insert + 2)[:-1]
        E_dense.extend(x_insert)
        T_dense.extend(y_insert)
    
    E_dense.append(E_original[-1])
    T_dense.append(T_original[-1])
    return np.array(E_dense, dtype=np.float32), np.array(T_dense, dtype=np.float32)

def read_para_params(param_filename):
    peak_params = []
    with open(param_filename, 'r') as f:
        f.readline()
        f.readline()
        for line_num, line in enumerate(f, 3):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) != 6:
                print(f"There is a parameter format error on line {line_num} (it should have 6 columns), so it has been skipped.")
                continue
            try:
                e_center = float(parts[2])
                theta_deg = float(parts[3])
                gamma_r = float(parts[4])
                gamma_l = float(parts[5])
                peak_params.append({
                    "e_center": e_center,
                    "theta_deg": theta_deg,
                    "gamma_r": gamma_r,
                    "gamma_l": gamma_l
                })
            except Exception as e:
                print(f"Error in parameter value on line {line_num},({e}), skipped.")
    if not peak_params:
        raise ValueError("No valid parameters")
    print(f"Peak parameters have been read: a total of {len(peak_params)} peaks")
    return peak_params

# -------------------------- Calculation --------------------------
def calculate_fitting_values(E_dense, peak_params):
    n_points = len(E_dense)
    total_t = np.zeros(n_points, dtype=np.complex64)
    
    for peak in peak_params:
        e_c = peak["e_center"]
        theta_deg = peak["theta_deg"]
        gamma_r = peak["gamma_r"]
        gamma_l = peak["gamma_l"]
        
        theta_rad = np.radians(theta_deg)
        
        exp_jtheta = np.cos(theta_rad) + 1j * np.sin(theta_rad)
        numerator = exp_jtheta * np.sqrt(gamma_r * gamma_l)
        denominator = (E_dense - e_c) + 1j * (gamma_r + gamma_l) / 2
        t_i = numerator / denominator
        
        total_t += t_i
    
    fitting_T = np.abs(total_t) ** 2
    return fitting_T

# -------------------------- Plot--------------------------
def plot_negf_vs_fitting(E_dense, negf_T, fitting_T, save_path="conductance_fitting.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(E_dense, negf_T, linewidth=4, c="#2597DF", label="AAOO NEGF", alpha=1)
    plt.plot(E_dense, fitting_T, c="black", linewidth=1.5, label="Fitting", alpha=0.8,linestyle="--")
    
    plt.xlabel(r"E-E$_F$ (eV)", fontsize=18)
    plt.ylabel("T", fontsize=18)
    plt.ylim(0, 1.1)
    plt.xlim(-3, 4)  
    
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.legend(fontsize=16, frameon=True, shadow=False, loc="center",bbox_to_anchor=(0.5, 0.2))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # -------------------------- File path --------------------------
    file_pattern = "./*-f3.TBT.AVTRANS_Left-Right"
    NEGF_FILE = glob.glob(file_pattern)[0]
    base_name = os.path.basename(NEGF_FILE)
    filename = base_name.split("-f3.")[0] 
    PARA_PARAM_FILE = "para_%s.txt"%filename
    SAVE_IMAGE_PATH = "%s_negf_fitting.png"%filename
    # ------------------------------------------------------------------------------
    
    E_dense, negf_T = read_original_negf_data(NEGF_FILE)
    
    peak_params = read_para_params(PARA_PARAM_FILE)
    
    fitting_T = calculate_fitting_values(E_dense, peak_params)
    
    plot_negf_vs_fitting(E_dense, negf_T, fitting_T, SAVE_IMAGE_PATH)
