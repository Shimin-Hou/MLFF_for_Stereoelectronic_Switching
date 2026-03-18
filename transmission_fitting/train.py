import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------------------
# 1. Data reading and preprocessing
# ------------------------------------------------------------------------------
def read_transmission_data(filename, min_T_for_log=1e-20):
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
                    E_val = float(parts[0])
                    T_val = float(parts[1])
                    T_val = max(T_val, min_T_for_log)
                    data.append([E_val, T_val])
                except:
                    print(f"Line {line_num} has invalid data format, skipped")
    if not data:
        raise ValueError("No valid data read")
    return np.array(data, dtype=np.float32)

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
        if n_insert <= 0:
            E_dense.append(x1)
            T_dense.append(y1)
            continue
        
        x_insert = np.linspace(x1, x2, n_insert + 2)[:-1]
        y_insert = np.linspace(y1, y2, n_insert + 2)[:-1]
        E_dense.extend(x_insert)
        T_dense.extend(y_insert)
    
    E_dense.append(E_original[-1])
    T_dense.append(T_original[-1])
    return np.array(E_dense, dtype=np.float32), np.array(T_dense, dtype=np.float32)

# ------------------------------------------------------------------------------
# 2. Custom Dataset (support dynamic weights)
# ------------------------------------------------------------------------------
class TransmissionDataset(Dataset):
    def __init__(self, E, T, device):
        self.E = torch.tensor(E.reshape(-1, 1), dtype=torch.float32, device=device)
        self.T = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)
    
    def __len__(self):
        return len(self.E)
    
    def __getitem__(self, idx):
        return self.E[idx], self.T[idx]

# ------------------------------------------------------------------------------
# 3. Transmission Model (with physical constraints)
# ------------------------------------------------------------------------------
class PeakTransmissionModel(nn.Module):
    def __init__(self, n_peaks, tunable_peaks, e0_list, e_tune_range=0.1):
        super().__init__()
        self.n_peaks = n_peaks
        self.tunable_peaks = tunable_peaks
        self.e0_list = e0_list
        self.e_tune_range = e_tune_range
        
        self.e_centers = nn.ModuleList()  # e
        self.thetas = nn.ModuleList()     # theta
        self.gamma_rs = nn.ModuleList()   # gamma_r
        self.gamma_ls = nn.ModuleList()   # gamma_l
        
        for i in range(n_peaks):
            if i in tunable_peaks:
                self.e_centers.append(nn.Linear(1, 1))
            else:
                self.e_centers.append(None)
            
            self.thetas.append(nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid()
            ))
            
            gamma_min=0.0001
            gamma_max=0.5
            self.gamma_rs.append(nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid()
            ))
            self.gamma_ls.append(nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid()
            ))
    
    def forward(self, E):
        batch_size = E.shape[0]
        j = torch.tensor(1j, dtype=torch.complex64, device=device)
        ones = torch.ones(batch_size, 1, device=device)
        gamma_min = 0.0001
        gamma_max = 0.5
        
        e_centers = torch.zeros(self.n_peaks, batch_size, 1, device=device, dtype=torch.float32)
        theta_rads = torch.zeros(self.n_peaks, batch_size, 1, device=device, dtype=torch.float32)
        gamma_rs = torch.zeros(self.n_peaks, batch_size, 1, device=device, dtype=torch.float32)
        gamma_ls = torch.zeros(self.n_peaks, batch_size, 1, device=device, dtype=torch.float32)
        
        for i in range(self.n_peaks):
            if i in self.tunable_peaks:
                e_center_raw = self.e_centers[i](ones)
                e_centers[i] = self.e0_list[i] + self.e_tune_range * torch.tanh(e_center_raw)
            else:
                e_centers[i] = torch.full((batch_size, 1), self.e0_list[i], device=device)
            
            theta_raw = self.thetas[i](ones)
            theta_rads[i] = 180 * theta_raw * torch.pi / 180
            
            gamma_r_raw = self.gamma_rs[i](ones)
            gamma_rs[i] = gamma_min + (gamma_max - gamma_min) * gamma_r_raw
            
            gamma_l_raw = self.gamma_ls[i](ones)
            gamma_ls[i] = gamma_min + (gamma_max - gamma_min) * gamma_l_raw
        
        e_centers_c = e_centers.to(torch.complex64)
        theta_rads_c = theta_rads.to(torch.complex64)
        gamma_rs_c = gamma_rs.to(torch.complex64)
        gamma_ls_c = gamma_ls.to(torch.complex64)
        E_c = E.to(torch.complex64).unsqueeze(0)
        
        exp_jtheta = torch.cos(theta_rads_c) + j * torch.sin(theta_rads_c)
        numerator = exp_jtheta * torch.sqrt(gamma_rs_c * gamma_ls_c)
        denominator = (E_c - e_centers_c) + j * (gamma_rs_c + gamma_ls_c) / 2
        t_i = numerator / denominator
        
        total_t = t_i.sum(dim=0)
        T_pred = torch.abs(total_t) ** 2
        return T_pred.to(torch.float32)
    
    def get_current_peak_centers(self):
        peak_centers = []
        dummy_input = torch.tensor([[1.0]], dtype=torch.float32, device=device)
        
        for i in range(self.n_peaks):
            if i in self.tunable_peaks:
                e_center_raw = self.e_centers[i](dummy_input)
                e_center = self.e0_list[i] + self.e_tune_range * torch.tanh(e_center_raw)
                peak_centers.append(e_center.item())
            else:
                peak_centers.append(self.e0_list[i])
        
        return peak_centers

# ------------------------------------------------------------------------------
# 4. Train model
# ------------------------------------------------------------------------------
def train_model():
    # 0. tunable parameters!
    ff='AAOO_0'
    set_random_seed(seed=43)
    filename = '%s-f3.TBT.AVTRANS_Left-Right'%(ff)
    fit_E_min, fit_E_max = -3.0, 4.0
    peak_window = 0.1  
    weight_peak = 1
    weight_default = 1  
    max_y_gap = 0.001
    e_tune_range = 0.2
    gamma_max=0.5
    gamma_min=0.0001
    val_split = 0.1  
    batch_size = 256
    max_epochs = 5000  
    lr = 0.001
    min_T_for_log = 1e-20
    loss_threshold = 2e-4
    weight_update_freq = 10
    e0_list = [3.55, 3.43, 3.14, 2.65, -1.66, -2.53, -2.72, -2.86, -2.99] # initial e, refer to MPSH
    tunable_peaks=range(len(e0_list))
    n_peaks = len(e0_list)
    
    
    
    # 1. Read and process data
    data = read_transmission_data(filename, min_T_for_log)
    E_raw, T_raw = data[:, 0], data[:, 1]
    mask = (E_raw >= fit_E_min) & (E_raw <= fit_E_max)
    E_original, T_original = E_raw[mask], T_raw[mask]
    print(f"Original data: [{fit_E_min}, {fit_E_max}]eV, total {len(E_original)} points")
    
    E_dense, T_dense = dense_sample_by_y(E_original, T_original, max_y_gap)
    add_num = len(E_dense) - len(E_original)
    print(f"Densified data: total {len(E_dense)} points (added {add_num} points)")
    
    # 2. Build dataset and dataloader (without initial weights)
    dataset = TransmissionDataset(E_dense, T_dense, device)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    E_train = E_dense[train_indices]
    E_val = E_dense[val_indices]
    train_idx_map = {idx: i for i, idx in enumerate(train_indices)}
    val_idx_map = {idx: i for i, idx in enumerate(val_indices)}
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Initialize model and optimizer
    model = PeakTransmissionModel(n_peaks, tunable_peaks, e0_list, e_tune_range).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.8, patience=5, verbose=True, min_lr=1e-5
    )
    
    # 4. Training loop (dynamic weight update)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') 
    
    current_peak_centers = model.get_current_peak_centers()
    train_weights = np.full(len(E_train), weight_default, dtype=np.float32)
    val_weights = np.full(len(E_val), weight_default, dtype=np.float32)
    
    for epoch in range(max_epochs):
        if epoch % weight_update_freq == 0:
            current_peak_centers = model.get_current_peak_centers()
            if epoch % (weight_update_freq*10) == 0:
                print(f"\nEpoch {epoch+1} - Update peak center weights, current peak centers: {[f'{c:.3f}' for c in current_peak_centers]}")
            
            train_weights[:] = weight_default
            for e_center in current_peak_centers:
                mask_peak = (E_train >= e_center - peak_window) & (E_train <= e_center + peak_window)
                train_weights[mask_peak] = weight_peak
            
            val_weights[:] = weight_default
            for e_center in current_peak_centers:
                mask_peak = (E_val >= e_center - peak_window) & (E_val <= e_center + peak_window)
                val_weights[mask_peak] = weight_peak
        
        model.train()
        train_loss = 0.0
        for batch_idx, (E_batch, T_batch) in enumerate(train_loader):
            batch_size_actual = E_batch.shape[0]
            
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size_actual
            batch_indices_in_split = train_dataset.indices[start_idx:end_idx]
            batch_indices_in_split = batch_indices_in_split[:batch_size_actual]
            
            try:
                weight_indices = [train_idx_map[idx] for idx in batch_indices_in_split] 
                batch_weights = torch.tensor(
                    train_weights[weight_indices],
                    device=device
                ).reshape(-1, 1)
            except ValueError as e:
                print(f"Batch index error: {e}")
                print(f"batch_indices_in_split: {batch_indices_in_split}")
                print(f"train_indices: {train_indices[:10]}...")
                raise
            
            assert batch_weights.shape[0] == E_batch.shape[0], \
                f"Weight size ({batch_weights.shape[0]}) does not match batch size ({E_batch.shape[0]})"
            
            E_batch = E_batch.to(device)
            T_batch = T_batch.to(device)
            
            optimizer.zero_grad()
            T_pred = model(E_batch)
            loss = torch.mean(batch_weights * (T_pred - T_batch) **2)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * E_batch.size(0)
            avg_train_loss = train_loss / len(train_dataset)
            train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (E_batch, T_batch) in enumerate(val_loader):
                batch_size_actual = E_batch.shape[0]
                
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size_actual
                batch_indices_in_split = val_dataset.indices[start_idx:end_idx]
                batch_indices_in_split = batch_indices_in_split[:batch_size_actual]
                
                try:
                    weight_indices = [val_indices.index(idx) for idx in batch_indices_in_split]
                    batch_weights = torch.tensor(
                        val_weights[weight_indices],
                        device=device
                    ).reshape(-1, 1)
                except ValueError as e:
                    print(f"Validation batch index error: {e}")
                    print(f"batch_indices_in_split: {batch_indices_in_split}")
                    print(f"val_indices: {val_indices[:10]}...")
                    raise
                
                assert batch_weights.shape[0] == E_batch.shape[0], \
                    f"Validation weight size({batch_weights.shape[0]}) does not match batch size ({E_batch.shape[0]})"
                
                E_batch = E_batch.to(device)
                T_batch = T_batch.to(device)
                
                T_pred = model(E_batch)
                loss = torch.mean(batch_weights * (T_pred - T_batch)** 2)
                val_loss += loss.item() * E_batch.size(0)
                avg_val_loss = val_loss / len(val_dataset)
                val_losses.append(avg_val_loss)
                
        if epoch % (weight_update_freq*10) == 0:
            print(f"\n                  train_loss: {avg_train_loss:.6f} - val_loss: {avg_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)

        if avg_val_loss <= loss_threshold:
            print(f"\nValidation loss {avg_val_loss:.6f} reaches threshold {loss_threshold}, stop training")
            break
    
    model.eval()
    with torch.no_grad():
        E_all = torch.tensor(E_dense.reshape(-1, 1), dtype=torch.float32, device=device)
        T_pred_full = model(E_all).cpu().numpy().flatten()
        
    # 6. Visualize results
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='valid')
    plt.axhline(y=loss_threshold, color='r', linestyle='--', label=f'Threshold {loss_threshold}')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    
    # Plot fitting results
    plt.subplot(2, 1, 2)
    plt.scatter(E_dense, T_dense, s=3, c='blue', label='true data')
    plt.plot(E_dense, T_pred_full, 'r-', linewidth=2, label='fitting data')

    final_peak_centers = model.get_current_peak_centers()
    for e_center in final_peak_centers:
        plt.axvline(x=e_center, color='green', linestyle='--', alpha=0.5)
    plt.xlabel('E (eV)')
    plt.ylabel('Conductance (T [G0])')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('%s_test.png'%ff)
    plt.show()
    
    # 7. Output learned parameters
    print("Peak | Initial center(eV) | Learned center(eV) | theta(deg) | gamma_r | gamma_l")
    print("-" * 70)
    with open('para_%s.txt'%ff, 'w') as f:
        f.write("Peak | Initial center(eV) | Learned center(eV) | theta(deg) | gamma_r | gamma_l\n")
        f.write("-" * 70 + "\n")
        
        for i in range(n_peaks):
            e0 = e0_list[i]
            # e
            if i in tunable_peaks:
                e_layer = model.e_centers[i]
                w = e_layer.weight.item()
                b = e_layer.bias.item()
                e_center = e0 + e_tune_range * np.tanh(w * 1 + b)
            else:
                e_center = e0
            
            # Theta
            theta_layer = model.thetas[i][0]
            w = theta_layer.weight.item()
            b = theta_layer.bias.item()
            theta_raw = 1 / (1 + np.exp(-(w * 1 + b)))
            theta = 180 * theta_raw
            
            # Gamma_r
            gamma_r_layer = model.gamma_rs[i][0]
            w = gamma_r_layer.weight.item()
            b = gamma_r_layer.bias.item()
            gamma_r_raw = 1 / (1 + np.exp(-(w * 1 + b)))
            gamma_r = gamma_min + (gamma_max - gamma_min) * gamma_r_raw
            
            # Gamma_l
            gamma_l_layer = model.gamma_ls[i][0]
            w = gamma_l_layer.weight.item()
            b = gamma_l_layer.bias.item()
            gamma_l_raw = 1 / (1 + np.exp(-(w * 1 + b)))
            gamma_l = gamma_min + (gamma_max - gamma_min) * gamma_l_raw
            
            line = f"{i+1:4d} | {e0:12.3f} | {e_center:14.3f} | {theta:10.1f} | {gamma_r:.4f} | {gamma_l:.4f}"
            print(line)
            f.write(line + "\n")
    
    print(f"Parameters saved to txt")
    
    
if __name__ == "__main__":
    train_model()