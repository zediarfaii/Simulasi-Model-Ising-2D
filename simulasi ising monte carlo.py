import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# =====================================
# Fungsi dasar Model Ising 2D
# =====================================
def init_lattice(L):
    """Inisialisasi kisi spin acak."""
    return np.random.choice([-1, 1], size=(L, L))

def energy_change(spins, i, j, J=1.0, h=0.0):
    """Hitung perubahan energi ΔE jika spin (i,j) dibalik."""
    L = len(spins)
    s = spins[i, j]
    neighbor_sum = (
        spins[(i+1)%L, j] +
        spins[(i-1)%L, j] +
        spins[i, (j+1)%L] +
        spins[i, (j-1)%L]
    )
    dE = 2 * J * s * (neighbor_sum + h/J)
    return dE

def metropolis_step(spins, T, J=1.0, h=0.0):
    """Satu sweep Monte Carlo."""
    L = len(spins)
    for _ in range(L*L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        dE = energy_change(spins, i, j, J, h)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            spins[i, j] *= -1
    return spins

def total_energy(spins, J=1.0, h=0.0):
    """Energi total sistem."""
    L = len(spins)
    E = 0
    for i in range(L):
        for j in range(L):
            s = spins[i, j]
            E -= J * s * (spins[(i+1)%L, j] + spins[i, (j+1)%L])
            E -= h * s
    return E

def total_magnetization(spins):
    """Total magnetisasi."""
    return np.sum(spins)

# =====================================
# Fungsi simulasi Ising dengan callback
# =====================================
def simulate_ising(L, Tmin, Tmax, Ntemp, steps, eq_steps, J, h, progress_callback):
    temps = np.linspace(Tmin, Tmax, Ntemp)
    energies, mags, heat_caps = [], [], []

    for idx, T in enumerate(temps):
        spins = init_lattice(L)

        # Thermalization
        for _ in range(eq_steps):
            metropolis_step(spins, T, J, h)

        # Measurement
        e_vals, m_vals = [], []
        for _ in range(steps):
            metropolis_step(spins, T, J, h)
            e_vals.append(total_energy(spins, J, h))
            m_vals.append(abs(total_magnetization(spins)))

        e_vals = np.array(e_vals)
        m_vals = np.array(m_vals)

        E_mean = np.mean(e_vals) / (L*L)
        M_mean = np.mean(m_vals) / (L*L)
        Cv = (np.mean(e_vals**2) - np.mean(e_vals)**2) / (L*L * T**2)

        energies.append(E_mean)
        mags.append(M_mean)
        heat_caps.append(Cv)

        # Update progress bar (persentase)
        progress_callback(idx + 1, Ntemp, T)

    return temps, mags, energies, heat_caps

# =====================================
# GUI Event Functions
# =====================================
def run_simulation_thread():
    thread = threading.Thread(target=jalankan_simulasi)
    thread.start()

def jalankan_simulasi():
    try:
        # Ambil input
        L = int(entry_L.get())
        Tmin = float(entry_Tmin.get())
        Tmax = float(entry_Tmax.get())
        Ntemp = int(entry_Ntemp.get())
        steps = int(entry_steps.get())
        eq_steps = int(entry_eq.get())
        J = float(entry_J.get())
        h = float(entry_h.get())

        progress_bar["value"] = 0
        label_status.config(text="🔄 Menjalankan simulasi...")
        root.update_idletasks()

        def update_progress(current, total, temp_now):
            percent = (current / total) * 100
            progress_bar["value"] = percent
            label_status.config(text=f"Simulasi suhu {temp_now:.2f} K ({percent:.0f}%)")
            root.update_idletasks()

        # Jalankan simulasi
        Ts, Ms, Es, Cv = simulate_ising(L, Tmin, Tmax, Ntemp, steps, eq_steps, J, h, update_progress)

        # Hapus plot lama
        for widget in frame_plot.winfo_children():
            widget.destroy()

        # Buat grafik hasil
        fig, ax = plt.subplots(1, 2, figsize=(9,4))

        ax[0].plot(Ts, Ms, 'o-', color='blue')
        ax[0].set_xlabel("Temperatur (T)")
        ax[0].set_ylabel("Magnetisasi rata-rata |M|")
        ax[0].set_title("Kurva Magnetisasi vs Temperatur")
        ax[0].grid(True)

        ax[1].plot(Ts, Cv, 'o-', color='red')
        ax[1].set_xlabel("Temperatur (T)")
        ax[1].set_ylabel("Kapasitas Panas Cv")
        ax[1].set_title("Kurva Kapasitas Panas vs Temperatur")
        ax[1].grid(True)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        label_status.config(text="✅ Simulasi selesai! Grafik M–T dan Cv–T ditampilkan.")
        progress_bar["value"] = 100

    except ValueError:
        messagebox.showerror("Error", "Isi semua kolom input dengan angka valid!")

def tampilkan_konfigurasi():
    try:
        L = int(entry_L.get())
        T = float(entry_Tkonf.get())
        J = float(entry_J.get())
        h = float(entry_h.get())

        spins = init_lattice(L)
        for _ in range(2000):
            metropolis_step(spins, T, J, h)

        for widget in frame_plot.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(spins, cmap='coolwarm', interpolation='nearest')
        ax.set_title(f"Konfigurasi Spin (L={L}, T={T})")
        ax.axis('off')

        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        label_status.config(text="✅ Konfigurasi spin berhasil ditampilkan.")
    except ValueError:
        messagebox.showerror("Error", "Masukkan nilai numerik yang valid.")

# =====================================
# GUI TKINTER
# =====================================
root = tk.Tk()
root.title("Simulasi Model Ising 2D (Metropolis Monte Carlo)")
root.geometry("900x720")

frame_input = ttk.LabelFrame(root, text="Input Parameter Simulasi")
frame_input.pack(fill="x", padx=10, pady=5)

# Baris 1
ttk.Label(frame_input, text="Ukuran kisi (L):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
entry_L = ttk.Entry(frame_input); entry_L.insert(0, "20"); entry_L.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(frame_input, text="Jumlah langkah (steps):").grid(row=0, column=2, padx=5, pady=5, sticky="w")
entry_steps = ttk.Entry(frame_input); entry_steps.insert(0, "2000"); entry_steps.grid(row=0, column=3, padx=5, pady=5)

# Baris 2
ttk.Label(frame_input, text="Langkah thermalization (eq_steps):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
entry_eq = ttk.Entry(frame_input); entry_eq.insert(0, "500"); entry_eq.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(frame_input, text="Konstanta interaksi (J):").grid(row=1, column=2, padx=5, pady=5, sticky="w")
entry_J = ttk.Entry(frame_input); entry_J.insert(0, "1.0"); entry_J.grid(row=1, column=3, padx=5, pady=5)

# Baris 3
ttk.Label(frame_input, text="Medan magnet eksternal (h):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
entry_h = ttk.Entry(frame_input); entry_h.insert(0, "0.0"); entry_h.grid(row=2, column=1, padx=5, pady=5)

ttk.Label(frame_input, text="T untuk konfigurasi (Tkonf):").grid(row=2, column=2, padx=5, pady=5, sticky="w")
entry_Tkonf = ttk.Entry(frame_input); entry_Tkonf.insert(0, "2.2"); entry_Tkonf.grid(row=2, column=3, padx=5, pady=5)

# Baris 4
ttk.Label(frame_input, text="Temperatur minimum (Tmin):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
entry_Tmin = ttk.Entry(frame_input); entry_Tmin.insert(0, "1.5"); entry_Tmin.grid(row=3, column=1, padx=5, pady=5)

ttk.Label(frame_input, text="Temperatur maksimum (Tmax):").grid(row=3, column=2, padx=5, pady=5, sticky="w")
entry_Tmax = ttk.Entry(frame_input); entry_Tmax.insert(0, "3.5"); entry_Tmax.grid(row=3, column=3, padx=5, pady=5)

# Baris 5
ttk.Label(frame_input, text="Jumlah titik suhu (Ntemp):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
entry_Ntemp = ttk.Entry(frame_input); entry_Ntemp.insert(0, "10"); entry_Ntemp.grid(row=4, column=1, padx=5, pady=5)

# Tombol
frame_buttons = ttk.Frame(root); frame_buttons.pack(pady=5)
btn_run = ttk.Button(frame_buttons, text="Jalankan Simulasi", command=run_simulation_thread)
btn_run.grid(row=0, column=0, padx=10)
btn_conf = ttk.Button(frame_buttons, text="Tampilkan Konfigurasi", command=tampilkan_konfigurasi)
btn_conf.grid(row=0, column=1, padx=10)

# Progress bar
progress_bar = ttk.Progressbar(root, length=600, mode="determinate")
progress_bar.pack(pady=10)

# Label status & area plot
label_status = ttk.Label(root, text="Masukkan semua parameter lalu jalankan simulasi.")
label_status.pack(pady=10)

frame_plot = ttk.LabelFrame(root, text="Visualisasi Hasil")
frame_plot.pack(fill="both", expand=True, padx=10, pady=5)

root.mainloop()
