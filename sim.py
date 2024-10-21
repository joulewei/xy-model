from XY_Model import XY_Model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider


# Initialisierung der Simulation
steps = 100_000
L = 100  # Für eine bessere Visualisierung kleinere Gittergröße
sim = XY_Model(L, steps)

# Vorbereitung für die Animation
fig = plt.figure(figsize=(12,6))
plt.subplots_adjust(bottom=0.25)  # Platz für Widgets

# Subplots für die Spin-Konfiguration und die Magnetisierung
ax_spin = plt.axes([0.05, 0.3, 0.4, 0.6])  # [left, bottom, width, height]
ax_mag = plt.axes([0.55, 0.3, 0.4, 0.6])

# Plot für das Spin-Gitter
X, Y = np.meshgrid(np.arange(L), np.arange(L))
U = np.cos(sim.spins)
V = np.sin(sim.spins)

# Matshow für die Energie im Hintergrund

E = ax_spin.matshow(sim.vorticity, cmap='viridis', origin='lower', alpha=0.6)
cbar = plt.colorbar(E, ax=ax_spin, fraction=0.046, pad=0.04, label='Energie')

# Quiver für die Spins
Q = ax_spin.quiver(X, Y, U, V, pivot='middle', color='white')
ax_spin.set_title("Spin-Konfiguration mit Energie")
ax_spin.set_xlim(-0.5, L-0.5)
ax_spin.set_ylim(-0.5, L-0.5)
ax_spin.set_aspect('equal')
# V_text = ax_spin.text(0.5, 0.05, f'Vortizität: {sim.calculate_vorticity():.2f}', transform=ax_spin.transAxes, ha='center')

# Plot für die Magnetisierung
ax_mag.set_title("Magnetisierung")
ax_mag.set_xlabel("Temperatur T")
ax_mag.set_ylabel("M(T)")
ax_mag.set_xlim(0, 2.1)
ax_mag.set_ylim(-0.1, 1.1)
line_mag, = ax_mag.plot([], [], 'o', label='M(T)')
ax_mag.legend()

# Start/Stop Button
ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Start', color='lightgoldenrodyellow', hovercolor='0.975')

# Temperatur Slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgoldenrodyellow')
slider_T = Slider(
    ax=ax_slider,
    label='Temperatur T',
    valmin=0.0,
    valmax=2.0,
    valinit=2.0,
    valstep=0.1
)

# Variablen zur Steuerung der Animation
running = False
T_current = slider_T.val

def start_stop(event):
    global running
    running = not running
    if running:
        button.label.set_text('Stop')
    else:
        button.label.set_text('Start')

button.on_clicked(start_stop)

def update(frame):
    global T_current
    if running:
        sim.metropolis_step(T_current)

        # Aktualisiere Quiver
        Q.set_UVC(np.cos(sim.spins), np.sin(sim.spins))
        # V_text.set_text(f'Vortizität: {sim.calculate_vorticity():.2f}')

        # Aktualisiere Energie-Matshow
        E.set_data(sim.vorticity)

        # Optional: Dynamische Anpassung der Farbskala
        # E.set_clim(vmin=np.min(sim.vorticity), vmax=np.max(sim.vorticity))
        E.set_clim(vmin=-1, vmax=1)
        cbar.update_normal(E)

        # Aktualisiere Magnetisierung
        M_values = []
        T_values = []
        for T, M_list in sim.mean_magnetisation.items():
            T_values.append(T)
            M_values.append(np.mean(M_list))
        line_mag.set_data(T_values, M_values)

        # Optional: Begrenzung der x-Achse, falls nötig
        if len(T_values) > 0:
            ax_mag.set_xlim(0, max(T_values) + 0.5)

    return Q, E, line_mag

# Aktualisiere den Temperaturwert, wenn der Slider bewegt wird
def update_temperature(val):
    global T_current
    T_current = slider_T.val
    # Optional: Reset der Magnetisierung, wenn die Temperatur geändert wird
    # sim.mean_magnetisation = {}

slider_T.on_changed(update_temperature)

# Animation starten
ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=False)

plt.show()
