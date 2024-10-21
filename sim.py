from XY_Model import XY_Model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, CheckButtons


# Initialisierung der Simulation
steps = 100_000
L = 40  # Für eine bessere Visualisierung kleinere Gittergröße
sim = XY_Model(L, steps)

# Vorbereitung für die Animation
fig = plt.figure(figsize=(12,6))
plt.subplots_adjust(bottom=0.25)  # Platz für Widgets

# Subplots für die Spin-Konfiguration und die Magnetisierung
ax_spin = plt.axes([0.05, 0.3, 0.4, 0.6])  # [left, bottom, width, height]
ax_mag = plt.axes([0.55, 0.7, 0.4, 0.2])
ax_corr = plt.axes([0.55, 0.3, 0.4, 0.2])

# Plot für das Spin-Gitter
X, Y = np.meshgrid(np.arange(L), np.arange(L))
U = np.cos(sim.spins)
V = np.sin(sim.spins)

# Matshow für die Energie im Hintergrund

E = ax_spin.matshow(sim.vorticity, cmap='viridis', origin='lower', alpha=0.6)
cbar = plt.colorbar(E, ax=ax_spin, fraction=0.046, pad=0.04, label='Vorticity')

# Quiver für die Spins
Q = ax_spin.quiver(X, Y, U, V, pivot='middle', color='white')
ax_spin.set_title("Spin configuration with vortices")
ax_spin.set_xlim(-0.5, L-0.5)
ax_spin.set_ylim(-0.5, L-0.5)
ax_spin.set_aspect('equal')
V_text = ax_spin.text(0.5, -0.05, rf'$C_+$: {sim.total_vortices:.0f}, $C_-$: {sim.total_antivortices:.0f}', transform=ax_spin.transAxes, ha='center')

# Plot für die Magnetisierung
ax_mag.set_title("Mean magnetisation")
ax_mag.set_xlabel(rf"$T$")
ax_mag.set_ylabel(rf"$\langle M(T) \rangle$")
ax_mag.set_xlim(0, 2.1)
ax_mag.set_ylim(-0.1, 1.1)
line_mag, = ax_mag.plot([], [], 'o', label=rf'$\langle M(T) \rangle$')
ax_mag.legend()

# Plot für die Korrelationsfunktion
ax_corr.set_title("Correlation function")
ax_corr.set_xlabel(rf"$r$")
ax_corr.set_ylabel(rf"$C(r)$")
ax_corr.set_xlim(0, L//2)
ax_corr.set_ylim(-1, 1)
line_corr, = ax_corr.plot([], [], 'o', label=rf'$C(r)$')
ax_corr.legend()

# Start/Stop Button
ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Start', color='lightgoldenrodyellow', hovercolor='0.975')

# Start/Stop Button
ax_corr_checkbox = plt.axes([0.6, 0.05, 0.1, 0.075])
corr_checkbox = CheckButtons(ax_corr_checkbox, ('Calc <M>', 'Calc. C(r)',), (False, False))

# Temperatur Slider
ax_slider = plt.axes([0.2, 0.1, 0.3, 0.03], facecolor='lightgoldenrodyellow')
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
        V_text.set_text(rf'$C_+$: {sim.total_vortices:.0f}, $C_-$: {sim.total_antivortices:.0f}')

        # Aktualisiere Energie-Matshow
        E.set_data(sim.vorticity)

        # Optional: Dynamische Anpassung der Farbskala
        # E.set_clim(vmin=np.min(sim.vorticity), vmax=np.max(sim.vorticity))
        # E.set_clim(vmin=-1, vmax=1)
        cbar.update_normal(E)

        # Aktualisiere Korrelationsfunktion
        if corr_checkbox.get_status()[1]:
            distances, correlations = sim.calculate_correlation_function()
            line_corr.set_data(distances, correlations)


        # Aktualisiere Magnetisierung
        if corr_checkbox.get_status()[0]:
            M_values = []
            T_values = []
            for T, M_list in sim.mean_magnetisation.items():
                T_values.append(T)
                M_values.append(np.mean(M_list))
            line_mag.set_data(T_values, M_values)


    return Q, E, line_mag

# Aktualisiere den Temperaturwert, wenn der Slider bewegt wird
def update_temperature(val):
    global T_current
    T_current = slider_T.val
    # Optional: Reset der Magnetisierung, wenn die Temperatur geändert wird
    # sim.mean_magnetisation = {}

slider_T.on_changed(update_temperature)

# Animation starten
ani = animation.FuncAnimation(fig, update, frames=steps, interval=0, blit=False)

plt.show()
