from XY_Model import XY_Model
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, RadioButtons

L = input("Enter lattice size (default 16): ")
interval = input("Enter delay between frames in ms (default 1ms): ")

L = int(L) if L is not None and L.isnumeric() and int(L) > 0 else 16
interval = int(interval) if interval is not None and interval.isnumeric(
) and int(interval) > 0 else 1

L_range = np.arange(L)
sim = XY_Model(L)

# Preparation for the animation and plots
fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)  # Space for buttons and sliders

# Subplots for the spin lattice, magnetization and correlation function
ax_spin = plt.axes([-0.05, 0.3, 0.5, 0.6])  # [left, bottom, width, height]
ax_mag = plt.axes([0.55, 0.7, 0.3, 0.2])
ax_corr = plt.axes([0.55, 0.3, 0.3, 0.3])

# Plot the initial spin configuration
X, Y = np.meshgrid(L_range, L_range)
U = np.cos(sim.spins)
V = np.sin(sim.spins)

# Matshow for energy/vorticity
E_V = ax_spin.matshow(np.zeros_like(sim.spins), cmap='bwr',
                      origin='lower', alpha=0.6, animated=True)
cbar = plt.colorbar(E_V, ax=ax_spin, fraction=0.046,
                    pad=0.04, label='Energy / Vorticity')

# Quiver plot for the spins
Q = ax_spin.quiver(X, Y, U, V, pivot='middle', color='k', animated=True)
ax_spin.set_title("Spin configuration with vortices")
ax_spin.set_xlim(-0.5, L-0.5)
ax_spin.set_ylim(-0.5, L-0.5)
ax_spin.set_aspect('equal')
V_text = ax_spin.text(0.5, -0.05, "", transform=ax_spin.transAxes, ha='center', animated=True,
                      bbox=dict(facecolor='white', edgecolor='none', pad=0))
# ax_spin.draw_artist(Q)
ax_spin.draw_artist(V_text)

# Plot for magnetisation
ax_mag.set_title("Mean magnetisation")
ax_mag.set_xlabel(rf"$T$")
ax_mag.set_ylabel(rf"$\langle M(T) \rangle$")
ax_mag.set_xlim(0, 2.1)
ax_mag.set_ylim(-0.1, 1.1)
line_mag, = ax_mag.plot([], [], 'o', label=rf'$\langle M(T) \rangle$')
ax_mag.legend()

# Plot for correlation function
ax_corr.set_title("Correlation function")
ax_corr.set_xlabel(rf"$r$")
ax_corr.set_ylabel(rf"$C(r)$")
ax_corr.set_xlim(0.8, L//2)
ax_corr.set_ylim(1e-3, 2)
R_range = np.arange(1, L//2)
line_corr, = ax_corr.semilogy([1], [1], 'o', label=rf'$C(r)$')
# Optional: Plot for the expected r^(-1/4) decay.
# The pre factor has not been determined, so it is not great yet. ;
# line_corr_exp_decay, = ax_corr.semilogy(R_range, (R_range+1)**(-1/4), '--', label=r'$r^{-1/4}$')
ax_corr.legend()

# Start/Stop Button
ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Start',
                color='lightgoldenrodyellow', hovercolor='0.975')

# Set min-energy state Button
ax_set_min_energy = plt.axes([0.7, 0.05, 0.1, 0.075])
button_set_min_energy = Button(
    ax_set_min_energy, 'Set min-energy', color='lightgoldenrodyellow', hovercolor='0.975')

# Set single vortex Button
ax_set_vortex = plt.axes([0.7, 0.15, 0.1, 0.075])
button_set_vortex = Button(ax_set_vortex, 'Set vortex',
                           color='lightgoldenrodyellow', hovercolor='0.975')

# Set vortex pair Button
ax_set_vortex_pair = plt.axes([0.8, 0.15, 0.1, 0.075])
button_set_vortex_pair = Button(
    ax_set_vortex_pair, 'Set vortex pair', color='lightgoldenrodyellow', hovercolor='0.975')

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

# Energy Vorticity Radio Button
ax_radio = plt.axes([0.59, 0.1, 0.1, 0.1])
radio_E_V = RadioButtons(ax_radio, ('None', 'Energy', 'Vorticity'), active=0)

# Control variable(s) for the animation
running = False
T_current = slider_T.val


def update(frame):
    global T_current, running
    if running:
        sim.metropolis_step(T_current)

        # Update quiver plot
        Q.set_UVC(np.cos(sim.spins), np.sin(sim.spins))
        text = rf'$n_+$: {sim.total_vortices:.0f}, $n_-$: {
            sim.total_antivortices:.0f}, $\langle \epsilon \rangle$: {np.mean(sim.calculate_energy()):.2f}'
        V_text.set_text(text)

        # Update Matshow for energy/vorticity
        if radio_E_V.value_selected == 'Energy':
            E_V.set_data(sim.calculate_energy())
            E_V.set_clim(vmin=-4, vmax=4)
        elif radio_E_V.value_selected == 'Vorticity':
            E_V.set_data(sim.vorticity)
            E_V.set_clim(vmin=-1, vmax=1)
        else:
            E_V.set_data(np.zeros_like(sim.vorticity))

        # Update correlation function plot
        distances, correlations = sim.calculate_correlation_function()
        line_corr.set_data(distances, np.abs(correlations))
        # line_corr_exp_decay.set_data(distances, correlations[0]*(R_range+1)**(-1/4))

        # Update mean magnetisation plot

        M_values = []
        T_values = []
        for T, M_list in sim.mean_magnetisation.items():
            T_values.append(T)
            M_values.append(np.mean(M_list))
        line_mag.set_data(T_values, M_values)

    return Q, E_V, line_mag, line_corr, V_text, #line_corr_exp_decay


def start_stop(event):

    global running
    running = not running
    if running:
        button.label.set_text('Stop')
    else:
        button.label.set_text('Start')


def set_min_energy(event):
    sim.set_min_energy()
    Q.set_UVC(np.cos(sim.spins), np.sin(sim.spins))
    E_V.set_data(sim.calculate_energy())
    cbar.set_label('Energy')


def set_vortex(event):
    sim.set_single_vortex()
    Q.set_UVC(np.cos(sim.spins), np.sin(sim.spins))
    E_V.set_data(sim.calculate_energy())
    cbar.set_label('Vorticity')


def set_vortex_pair(event):
    sim.set_vortex_pair()
    Q.set_UVC(np.cos(sim.spins), np.sin(sim.spins))
    E_V.set_data(sim.calculate_energy())


def update_temperature(val):
    global T_current
    T_current = slider_T.val


button.on_clicked(start_stop)
button_set_min_energy.on_clicked(set_min_energy)
button_set_vortex.on_clicked(set_vortex)
button_set_vortex_pair.on_clicked(set_vortex_pair)
slider_T.on_changed(update_temperature)

# Run animation
ani = animation.FuncAnimation(fig, update, frames=itertools.count,
                              interval=interval, blit=True, cache_frame_data=False)

plt.show()
