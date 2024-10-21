import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider


class XY_Model:
    # Parameter
    L = None
    steps = None
    mean_magnetisation = None
    vorticity = None
    total_vortices = None
    total_antivortices = None

    @property
    def spins(self):
        return self._calc_spins[self.L:2*self.L, self.L:2*self.L]

    def __init__(self, L, steps):
        self.L = L
        self.steps = steps
        self._calc_spins = np.random.uniform(0, 2*np.pi, (3*L, 3*L))	
        self.mean_magnetisation = {}
        self.current_step = 0
        self.current_energy = self.calculate_energy()
        self.vorticity = self.calculate_vorticity()
        self.total_vortices = 0
        self.total_antivortices = 0
        self.J = 1

    def calculate_correlation_function(self):
        """Berechnet die Korrelationsfunktion C(r) als Funktion des Abstands r."""
        max_distance = self.L // 2
        distances = np.arange(1, max_distance)
        correlations = np.zeros(len(distances))
        counts = np.zeros(len(distances))

        for idx, r in enumerate(distances):
            for i in range(self.L):
                for j in range(self.L):
                    # Zielpunkt mit Abstand r in x-Richtung (periodische Randbedingungen)
                    x2 = (i + r) % self.L
                    y2 = j
                    delta_theta = self.spins[i, j] - self.spins[x2, y2]
                    correlations[idx] += np.cos(delta_theta)
                    counts[idx] += 1

                    # Zielpunkt mit Abstand r in y-Richtung
                    x2 = i
                    y2 = (j + r) % self.L
                    delta_theta = self.spins[i, j] - self.spins[x2, y2]
                    correlations[idx] += np.cos(delta_theta)
                    counts[idx] += 1

        # Normiere die Korrelationen
        correlations = correlations / counts

        return distances, correlations

    def calculate_magnetisation(self):
        X, Y = np.sum(np.cos(self.spins)), np.sum(np.sin(self.spins))
        M = np.sqrt(X**2 + Y**2)
        return M / (self.L * self.L)

    def calculate_energy(self):
        """Berechnet die lokale Energie jedes Spins."""
        energy = np.zeros((self.L, self.L))
        shifts = [(-1,0), (1,0), (0,-1), (0,1)]
        for dx, dy in shifts:
            # Verschieben des Spin-Gitters für Nachbarn
            spins_shifted = np.roll(self.spins, shift=(dx, dy), axis=(0,1))
            energy += -np.cos(self.spins - spins_shifted)
        # Da jede Wechselwirkung doppelt gezählt wird, teilen wir durch 2
        energy = energy / 2
        return energy

    def calculate_energy_difference(self, theta_new):
        delta_E = np.zeros((3*self.L, 3*self.L))
        L = self.L
        J = self.J
        
        # Shifts for the neighbors (periodic boundary conditions)
        shifts = [(-1,0), (1,0), (0,-1), (0,1)]

        for dx, dy in shifts:
            # Shift the spin lattice for the neighbors
            spins_shifted = np.roll(self._calc_spins, shift=(dx, dy), axis=(0,1))
            # Calculate the energy difference
            delta_E += -self.J * (np.cos(theta_new - spins_shifted) - np.cos(self._calc_spins - spins_shifted))

        return delta_E

    def calculate_vorticity(self):

        # Calculates vortex and antivortex positions on the lattice.

        L = self.L
        _s = self._calc_spins

        vorticity = np.zeros((self.L, self.L))
        for i in range(0, L):
            for j in range(0, L):

                i_shifted = i + L
                j_shifted = j + L
                
                # Phase differences along the plaquette
                delta_theta1 = _s[i_shifted, j_shifted] - _s[(i_shifted + 1), j_shifted]
                delta_theta2 = _s[(i_shifted + 1), j_shifted] - _s[(i_shifted + 1), (j_shifted + 1)]
                delta_theta3 = _s[(i_shifted + 1), (j_shifted + 1) ] - _s[i_shifted, (j_shifted + 1) ]
                delta_theta4 = _s[i_shifted, (j_shifted + 1)] - _s[i_shifted, j_shifted]

                # Unwrap phase differences to be between -pi and pi
                delta_theta1 = (delta_theta1 + np.pi) % (2 * np.pi) - np.pi
                delta_theta2 = (delta_theta2 + np.pi) % (2 * np.pi) - np.pi
                delta_theta3 = (delta_theta3 + np.pi) % (2 * np.pi) - np.pi
                delta_theta4 = (delta_theta4 + np.pi) % (2 * np.pi) - np.pi

                # Gesamte Windung um das Plaquette
                winding = delta_theta1 + delta_theta2 + delta_theta3 + delta_theta4

                # Vorticity als ganzzahliges Vielfaches von 2*pi
                vorticity[i, j] = winding / (2 * np.pi)

        return vorticity


    def metropolis_step(self, T):
        
        beta = 1 / T if T > 0 else np.inf
        L = self.L

        # Masks 90% of the spins to not be updated.
        # MCM usualy updates each spin once per step.
        # Doing 10% of the spins per step is a good compromise between speed and accuracy.
        mask = np.random.rand(3*L, 3*L) < 1

        
        
        # New angles for the spins
        theta_new = np.random.uniform(0, 2*np.pi, (3*L, 3*L))
        
        # Energy difference for each spin
        delta_E = self.calculate_energy_difference(theta_new)
        
        # Metropolis acceptance criterion
        rand_vals = np.random.rand(3*L, 3*L)
        accept = (delta_E < 0) | (rand_vals < np.exp(-beta * delta_E)) & mask

        self._calc_spins[accept] = theta_new[accept]

        # Calculates the vorticity
        self.vorticity = self.calculate_vorticity()
        self.total_vortices = np.sum(self.vorticity > 0.95)
        self.total_antivortices = np.sum(self.vorticity < -0.95)


        # Calculates magnetisation
        M = self.calculate_magnetisation()
        if T not in self.mean_magnetisation:
            self.mean_magnetisation[T] = np.zeros(20)
        self.mean_magnetisation[T] = np.roll(self.mean_magnetisation[T], -1)
        self.mean_magnetisation[T][-1] = M
        self.current_energy = self.calculate_energy()

        # Step counter
        self.current_step += 1