import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider


class XY_Model:
    # Parameter
    L = None
    steps = None
    spins = None
    mean_magnetisation = None
    vorticity = None
    total_vortices = None
    total_antivortices = None

    def __init__(self, L, steps):
        self.L = L
        self.steps = steps
        self.spins = np.random.uniform(0, 2*np.pi, (L, L))
        self.mean_magnetisation = {}
        self.current_step = 0
        self.current_energy = self.calculate_energy()
        self.vorticity = self.calculate_vorticity()
        self.total_vortices = 0
        self.total_antivortices = 0

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
        """Berechnet die Energiedifferenz ΔE für vorgeschlagene neue Winkel."""
        delta_E = np.zeros((self.L, self.L))
        shifts = [(-1,0), (1,0), (0,-1), (0,1)]
        for dx, dy in shifts:
            # Verschieben des Spin-Gitters
            spins_shifted = np.roll(self.spins, shift=(dx, dy), axis=(0,1))
            # Berechnung der Energiedifferenz
            delta_E += -(np.cos(theta_new - spins_shifted) - np.cos(self.spins - spins_shifted))
        return delta_E

    def calculate_vorticity(self):
        """Berechnet die Vortizität für jedes Plaquette."""
        vorticity = np.zeros((self.L, self.L))
        for i in range(self.L):
            for j in range(self.L):
                # Phasenunterschiede entlang des Plaquettes
                delta_theta1 = self.spins[i, j] - self.spins[(i + 1) % self.L, j]
                delta_theta2 = self.spins[(i + 1) % self.L, j] - self.spins[(i + 1) % self.L, (j + 1) % self.L]
                delta_theta3 = self.spins[(i + 1) % self.L, (j + 1) % self.L] - self.spins[i, (j + 1) % self.L]
                delta_theta4 = self.spins[i, (j + 1) % self.L] - self.spins[i, j]

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
        for _ in range(self.L * self.L):
            # Wähle einen zufälligen Spin aus
            i = np.random.randint(0, self.L)
            j = np.random.randint(0, self.L)

            # Aktueller Winkel
            theta_old = self.spins[i, j]

            # Vorschlagen eines neuen Winkels (hier ein zufälliger neuer Winkel)
            theta_new = np.random.uniform(0, 2*np.pi)

            # Berechnung der Energiedifferenz ΔE
            delta_E = 0.0


            neighbors = [((i + 1) % self.L, j),
                         ((i - 1) % self.L, j),
                         (i, (j + 1) % self.L),
                         (i, (j - 1) % self.L)]
            for ni, nj in neighbors:
                delta_E += -np.cos(theta_new - self.spins[ni, nj]) + np.cos(theta_old - self.spins[ni, nj])

            # Metropolis-Kriterium
            if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
                self.spins[i, j] = theta_new

        # # Berechne Vortizität nach dem Schritt
        self.vorticity = self.calculate_vorticity()
        self.total_vortices = np.sum(self.vorticity > 0.5)
        self.total_antivortices = np.sum(self.vorticity < -0.5)


        # Berechne Magnetisierung
        M = self.calculate_magnetisation()
        if T not in self.mean_magnetisation:
            self.mean_magnetisation[T] = np.zeros(20)
        self.mean_magnetisation[T] = np.roll(self.mean_magnetisation[T], -1)
        self.mean_magnetisation[T][-1] = M
        self.current_energy = self.calculate_energy()

        # Schrittzähler erhöhen
        self.current_step += 1