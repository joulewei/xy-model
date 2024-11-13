import numpy as np


class XY_Model:
    """
    Class for the 2D XY model.

    Attributes:
    L (int): Linear size of the lattice.
    mean_magnetisation (dict): Dictionary containing the mean magnetisation for each temperature.
    vorticity (np.ndarray): Vorticity of the lattice.
    total_vortices (int): Number of vortices in the lattice.
    total_antivortices (int): Number of antivortices in the
    spins (np.ndarray): Spin configuration of the lattice.
    current_energy (np.ndarray): Current energy of the lattice.
    J (float): Coupling constant of the model.

    Methods:
    calculate_correlation_function: Calculate the correlation function of the lattice.
    calculate_magnetisation: Calculate the magnetisation of the lattice.
    set_min_energy: Set the lattice to the minimum energy configuration.
    calculate_energy: Calculate the energy of the lattice.
    calculate_energy_difference: Calculate the energy difference of a new configuration.
    calculate_vorticity: Calculate the vorticity of the lattice.
    set_single_vortex: Set a single vortex in the lattice.
    set_vortex_pair: Set a vortex-antivortex pair in the lattice.
    metropolis_step: Perform a single Metropolis step in the
    """

    L = None
    _calc_spins = None
    mean_magnetisation = {}
    current_energy = None
    vorticity = None
    total_vortices = 0
    total_antivortices = 0
    J = 1

    @property
    def spins(self):
        return self._calc_spins[self.L:2*self.L, self.L:2*self.L]

    def __init__(self, L):
        self.L = L
        self._calc_spins = np.random.uniform(0, 2*np.pi, (3*L, 3*L))
        self.mean_magnetisation = {}
        self.current_step = 0
        self.current_energy = self.calculate_energy()
        self.vorticity = self.calculate_vorticity()
        self.total_vortices = 0
        self.total_antivortices = 0
        self.J = 1

    def calculate_correlation_function(self):
        """
        Calculate the correlation function of the lattice.

        Returns:
        distances (np.ndarray): Array of distances between spins.
        correlations (np.ndarray): Array of correlations between spins.

        """
        max_distance = self.L // 2
        distances = np.arange(1, max_distance)
        correlations = np.zeros(len(distances))
        counts = np.zeros(len(distances))

        for r in distances:
            # Principal components
            s_shifted_x = np.roll(self.spins, shift=(r, 0), axis=(0, 1))
            s_shifted_y = np.roll(self.spins, shift=(0, r), axis=(0, 1))

            # Diagonal components
            s_shifted_diag1 = np.roll(self.spins, shift=(r, r), axis=(0, 1))
            s_shifted_diag2 = np.roll(self.spins, shift=(r, -r), axis=(0, 1))

            # Calculate cosine of the differences
            correlations[r - 1] = (
                np.sum(np.cos(self.spins - s_shifted_x)) +
                np.sum(np.cos(self.spins - s_shifted_y)) +
                np.sum(np.cos(self.spins - s_shifted_diag1)) +
                np.sum(np.cos(self.spins - s_shifted_diag2))
            )

            # Number of spin pairs for each distance
            # This implementation is a relict from playing around with
            # different boundary conditions. Since the current boundary conditions are periodic,
            # the number of pairs is constant.
            counts[r - 1] = 4 * self.L * self.L  # x, y, diag1, diag2

        # Normalizing
        correlations = correlations / counts

        return distances, correlations

    def calculate_magnetisation(self):
        """
        Calculate the magnetisation of the lattice by adding up all x and y components of the spins.
        The magnetisation is the normalized length of the resulting vector.

        Returns:
        M (float): Magnetization of the lattice.
        """
        X, Y = np.sum(np.cos(self.spins)), np.sum(np.sin(self.spins))
        M = np.sqrt(X**2 + Y**2)
        return M / (self.L * self.L)

    def set_min_energy(self):
        """
        Set the lattice to a minimum energy configuration (all spins aligned).
        """
        self._calc_spins = np.zeros((3*self.L, 3*self.L))
        self.vorticity = np.zeros((self.L, self.L))
        self.total_vortices = 0
        self.total_antivortices = 0
        self.current_energy = np.zeros((self.L, self.L))

    def calculate_energy(self, theta_new=None, full_grid=False):
        """
        Calculate the energy at each point of the grid.
        If theta_new is given, the energy difference of the current vs. the new configuration is calculated.

        Args:
        theta_new (np.ndarray): New spin configuration.
        full_grid (bool): If True, the energy is returned for the entire 3L x 3L grid. The center L x L otherwise.
        """
        delta_E = np.zeros_like(self._calc_spins)

        # Shifts for the neighbors (periodic boundary conditions)
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in shifts:
            # Shift the spin lattice for the neighbors
            spins_shifted = np.roll(
                self._calc_spins, shift=(dx, dy), axis=(0, 1))
            # Calculate the energy difference
            if theta_new is None:
                delta_E += -self.J * np.cos(self._calc_spins - spins_shifted)
            else:
                delta_E += -self.J * \
                    (np.cos(theta_new - spins_shifted) -
                     np.cos(self._calc_spins - spins_shifted))

        return delta_E if full_grid else delta_E[self.L:2*self.L, self.L:2*self.L]

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
                delta_theta1 = _s[i_shifted, j_shifted] - \
                    _s[(i_shifted + 1), j_shifted]
                delta_theta2 = _s[(i_shifted + 1), j_shifted] - \
                    _s[(i_shifted + 1), (j_shifted + 1)]
                delta_theta3 = _s[(i_shifted + 1), (j_shifted + 1)
                                  ] - _s[i_shifted, (j_shifted + 1)]
                delta_theta4 = _s[i_shifted,
                                  (j_shifted + 1)] - _s[i_shifted, j_shifted]

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

    def set_single_vortex(self):
        """
        Sets a single vortex at the center of the lattice, replacing the entire _calc_spins lattice.
        """
        L = self.L
        total_size = 3 * L

        # Create grid of coordinates for the entire _calc_spins lattice
        x = np.arange(total_size)
        y = np.arange(total_size)
        xv, yv = np.meshgrid(x, y)

        # Shift coordinates so that the center of the central L x L region is at (0,0)
        x_center = total_size // 2
        y_center = total_size // 2
        xv_shifted = xv - x_center
        yv_shifted = yv - y_center

        # Calculate the angle at each point
        theta = np.arctan2(yv_shifted, xv_shifted)

        # Adjust theta to be in [0, 2*pi)
        theta = (theta + 2 * np.pi) % (2 * np.pi)

        # Replace the entire _calc_spins lattice
        self._calc_spins = theta

        # Update vorticity and energy
        self.vorticity = self.calculate_vorticity()
        self.current_energy = self.calculate_energy()
        self.current_step = 0

    def set_vortex_pair(self):
        """
        Sets a vortex-antivortex pair in the lattice, replacing the entire _calc_spins lattice.
        """
        L = self.L
        total_size = 3 * L

        # Create grid of coordinates for the entire _calc_spins lattice
        x = np.arange(total_size)
        y = np.arange(total_size)
        xv, yv = np.meshgrid(x, y)

        # Shift coordinates so that the center of the central L x L region is at (0,0)
        x_center = total_size // 2
        y_center = total_size // 2
        xv_shifted = xv - x_center
        yv_shifted = yv - y_center

        # Coordinates of the vortex centers relative to the center
        separation = L // 4  # Adjust separation as needed
        x0 = -separation
        y0 = 0
        x1 = separation
        y1 = 0

        # Calculate the angle for the first vortex
        xv_shifted_0 = xv_shifted - x0
        yv_shifted_0 = yv_shifted - y0
        theta0 = np.arctan2(yv_shifted_0, xv_shifted_0)

        # Calculate the angle for the second (anti)vortex
        xv_shifted_1 = xv_shifted - x1
        yv_shifted_1 = yv_shifted - y1
        theta1 = np.arctan2(yv_shifted_1, xv_shifted_1)

        # Total angle is the difference between the two contributions
        theta = theta0 - theta1

        # Adjust theta to be in [0, 2*pi)
        theta = (theta + 2 * np.pi) % (2 * np.pi)

        # Replace the entire _calc_spins lattice
        self._calc_spins = theta

        # Update vorticity and energy
        self.vorticity = self.calculate_vorticity()
        self.current_energy = self.calculate_energy()
        self.current_step = 0

    def metropolis_step(self, T):
        """
        Perform a single Metropolis step in the simulation.

        Args:
        T (float): Temperature
        """

        beta = 1 / T if T > 0 else np.inf
        L = self.L

        # Masks 90% of the spins to not be updated.
        # Metropolis usually updates one spin at the time.
        # Doing 10% of the spins per step is a good compromise between speed and accuracy.
        mask = np.random.rand(3*L, 3*L) < 1

        # New angles for the spins
        theta_new = np.random.uniform(0, 2*np.pi, (3*L, 3*L))

        # Energy difference for each spin
        delta_E = self.calculate_energy(theta_new, full_grid=True)

        # Metropolis acceptance criterion
        rand_vals = np.random.rand(3*L, 3*L)
        accept = ((delta_E < 0) | (rand_vals < np.exp(-beta * delta_E))) & mask

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

        # Step counter (not in use right now)
        self.current_step += 1
