import numpy as np

remaining_epsilon = np.Inf
""" The remaining epsilon in the whole domain """

start_time = None
""" The time when the analysis started """

end_time = None
""" The time when the analysis ended """

amount_of_samples = 0
""" The amount of samples that were used """

average_dim_reduction = 0.0
""" Average amount of reduced dimensions """

leafs = 0
""" Amount of subspaces """