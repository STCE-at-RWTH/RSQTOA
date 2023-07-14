
epsilon = 10e-2
""" The overall error threshold """

sampling_strategy = "latin"
""" Sampling strategy. Possible values:
    random, latin, latin_maximin, latin_corr """

regression_samples = 10
""" Amount of points at which a subspace regression is performed """

test_samples = 10
""" Amount of points at which the regression is tested in the entire domain """

test_values = 10
""" Amount of values which are tested with the test points """

debug = False
""" Debug mode flag """

# ---------------------------------------------------------------------------- #

max_taylor_order = 1
""" Maximum taylor order """

taylor_regression_samples = 0
""" Amount of samples used to improve the taylor regression.
    Taylor regression is always enabled and uses at least the minimum amount
    of samples such that the least squares problem is never underdetermined. """

periodic_regression_samples = 50
""" Amount of samples used to improve the periodic regression.
    Set to zero to disable periodic regression. """

exponential_regression_samples = 50
""" Amount of samples used to do the exponential regression.
    Set to zero to disable exponential regression. """

regression_selection = "greedy"
""" Regression selection strategy. Possible values:
    greedy, best """

# ---------------------------------------------------------------------------- #

subspace_ordering = "default"
""" The ordering strategy for the remaining dimensions. Possible values:
    default, max_error, min_error, area """

subspace_depth = 2
""" Maximum recursive subspace divisions """

subspace_threshold = 1e-1
""" Minimum subspace domain size """

subspace_needs_regression = True
""" Whether we allow subspaces with no regression or not """

subspace_selection = "greedy"
""" Subspace selection strategy. Possible values:
    greedy, best """

subspace_expansion_steps = 0
""" Amount of steps in the adaptive subspace expansion """

subspace_reduction_steps = 0
""" Amount of steps in the adaptive subspace reduction """

# ---------------------------------------------------------------------------- #

regressions = []
""" List which stores the current regressions """

reduced_model = None
""" Reduced model """

reduced_dims = 0
""" Reduced amount of dimensions """

reduced_lb = 0
""" Lower boundary of the reduced model domain """

reduced_ub = 0
""" Upper boundary of the reduced model domain """