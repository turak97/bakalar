


# preprocess the data and fits and returns the model
def lin_2D_regression(x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    reg = linear_model.LinearRegression()
    reg.fit(x, y)

    return reg


# prob = float from range [0, 1]; 0 = always False, 1 = always True
def throw_away(prob):
    if prob == 0:
        return False
    return prob > random.random()


# returns the beginning and the end of the line as a pair (x1, x2), (y1, y2)
# p = linear coefficient, q = constant (as y = px + q)
def lin_coef(p, q, x_vals):
    print(p, q, x_vals)
    x = np.array([x_vals[0], x_vals[len(x_vals) - 1]])
    y = np.array([x[0] * p + q, x[1] * p + q])
    return x, y


# range is a closed interval [a, b]
# TODO disp = cislo, kterym se bude krmit funce predana jako disp
def generate_lin_data(p=1, q=0, interval=(0, 10), step=0, disp=0, mode='uniform'):
    if step == 0:
        step = (interval[1] - interval[0])/20
    x_val = interval[0]
    y_arr = []
    x_arr = []
    while x_val <= interval[1]:
        x_arr.append(x_val)

        out = random.uniform(-disp, disp)
        y_val = p*x_val + q + out
        y_arr.append(y_val)
        x_val += step
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    return x_arr, y_arr


# clustering is a float from range [0, 1]; 0 = continuous, -> 0 ... one big cluster, -> 1 ... more smaller clusters
# ferocity is a float from range [0, inf); smaller values (around 0.1) makes better polynoms
# disp is weight of dispersion, which is calculated in the interval +- square root of calculated x_val
def gen_pol_data(polynom=np.array([1/100, 1/5, -1, 1]),
                 interval=(-100, 100),
                 clustering=0.5,
                 disp=1,
                 step=20,
                 seed=-1,
                 num_of_outliers=2):
    scale = interval[1] - interval[0]
    if step == 0:
        step = scale/40
    if seed < 0:
        seed = random.randint(0, int(scale * 2))
    x_val = interval[0]
    x_arr, y_arr = [], []

    # generate y values
    f = np.poly1d(polynom)
    while x_val <= interval[1]:
        prob = 0
        if clustering != 0:
            prob = math.sin(x_val * clustering / (scale / 40) + seed)/2 + 1/2
        if throw_away(prob):
            x_val += step
            continue

        y_val = f(x_val)
        y_arr.append(y_val)
        x_arr.append(x_val)
        x_val += step

    if disp != 0:
        y_arr = disperse(y_arr, disp)

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    return [x_arr, y_arr]