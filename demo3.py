# add a param selector to the plot
# add a legend to the plot
# later: stationary point solver method
# plot phase space or transient
# define initial conditions
# define new function
# Jacobian plotter?
# wrap matplotlib into function on ode objs

from scipy.integrate import solve_ivp

import numpy as np

from math import pi, sin

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

################

class ode:
    def __init__(self, fun, defaults, labels):
        self.fun        = fun
        self.num_params = len(defaults)
        self.defaults   = defaults
        self.labels     = labels
        
    def solve(self,params=None,fun=None):
        
        if params is None:
            params = self.defaults
        if fun is None:
            fun = self.fun
        
        sol = solve_ivp(
             fun=fun, 
             t_span=t_span, 
             y0=y0,
             t_eval=t_eval,
             dense_output=False,
             args=params,
             method='RK23'
             )
        return sol.y

def lotkavolterra_fun(t, z, a, b, c, d):
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]

def vanderpol_fun(t, z, mu):
    x, y = z
    return [mu*(x-x**3/3-y), x/mu]

################

t_eval = np.linspace(0, 10, 100)
t_span = [0, 10]
y0 = [10, 5]

################

# create slider for specified parameter
def makeslide(default,label,index,plot_bott):
    myaxis = plt.axes(
        [0.25, plot_bott-0.15-index*0.1, 0.65, 0.03], 
        facecolor='lightgoldenrodyellow'
    )
    myslider = Slider(
        ax=myaxis,
        label=label,
        valmin=0.01,
        valmax=20,
        valinit=default,
        orientation="horizontal"
    )
    return myslider

# plot transients for ode object
def odeplot(ode):
    
    # update lines when sliders are moved
    def update(val):
        params = tuple(slider.val for slider in sliders)
        for i,j in zip(lines,ode.solve(params=params)):
            i.set_ydata(j)
        fig.canvas.draw_idle()
        return None
    
    # reset sliders to initial positions
    def reset(event):
        for i in sliders: i.reset()
        return None    
    
    # create fig
    fig, ax = plt.subplots()
    lines = plt.plot(
        t_eval, 
        np.transpose(ode.solve()),
        lw=2
        )
    ax.set_xlabel('Time [s]')
    ax.set_ylim((-20,20))
    ax.margins(x=0)
    
    # adjust fig to make room for sliders
    plot_bott = 0.15 + 0.1 * ode.num_params
    plt.subplots_adjust(
        left=0.25, 
        bottom = plot_bott
        )
    
    # make sliders and register update function with each
    sliders=[
        makeslide(i,j,k,plot_bott)
        for i,j,k in 
        zip(
            ode.defaults,
            ode.labels,
            range(ode.num_params)
            )
        ]
    for i in sliders: i.on_changed(update)
    
    # create reset button
    resetax = plt.axes([0.8, 0.01, 0.1, 0.04])
    button = Button(
        resetax,  
        'Reset', 
        color='lightgoldenrodyellow', 
        hovercolor='0.975'
        )
    button.on_clicked(reset)
    
    return plt.show()

#######################

def lotkavolterra_fun(t, z, a, b, c, d):
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]

def vanderpol_fun(t, z, mu):
    x, y = z
    return [mu*(x-x**3/3-y), x/mu]

def forced_vdp_fun(t, z, mu, A, omega):
    x, y = z
    return [y, mu*(1-x**2)*y - x + A*sin(omega*t)]

lotkavolterra = ode(
    lotkavolterra_fun, 
    (10, 1, 10 ,1), 
    ("a", "b", "c", "d")
    )
odeplot(lotkavolterra)

# vanderpol = ode(
#     vanderpol_fun, 
#     (1,), 
#     ("Damping",)
#     )
# odeplot(vanderpol)

# forced_vdp = ode(
#     forced_vdp_fun, 
#     (0.01, 10, pi),
#     ("Damping","Amplitude","Angular Velocity")
#     )
# odeplot(forced_vdp)