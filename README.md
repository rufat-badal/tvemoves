
### Thermoviscoelatic evolution of shape memory alloys via minimizing movements
The library implements the staggered minimizing movements scheme from the paper [Nonlinear and Linearized Models in Thermoviscoelasticity](https://arxiv.org/abs/2203.02375) in 2D.

The martensitic potential has currently fixed wells at [1.0 0.5; 0.0 1.0] and [1.0 -0.5; 0.0 1.0].
We support, both, P1 as well as [C1 finite elements](https://www.sciencedirect.com/science/article/pii/S0895717713000071) for the deformation. Be aware that C1 finite elements have a heave penalty due to the high degree of freedom. The space discretization should be therefore kept coarse in this case.

### Usage example
Start by fixing parameters of the simulation.
```python
params = SimulationParams(initial_temperature=0.0, search_radius=10.0, fps=3, scale=0.25)
```

Define the initial domain. Currently only rectagular domains are supported. Below a square domain of width and height of 1.0 is defined. Moreover, we fix the lower edge of the square. (It is not allowed to move during the evolution.)
```python
domain = RectangleDomain(width=1.0, height=1.0, fix=["lower"])
```

Create a simulation object given the above parameters and initial domain.
```pytho
sim = Simulation(domain, params)
```

Note that the constructor of Simulation will perform a first mechanical and thermal step. The simulations steps are saved into the `sim.steps` list.
```python
print(len(sim.steps)) # == 2
```

Run the simulation and visualize some steps.
```python
sim.run(num_steps=28)
sim.plot_step(0), sim.plot_step(9), sim.plot_step(19), sim.plot_step(29)
```