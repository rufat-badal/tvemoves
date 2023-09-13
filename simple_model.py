import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

model = pyo.ConcreteModel()

model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)
model.OBJ = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2])
model.Constraint1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)

ipopt = SolverFactory("ipopt")
ipopt.options["print_level"] = 0
ipopt.options["linear_solver"] = "mumps"
results = ipopt.solve(model)

if (results.solver.status == SolverStatus.ok) and (
    results.solver.termination_condition == TerminationCondition.optimal
):
    model.x.pprint()
else:
    print(str(results.solver))
