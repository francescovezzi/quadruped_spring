import cvxpy as cp
import numpy as np
from cvxpygen import cpg
import pinocchio as pin

n = 12
m = 6
l=4
A = cp.Parameter((m,n),name="A")
bd = cp.Parameter(m,name="bd")
frict_coeff = 1
Fz_max = 40
F_contact = cp.Parameter(l,name="F_contact")

# S = cp.Parameter((m,m),name="S")
np.random.seed(0)
A.value = np.random.randn(m,n)
bd.value = np.random.randn(m)
F_contact.value = np.random.randn(l)

S = 80*np.eye(m)


F = cp.Variable(n,name="F")
constr = []

for foot in range(4):
    # index [foot*3+2] means Fz of foot X.
    constr += [F_contact[foot]*F[foot * 3 + 2] == 0]
    constr += [0 <= F[foot * 3 + 2], F[foot * 3 + 2] <= Fz_max]
    constr += [-frict_coeff * F[foot * 3 + 2] <= F[foot * 3], F[foot * 3] <= frict_coeff * F[foot * 3 + 2]]
    constr += [-frict_coeff * F[foot * 3 + 2] <= F[foot * 3 + 1], F[foot * 3 + 1] <= frict_coeff * F[foot * 3 + 2]]

cost = cp.quad_form((A @ F - bd),S) 

prob = cp.Problem(cp.Minimize(cost), constr)

# prob.solve()

cpg.generate_code(prob,code_dir='test_code_gen',solver='ECOS')