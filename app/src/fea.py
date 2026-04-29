# from pathlib import Path

# from mpi4py import MPI
# from petsc4py.PETSc import ScalarType  # type: ignore

# import numpy as np

# import ufl
# from dolfinx import fem, io, mesh, plot
# from dolfinx.fem.petsc import LinearProblem

import numpy as np
from pynite import FEModel3D, BoundaryCondition

def analyze_structure(dome) -> FEModel3D: # Hopefully works for both Geodesic and Catenary=based domes
    """
    Create a finite element model of the dome structure, apply boundary conditions and loads, and perform a linear static analysis.
    """

    model = FEModel3D() # Create a new finite element model

    for i, node in enumerate(dome.nodes):
        name = "Apex" if i == dome.apex_id else f"Node_{i}"
        model.add_node(name, *node)

    A = 1e-4 # cross-sectional area
    Iy, Iz = 1e-8 # moment of inertia
    J = 1e-8

    model.add_material("PLA", E=2100, G=800, nu=0.3, rho=1200)  # Example material properties (E, G, nu, rho)
    model.add_section("Section", "PLA", A=A, Iy=Iy, Iz=Iz, J=J)  # Example section properties (A, Iy, Iz, J)

    for i, (n1, n2) in enumerate(dome.members):
        i_name = "Apex" if n1 == dome.apex_id else f"Node_{n1}"
        j_name = "Apex" if n2 == dome.apex_id else f"Node_{n2}"

        model.add_member(f"Member_{i}", i_node=i_name, j_node=j_name,
                        material_name="PLA", section_name="Section")
    
    for node in dome.base_ids:
        model.def_support(f"Node_{node}", support_DZ=True)  # Fix Z direction

    model.def_support(f"Node_{dome.base_ids[0]}", support_DX=True, support_DY=True, support_DZ=True)  # Fix one node completely
    model.def_support(f"Node_{dome.base_ids[1]}", support_DX=True, support_DY=False, support_DZ=True)  # Fix another node in only X and Z to prevent sliding

    model.add_node_load("Apex", direction="FZ", P=-1000.0)  # Example load (force in Z direction)

    model.analyze_linear(log=True)

    # Results
    return model