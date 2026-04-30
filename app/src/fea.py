# from pathlib import Path

# from mpi4py import MPI
# from petsc4py.PETSc import ScalarType  # type: ignore

# import numpy as np

# import ufl
# from dolfinx import fem, io, mesh, plot
# from dolfinx.fem.petsc import LinearProblem

import numpy as np
from Pynite import FEModel3D

YIELD_STRESS = 5e7   # Pa, PLA
DENSITY = 1240       # kg/m^3, PLA
REFERENCE_LOAD = 1000.0  # N, downward apex load used for linear analysis

def compute_model(model):
    """
    Compute the finite element model and print the results.
    """
    model.analyze_linear()

    # rndr = Rendering.Renderer(model)
    # rndr.labels = False
    # rndr.annotation_size = 0
    # rndr.deformed_shape = True
    # rndr.render_model()
    
    return model

def analyze_structure(dome, thicknesses) -> FEModel3D: # Hopefully works for both Geodesic and Catenary=based domes
    """
    Create a finite element model of the dome structure, apply boundary conditions and loads, and perform a linear static analysis.

    thicknesses: array of solid-circular rod radii (m), one per member, in
    the same order as dome.members.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    assert thicknesses.shape == (len(dome.members),), (
        f"need {len(dome.members)} thicknesses, got {thicknesses.shape}"
    )

    model = FEModel3D() # Create a new finite element model

    for i, node in enumerate(dome.nodes):
        name = "Apex" if i == dome.apex_id else f"Node_{i}"
        model.add_node(name, *node)

    model.add_material("PLA", E=3.5e9, G=1.3e9, nu=0.35, rho=DENSITY)  # PLA material properties (E, G, nu, rho) in N, m, Pa

    for i, ((n1, n2), r) in enumerate(zip(dome.members, thicknesses)):
        # Solid circular rod cross-section sized from per-member radius gene
        A = np.pi * r * r            # cross-sectional area
        I = np.pi * r ** 4 / 4.0     # moment of inertia (Iy = Iz)
        J = np.pi * r ** 4 / 2.0     # polar moment of inertia
        sec_name = f"Sec_{i}"
        model.add_section(sec_name, A, I, I, J)  # Section sized for member i

        i_name = "Apex" if n1 == dome.apex_id else f"Node_{n1}"
        j_name = "Apex" if n2 == dome.apex_id else f"Node_{n2}"

        model.add_member(f"Member_{i}", i_node=i_name, j_node=j_name,
                        material_name="PLA", section_name=sec_name)

    for node in dome.base_ids:
        model.def_support(f"Node_{node}", support_DZ=True)  # Fix Z direction

    model.def_support(f"Node_{dome.base_ids[0]}", support_DX=True, support_DY=True, support_DZ=True)  # Fix one node completely
    model.def_support(f"Node_{dome.base_ids[1]}", support_DX=True, support_DY=False, support_DZ=True)  # Fix another node in only X and Z to prevent sliding

    model.add_node_load("Apex", direction="FZ", P=-REFERENCE_LOAD)  # Reference load (1 kN downward) for linear scaling

    compute_model(model)

    # Results
    return model


def force_at_first_failure(model, thicknesses,
                           reference_load=REFERENCE_LOAD,
                           yield_stress=YIELD_STRESS):
    """Linearly scale the reference load to the load at which the worst-case
    axial stress first reaches yield. Returns the failure force (N).

    v1 limitation: uses axial stress only, not combined axial + bending,
    and does not account for buckling.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    max_stress = 0.0
    for i, r in enumerate(thicknesses):
        A = np.pi * r * r
        member = model.members[f"Member_{i}"]
        f_axial = max(abs(member.max_axial()), abs(member.min_axial()))  # worst-case axial force magnitude
        sigma = f_axial / A
        if sigma > max_stress:
            max_stress = sigma
    if max_stress <= 0.0:
        return float("inf")
    return reference_load * yield_stress / max_stress


def total_mass(dome, thicknesses, density=DENSITY):
    """Sum of length * cross-section area * density across members (kg)."""
    thicknesses = np.asarray(thicknesses, dtype=float)
    mass = 0.0
    for (n1, n2), r in zip(dome.members, thicknesses):
        L = float(np.linalg.norm(dome.nodes[n1] - dome.nodes[n2]))
        A = np.pi * r * r
        mass += L * A * density
    return mass


def specific_strength(model, dome, thicknesses):
    """GA fitness: failure force divided by total mass (N/kg)."""
    f = force_at_first_failure(model, thicknesses)
    m = total_mass(dome, thicknesses)
    return f / m if m > 0 else 0.0