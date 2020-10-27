/*
 * MAST: Multidisciplinary-design Adaptation and Sensitivity Toolkit
 * Copyright (C) 2013-2020  Manav Bhatia and MAST authors
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

// C/C++ includes.
#include <iostream>

// libMesh includes.
#include <libmesh/libmesh.h>
#include <libmesh/parallel.h>
#include <libmesh/replicated_mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/equation_systems.h>
#include <libmesh/fe_type.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/boundary_info.h>
#include <base/eigenproblem_assembly.h>
#include <libmesh/exodusII_io.h>

// MAST includes.
#include "base/nonlinear_system.h"
#include "elasticity/structural_system_initialization.h"
#include "base/physics_discipline_base.h"
#include "boundary_condition/dirichlet_boundary_condition.h"
#include "base/parameter.h"
#include "base/constant_field_function.h"
#include "property_cards/isotropic_material_property_card.h"
#include "property_cards/solid_1d_section_element_property_card.h"
#include "base/nonlinear_implicit_assembly.h"
#include "elasticity/structural_nonlinear_assembly.h"
#include "solver/slepc_eigen_solver.h"
#include "elasticity/structural_modal_eigenproblem_assembly.h"
#include "base/physics_discipline_base.h"


int main(int argc, const char** argv)
{
    // BEGIN_TRANSLATE Extension of bar
    //
    // This example solves an axial bar extension problem. 
    //
    // Initialize libMesh library.
    libMesh::LibMeshInit init(argc, argv);

    // Create Mesh object on default MPI communicator and generate a line mesh (5 elements, 10 units long).
    //   Note that in libMesh, all meshes are parallel by default in the sense that the equations on the mesh are solved in parallel by PETSc.
    //   A "ReplicatedMesh" is one where all MPI processes have the full mesh in memory, as compared to a "DistributedMesh" where the mesh is
    //   "chunked" up and distributed across processes, each having their own piece.
    libMesh::ReplicatedMesh mesh(init.comm());
    libMesh::MeshTools::Generation::build_line(mesh, 5000, 0.0, 10.0,libMesh::EDGE3 );
    mesh.print_info();
    //mesh.boundary_info->print_info();

    // Create EquationSystems object, which is a container for multiple systems of equations that are defined on a given mesh.
    libMesh::EquationSystems equation_systems(mesh);

    // Add system of type MAST::NonlinearSystem (which wraps libMesh::NonlinearImplicitSystem) to the EquationSystems container.
    //   We name the system "structural" and also get a reference to the system so we can easily reference it later.
    MAST::NonlinearSystem & system = equation_systems.add_system<MAST::NonlinearSystem>("structural");

    system.set_eigenproblem_type(libMesh::GHEP);

    // Create a finite element type for the system. Here we use first order
    // Lagrangian-type finite elements.
    libMesh::FEType fetype(libMesh::SECOND, libMesh::LAGRANGE);

    // Initialize the system to the correct set of variables for a structural
    // analysis. In libMesh this is analogous to adding variables (each with
    // specific finite element type/order to the system for a particular
    // system of equations.
    MAST::StructuralSystemInitialization structural_system(system,
                                                           system.name(),
                                                           fetype);

    // Initialize a new structural discipline using equation_systems.
    MAST::PhysicsDisciplineBase discipline(equation_systems);

    // Create and add boundary conditions to the structural system. A Dirichlet BC fixes the left end of the bar.
    // This definition uses the numbering created by the libMesh mesh generation function.
    MAST::DirichletBoundaryCondition dirichlet_bc_left,dirichlet_bc_right;
    dirichlet_bc_left.init(0, {0,1,2,3,4,5});
    dirichlet_bc_right.init(1, {0,1,2,3,4,5});

    discipline.add_dirichlet_bc(0, dirichlet_bc_left);
    discipline.add_dirichlet_bc(1, dirichlet_bc_right);

    discipline.init_system_dirichlet_bc(system);

    // Initialize the equation system since we now know the size of our
    // system matrices (based on mesh, element type, variables in the
    // structural_system) as well as the setup of dirichlet boundary conditions.
    // This initialization process is basically a pre-processing step to
    // preallocate storage and spread it across processors.
    equation_systems.init();

    //The EigenSolver, definig which interface, i.e solver package to use.
    // solve eigenproblem around steady state

    system.eigen_solver->set_position_of_spectrum(libMesh::LARGEST_MAGNITUDE);
    system.set_exchange_A_and_B(true);
    system.set_n_requested_eigenvalues(20);
    system.initialize_condensed_dofs(discipline);

    //equation_systems.print_info();

    // Create parameters.
    MAST::Parameter thickness_y("thy", 0.05);
    MAST::Parameter thickness_z("thz", 0.05);
    MAST::Parameter E("E", 72.0e9);
    MAST::Parameter nu("nu", 0.33);
    MAST::Parameter zero("zero", 0.0);
    MAST::Parameter pressure("p", -1.0e0);
    MAST::Parameter kappa_yy("kappa_yy", 5./6.);
    MAST::Parameter kappa_zz("kappa_zz", 5./6.);
    MAST::Parameter rho("rho", 2700.0);
    MAST::Parameter offset("offset", -0.05);

    // Create ConstantFieldFunctions used to spread parameters throughout the model.
    MAST::ConstantFieldFunction thy_f("hy", thickness_y);
    MAST::ConstantFieldFunction thz_f("hz", thickness_z);
    MAST::ConstantFieldFunction E_f("E", E);
    MAST::ConstantFieldFunction nu_f("nu", nu);
    MAST::ConstantFieldFunction hyoff_f("hy_off", offset);
    MAST::ConstantFieldFunction hzoff_f("hz_off", zero);
    MAST::ConstantFieldFunction pressure_f("pressure", pressure);
    MAST::ConstantFieldFunction kappa_yy_f("Kappayy", kappa_yy);
    MAST::ConstantFieldFunction kappa_zz_f("Kappazz", kappa_zz);
    MAST::ConstantFieldFunction rho_f("rho", rho);

    // Initialize load.
    // TODO - Switch this to a concentrated/point load on the right end of the bar.
    MAST::BoundaryConditionBase right_end_pressure(MAST::SURFACE_PRESSURE);
    right_end_pressure.add(pressure_f);
    discipline.add_volume_load(0, right_end_pressure);

//    MAST::BoundaryConditionBase pressure_surf(MAST::SURFACE_PRESSURE);
//    pressure_surf.add(pressure_f);
//    discipline.add_volume_load(0,pressure_surf);

    // Create the material property card ("card" is NASTRAN lingo) and the relevant parameters to it. An isotropic
    // material needs elastic modulus (E) and Poisson ratio (nu) to describe its behavior.
    MAST::IsotropicMaterialPropertyCard material;
    material.add(E_f);
    material.add(nu_f);
    material.add(rho_f);

    // Create the section property card. Attach all property values.
    MAST::Solid1DSectionElementPropertyCard section;
    section.add(thy_f);
    section.add(thz_f);
    section.add(hyoff_f);
    section.add(hzoff_f);
    section.add(kappa_yy_f);
    section.add(kappa_zz_f);

    // Specify a section orientation point and add it to the section.
    RealVectorX orientation = RealVectorX::Zero(3);
    orientation(1) = 1.0;
    section.y_vector() = orientation;

    // Attach material to the card.
    section.set_material(material);
    section.set_strain(MAST::NONLINEAR_STRAIN);

    // Initialize the section and specify the subdomain in the mesh that it applies to.
    section.init();
    discipline.set_property_for_subdomain(0, section);

    // Create nonlinear assembly object and set the discipline and
    // structural_system. Create reference to system.

    MAST::NonlinearImplicitAssembly assembly;
    MAST::StructuralNonlinearAssemblyElemOperations elem_ops;
    MAST::EigenproblemAssembly                               modal_assembly;
    MAST::StructuralModalEigenproblemAssemblyElemOperations  modal_elem_ops;

    assembly.set_discipline_and_system(discipline, structural_system);
    elem_ops.set_discipline_and_system(discipline, structural_system);
    modal_assembly.set_discipline_and_system(discipline, structural_system); // modf_w
    modal_elem_ops.set_discipline_and_system(discipline, structural_system);

    // Zero the solution before solving.
    system.solution->zero();

    // Solve the system and print displacement degrees-of-freedom to screen.
    system.solve(elem_ops, assembly);

    libMesh::out << "Writing solution to : solution.exo" << std::endl;
    libMesh::ExodusII_IO(mesh).write_equation_systems("solution.exo",
                                                        equation_systems);
    //system.solution->print_global();

    modal_assembly.set_base_solution(*system.solution);
    system.eigenproblem_solve( modal_elem_ops, modal_assembly);



    libMesh::ExodusII_IO writer(mesh) ;
    std::vector<libMesh::NumericVector<Real> *> _basis;
    unsigned int
            nconv = std::min(system.get_n_converged_eigenvalues(),
                             system.get_n_requested_eigenvalues());

    _basis.resize(nconv);

    for (unsigned int i = 0; i < nconv; i++) {

        // create a vector to store the basis

            _basis[i] = (system.solution->zero_clone().release()); // what happens in this line ?

        // now write the eigenvalue
        Real
                re = 0.,
                im = 0.;
        system.get_eigenpair(i, re, im, *_basis[i]);

        libMesh::out
                << std::setw(35) << std::fixed << std::setprecision(15)
                << re << std::endl;

            // copy the solution for output
            system.solution->zero();
            system.solution->add(*_basis[i]);

            // We write the file in the ExodusII format.
            writer.write_timestep("modes.exo",
                                   equation_systems,
                                   i + 1, //  time step
                                   i);    //  time

    }

    modal_assembly.clear_base_solution();
    assembly.clear_discipline_and_system();
    elem_ops.clear_discipline_and_system();

    modal_assembly.clear_discipline_and_system();
    modal_elem_ops.clear_discipline_and_system();

    for (unsigned int i = 0; i < nconv; i++)
       delete  _basis[i];
    
    // END_TRANSLATE
    return 0;
}
