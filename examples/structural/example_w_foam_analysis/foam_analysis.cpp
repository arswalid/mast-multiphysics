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
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh_refinement.h>

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
#include <property_cards/solid_2d_section_element_property_card.h>
#include <base/eigenproblem_assembly.h>
#include <elasticity/stress_assembly.h>
#include <elasticity/stress_output_base.h>
#include <fstream>
#include <examples/base/input_wrapper.h>
#include <examples/old/structural/stiffened_plate_optimization/stiffened_plate_optimization_base.h>
#include "examples/structural/base/foam.h"



int main(int argc, const char** argv)
{


    // BEGIN_TRANSLATE Extension of bar
    //
    // This example solves an axial bar extension problem.
    //
    // Initialize libMesh library.
    libMesh::LibMeshInit init(argc, argv);

    MAST::Examples::GetPotWrapper input(argc, argv, "input");


    bool if_vk = input("if_vk",  "account for geometric nonlinearities ", false);
    bool if_continuation_solver = input("continuation_solver",  "use the continuation solver  ", false);
    int n_eig  = input("n_eig",  "number of eigeinvalues", 20);
    int nl_steps = input("nl_steps",  "number of steps in nonlinear solver", 10);

    // Create Mesh object on default MPI communicator and generate a line mesh (5 elements, 10 units long).
    //   Note that in libMesh, all meshes are parallel by default in the sense that the equations on the mesh are solved in parallel by PETSc.
    //   A "ReplicatedMesh" is one where all MPI processes have the full mesh in memory, as compared to a "DistributedMesh" where the mesh is
    //   "chunked" up and distributed across processes, each having their own piece.
    libMesh::ReplicatedMesh mesh(init.comm());



    Real    length = input("length",  "length of the structure ", 2.),
            width  = input("width",    "width of the structure ", 1.),
            height = input("height",  "height of the structure ", 1.);

    unsigned int n_divs_x      = input("n_divs_x",  "number of unit cells in the x dir", 2),
                 n_divs_y      = input("n_divs_y",  "number of unit cells in the y dir", 1),
                refinement_lvl = input("refinement_lvl",  "refinement of the mesh (n_elems^refinement_lvl)", 0);
    START_LOG("foam_mesh()","FoamMesh")
    MAST::FoamMesh foam_mesh;
    foam_mesh.init(n_divs_x,
                   n_divs_y,
                   length,
                   width,
                   height,
                   mesh,
                   refinement_lvl);
    STOP_LOG("foam_mesh()","FoamMesh")
    mesh.print_info();
//    mesh.boundary_info->print_info();



    // create the equation system
    libMesh::EquationSystems eq_sys(mesh);
    // create the libmesh system
    MAST::NonlinearSystem* sys = &(eq_sys.add_system<MAST::NonlinearSystem>("structural"));
    // FEType to initialize the system
    libMesh::FEType fetype (libMesh::FIRST, libMesh::LAGRANGE);
    // specifying the type of eigenproblem we'd like to solve
    sys->set_eigenproblem_type(libMesh::GHEP);
    // initialize the system to the right set of variables
     MAST::StructuralSystemInitialization structural_sys(*sys,
                                                        sys->name(),
                                                        fetype);

     MAST::PhysicsDisciplineBase discipline(eq_sys);

   // bc's

    // Create and add boundary conditions to the structural system. A Dirichlet BC fixes the left end of the bar.
    // This definition uses the numbering created by the libMesh mesh generation function.
    MAST::DirichletBoundaryCondition dirichlet_bc_bottom_toppannel;
    MAST::DirichletBoundaryCondition dirichlet_bc_right_toppannel;
    MAST::DirichletBoundaryCondition dirichlet_bc_top_toppannel;
    MAST::DirichletBoundaryCondition dirichlet_bc_left_toppannel;

    dirichlet_bc_bottom_toppannel.init(4, structural_sys.vars());
    dirichlet_bc_right_toppannel.init(5, structural_sys.vars());
    dirichlet_bc_top_toppannel.init(6, structural_sys.vars());
    dirichlet_bc_left_toppannel.init(7, structural_sys.vars());

    discipline.add_dirichlet_bc(4, dirichlet_bc_bottom_toppannel);
    discipline.add_dirichlet_bc(5, dirichlet_bc_right_toppannel);
    discipline.add_dirichlet_bc(6, dirichlet_bc_top_toppannel);
    discipline.add_dirichlet_bc(7, dirichlet_bc_left_toppannel);

    MAST::DirichletBoundaryCondition dirichlet_bc_bottom_bottompannel;
    MAST::DirichletBoundaryCondition dirichlet_bc_right_bottompannel;
    MAST::DirichletBoundaryCondition dirichlet_bc_top_bottompannel;
    MAST::DirichletBoundaryCondition dirichlet_bc_left_bottompannel;

    dirichlet_bc_bottom_bottompannel.init(0, structural_sys.vars());
    dirichlet_bc_right_bottompannel.init(1, structural_sys.vars());
    dirichlet_bc_top_bottompannel.init(2, structural_sys.vars());
    dirichlet_bc_left_bottompannel.init(3, structural_sys.vars());

    discipline.add_dirichlet_bc(0, dirichlet_bc_bottom_bottompannel);
    discipline.add_dirichlet_bc(1, dirichlet_bc_right_bottompannel);
    discipline.add_dirichlet_bc(2, dirichlet_bc_top_bottompannel);
    discipline.add_dirichlet_bc(3, dirichlet_bc_left_bottompannel);

    discipline.init_system_dirichlet_bc(*sys);


    // initialize the equation system
    eq_sys.init();

    //The EigenSolver, definig which interface, i.e solver package to use.
    sys->eigen_solver->set_position_of_spectrum(libMesh::LARGEST_MAGNITUDE);

    //sets the flag to exchange the A and B matrices for a generalized eigenvalue problem.
    //This is needed typically when the B matrix is not positive semi-definite.
    sys->set_exchange_A_and_B(true);

    //sets the number of eigenvalues requested
    sys->set_n_requested_eigenvalues(n_eig);

    //Loop over the dofs on each processor to initialize the list of non-condensed dofs.
    //These are the dofs in the system that are not contained in global_dirichlet_dofs_set.
    sys->initialize_condensed_dofs(discipline);

    //eq_sys.print_info();

    //////////////////////





    MAST::Parameter             p_cav("p_cav", input("p_cav",  "mechanical load", -1.e6));
    MAST::ConstantFieldFunction p_cav_f("pressure", p_cav);

    MAST::Parameter zero("zero", 0.);
    MAST::ConstantFieldFunction zero_f("zero_constant_field", zero);

    MAST::ConstantFieldFunction ref_temp_f("ref_temperature", zero);
    MAST::Parameter  temp("temperature",  input("temp",  "thermal load", 10.));
    MAST::ConstantFieldFunction temp_f("temperature", temp);

    MAST::BoundaryConditionBase T_load(MAST::TEMPERATURE);
    T_load.add(temp_f);
    T_load.add(ref_temp_f);

//    discipline.add_volume_load(0, T_load);          // for the panel
//    discipline.add_volume_load(1, T_load);   // for the beams

    // pressure load
    MAST::BoundaryConditionBase p_load(MAST::SURFACE_PRESSURE);
    p_load.add(p_cav_f);
    //discipline.add_volume_load(0, p_load);          // for the panel
    discipline.add_volume_load(0, p_load);          // for the panel

    /////////////////////////////////

    MAST::Parameter E("E", 72.0e9);
    MAST::Parameter nu("nu", 0.33);
    MAST::Parameter alpha("alpha", 2.5e-5);
    MAST::Parameter rho("rho", 2700.);

    MAST::ConstantFieldFunction E_f("E", E);
    MAST::ConstantFieldFunction nu_f("nu", nu);
    MAST::ConstantFieldFunction alpha_f("alpha", alpha);
    MAST::ConstantFieldFunction rho_f("rho", rho);

    // Create the material property card ("card" is NASTRAN lingo) and the relevant parameters to it. An isotropic
    // material needs elastic modulus (E) and Poisson ratio (nu) to describe its behavior.
    MAST::IsotropicMaterialPropertyCard material;
    material.add(E_f);
    material.add(nu_f);
    material.add(rho_f);
    material.add(alpha_f);

    /////////////////////////////////


    // plates
    // Create the section property card. Attach all property values.
    MAST::Parameter kappa("kappa", 5./6.);
    MAST::Parameter thickness("th", input("th_plate",  "thickness of top and bottom sheet", 0.001));

    MAST::ConstantFieldFunction kappa_f("kappa",  kappa);
    MAST::ConstantFieldFunction h_f("h",  thickness);
    MAST::ConstantFieldFunction off_f("off", zero);


    MAST::Solid2DSectionElementPropertyCard section_plate;
    // add the section properties to the card
    section_plate.add(h_f);
    section_plate.add(kappa_f);
    section_plate.add(off_f);
    if (if_vk) section_plate.set_strain(MAST::NONLINEAR_STRAIN);

    // tell the section property about the material property
    section_plate.set_material(material);

    discipline.set_property_for_subdomain(0, section_plate);
    discipline.set_property_for_subdomain(1, section_plate);


    // beams
    // Create parameters.
    MAST::Parameter thickness_y("thy", input("width_beam",  "width of beams", 0.0005));
    MAST::Parameter thickness_z("thz", input("height_beam", "height of beams",0.0005));
    MAST::Parameter kappa_yy("kappa_yy", 5./6.);
    MAST::Parameter kappa_zz("kappa_zz", 5./6.);

    MAST::ConstantFieldFunction kappa_yy_f("Kappayy", kappa_yy);
    MAST::ConstantFieldFunction kappa_zz_f("Kappazz", kappa_zz);
    MAST::ConstantFieldFunction thy_f("hy", thickness_y);
    MAST::ConstantFieldFunction thz_f("hz", thickness_z);
    MAST::ConstantFieldFunction hyoff_f("hy_off", zero);
    MAST::ConstantFieldFunction hzoff_f("hz_off", zero);


    std::vector<MAST::Solid1DSectionElementPropertyCard *> section_beam(mesh.n_subdomains() - 2, nullptr);


    libMesh::MeshBase::const_element_iterator
            el_it = mesh.elements_begin(),
            el_end = mesh.elements_end();

    RealVectorX direction = RealVectorX::Zero(3),
            A= RealVectorX::Zero(3),
            B= RealVectorX::Zero(3),
            normal= RealVectorX::Zero(3);
    Real k = 0;
    unsigned int ind = 0;

    for (; el_it != el_end; el_it++) {
        libMesh::Elem *old_elem = *el_it;
        // add boundary condition tags for the panel boundary
        if (old_elem->subdomain_id() > 1){

            ind = old_elem->subdomain_id() - 2;

            if (section_beam[ind] == nullptr) {

                if (old_elem->subdomain_id() < 14 ) {
                    section_beam[ind] = new MAST::Solid1DSectionElementPropertyCard;

                    section_beam[ind]->add(thy_f);
                    section_beam[ind]->add(thz_f);
                    section_beam[ind]->add(hyoff_f);
                    section_beam[ind]->add(hzoff_f);
                    section_beam[ind]->add(kappa_yy_f);
                    section_beam[ind]->add(kappa_zz_f);
                    if (if_vk) section_beam[ind]->set_strain(MAST::NONLINEAR_STRAIN);

                    A(0) = old_elem->point(0)(0);
                    A(1) = old_elem->point(0)(1);
                    A(2) = old_elem->point(0)(2);

                    B(0) = old_elem->point(1)(0);
                    B(1) = old_elem->point(1)(1);
                    B(2) = old_elem->point(1)(2);

                    direction = B - A;
                    direction /= direction.norm();

                    k = -(direction(0) * A(0) + direction(1) * A(1) + direction(2) * A(2)) /
                        (pow(direction(0), 2) + pow(direction(1), 2) + pow(direction(2), 2));

                    normal(0) = direction(0) * k + A(0);
                    normal(1) = direction(1) * k + A(1);
                    normal(2) = direction(2) * k + A(2);

                    if (normal.norm() < 1.e-8)
                        libmesh_error();

                    normal /= normal.norm();

                    section_beam[ind]->y_vector() = normal;

                    // Attach material to the card.
                    section_beam[ind]->set_material(material);

                    // Initialize the section and specify the subdomain in the mesh that it applies to.
                    section_beam[ind]->init();

                    discipline.set_property_for_subdomain(old_elem->subdomain_id(), *section_beam[ind]);
                }
                else if (old_elem->subdomain_id() >= 14){
                    section_beam[ind] = new MAST::Solid1DSectionElementPropertyCard;

                    section_beam[ind]->add(thy_f);
                    section_beam[ind]->add(thz_f);
                    section_beam[ind]->add(hyoff_f);
                    section_beam[ind]->add(hzoff_f);
                    section_beam[ind]->add(kappa_yy_f);
                    section_beam[ind]->add(kappa_zz_f);
                    if (if_vk) section_beam[ind]->set_strain(MAST::NONLINEAR_STRAIN);

                    if (old_elem->subdomain_id() == 14) {
                        normal(0) = 1;
                        normal(1) = 0;
                        normal(2) = 0;
                    }
                    else if (old_elem->subdomain_id() == 15) {
                        normal(0) = 0;
                        normal(1) = 1;
                        normal(2) = 0;
                    }
                    else if (old_elem->subdomain_id() == 16) {
                        normal(0) = 0;
                        normal(1) = 0;
                        normal(2) = 1;
                    }
                    else {libmesh_error();}

                    section_beam[ind]->y_vector() = normal;

                    // Attach material to the card.
                    section_beam[ind]->set_material(material);

                    // Initialize the section and specify the subdomain in the mesh that it applies to.
                    section_beam[ind]->init();

                    discipline.set_property_for_subdomain(old_elem->subdomain_id(), *section_beam[ind]);

                }
            }
        }

    }

    libmesh_assert_equal_to(section_beam.size(), mesh.n_subdomains() - 2);

    // Specify a section orientation point and add it to the section.
//    RealVectorX orientation = RealVectorX::Zero(3);
//    orientation(0) = 0.0;
//    orientation(1) = 1;
//    orientation(2) = 0.0;

//    orientation /= orientation.norm();


    ////////////////////   solving the problem       ////////////////////////
    // Create nonlinear assembly object and set the discipline and
    // structural_system. Create reference to system.
    libMesh::ExodusII_IO writer(sys->get_mesh());

    MAST::NonlinearImplicitAssembly assembly;
    MAST::StructuralNonlinearAssemblyElemOperations elem_ops;
    assembly.set_discipline_and_system(discipline, structural_sys);
    elem_ops.set_discipline_and_system(discipline, structural_sys);
    MAST::NonlinearSystem& nonlinear_system = assembly.system();

    ////////////////////   defining point at center of bottom plate       ////////////////////////

    libMesh::Point
            pt((length)/2., (width)/2., 0.), // location of mid-point before shift
    pt0,
            R = 0, //radius of the circle where the circumference defines the curved plate
    dr1, dr2;
    const libMesh::Node
            *nd = nullptr;

    // if a finite radius is defined, change the mesh to a circular arc of specified radius
    libMesh::MeshBase::node_iterator
            n_it   = mesh.nodes_begin(),
            n_end  = mesh.nodes_end();

    // initialize the pointer to a node
    nd   = *n_it;
    pt0  = *nd;

    for ( ; n_it != n_end; n_it++) {

        dr1  = pt0;
        dr1 -= pt;

        dr2  = **n_it;
        dr2 -= pt;

        if (dr2.norm() < dr1.norm()) {

            nd  = *n_it;
            pt0 = *nd;
        }
    }

    const unsigned int
            dof_num = nd->dof_number(0, 2, 0);
    // first solve the the temperature increments
    std::vector<Real> vec1;
    std::vector<unsigned int> vec2 = {dof_num};

    ////////////////////   solving static problem        ////////////////////////

    std::ofstream out;   // text file for nl solution
    std::ofstream out_eig;  // text file for eigenvalues

    if (init.comm().rank() == 0) {

        out.open("continuation_solver_load.txt", std::ofstream::out);
        out
                << std::setw(10) << "iter"
                << std::setw(25) << "temperature"
                << std::setw(25) << "pressure"
                << std::setw(25) << "displ" << std::endl;

        out_eig.open("continuation_solver_eig.txt", std::ofstream::out);
        out_eig
                << std::setw(10) << "iter"
                << std::setw(25) << "temperature"
                << std::setw(25) << "pressure";

        for (int di = 0; di < n_eig; di++)
            out_eig  << std::setw(25) << "Re_of_eigenvalue" << di+1;

        out_eig << std::endl;
    }

    // Zero the solution before solving.
    sys->solution->zero();

    // set the number of load steps
    unsigned int
            n_steps = 1,
            inc = 0 ;
    if (if_vk) n_steps = nl_steps;
    Real
            P0      = (p_cav)();

    unsigned int i;
    for ( i = 0; i < n_steps; i++) {

        (p_cav)() = P0*(i+1.)/(1.*n_steps);

        libMesh::out
                << "Load step: " << i
                << "  : T = " << (temp)()
                << "  : p = " << (p_cav)()
                << std::endl;
        // Solve the system and print displacement degrees-of-freedom to screen.
        sys->solve(elem_ops, assembly);

        sys->solution->localize(vec1, vec2);

        // write the value to the load.txt file
        if (init.comm().rank() == 0) {
            out
                    << std::setw(10) << i
                    << std::setw(25) << (temp)()
                    << std::setw(25) << (p_cav)()
                    << std::setw(25) << vec1[0] << std::endl;
        }


        //  we can then write the solution into the .exo file which will contain all variables
        writer.write_timestep("sol_n_r_solver.exo",
                               eq_sys,
                               inc + 1,
                               inc + 1);
        inc++;
    }



    // us this solution as the base solution later if no flutter is found.
    libMesh::NumericVector<Real> &
            static_sol = *sys->solution;

    ////////////////////   printing stress dist to exodus file       ////////////////////////
    MAST::StressAssembly stress_assembly;
    MAST::StressStrainOutputBase stress_elem;

    stress_elem.set_aggregation_coefficients(2.,1.,2,1.1e9);
    stress_elem.set_participating_elements_to_all();
    stress_elem.set_discipline_and_system(discipline,structural_sys);
    stress_assembly.set_discipline_and_system(discipline,structural_sys);
    stress_assembly.update_stress_strain_data(stress_elem, static_sol);

    libMesh::out << "Writing output to : output.exo" << std::endl;

    //std::set<std::string> nm;
    //nm.insert(_sys->name());
    // write the solution for visualization
    libMesh::ExodusII_IO(mesh).write_equation_systems("output.exo",
                                                      eq_sys);//,&nm);

    stress_elem.clear_discipline_and_system();
    stress_assembly.clear_discipline_and_system();

    ////////////////////   solving the eigenvalue problem       ////////////////////////

    MAST::EigenproblemAssembly                               modal_assembly;
    MAST::StructuralModalEigenproblemAssemblyElemOperations  modal_elem_ops;

    modal_assembly.set_discipline_and_system(discipline, structural_sys); // modf_w
    modal_elem_ops.set_discipline_and_system(discipline, structural_sys);

    modal_assembly.set_base_solution(static_sol);
    sys->eigenproblem_solve( modal_elem_ops, modal_assembly);

//    std::vector<libMesh::NumericVector<Real> *> basis;

    unsigned int
            nconv = std::min(sys->get_n_converged_eigenvalues(),
                             sys->get_n_requested_eigenvalues());
//    if (basis.size() > 0)
//        libmesh_assert(basis.size() == nconv);
//    else {
//        basis.resize(nconv);
//        for (unsigned int i = 0; i < basis.size(); i++)
//            basis[i] = nullptr;
//    }



    libMesh::ExodusII_IO writer_eig(sys->get_mesh());

    if (init.comm().rank() == 0) {
        out_eig
                << std::setw(10) << 0
                << std::setw(25) << (temp)()
                << std::setw(25) << (p_cav)();
    }


    for (unsigned int i = 0; i < nconv; i++) {

//        // create a vector to store the basis
//        if (basis[i] == nullptr)
//            basis[i] = sys->solution->zero_clone().release(); // what happens in this line ?

        // now write the eigenvalue
        Real
                re = 0.,
                im = 0.;
//        sys->get_eigenpair(i, re, im, *basis[i]);
        sys->get_eigenvalue(i, re, im);
        libMesh::out
                << std::setw(35) << std::fixed << std::setprecision(15)
                << re << std::endl;

        out_eig  << std::setw(25) << re  ;



//        // copy the solution for output
//        (*sys->solution) = *basis[i];
//
//        // We write the file in the ExodusII format.
//        std::set<std::string> nm;
//        nm.insert(sys->name());
//        writer_eig.write_timestep("modes.exo",
//                              eq_sys,
//                              i + 1, //  time step
//                              i);    //  time



    }

    if (nconv < n_eig) {
        int diff_eigs = n_eig - nconv ;
        for (int di = 0; di < diff_eigs; di++)
            out_eig << std::setw(25) << "N/A";
    }

    out_eig << std::endl;

    MAST::StiffenedPlateWeight weight(discipline);
    Real
            wt = 0.;
    // calculate weight
    (weight)(pt, 0., wt);

    libMesh::out << "the weight of the structure is:  "  << wt << std::endl;

    assembly.clear_discipline_and_system();
    elem_ops.clear_discipline_and_system();

//    modal_assembly.clear_base_solution();
//    modal_assembly.clear_discipline_and_system();
//    modal_elem_ops.clear_discipline_and_system();

    // END_TRANSLATE
    return 0;
}

