/*
 * MAST: Multidisciplinary-design Adaptation and Sensitivity Toolkit
 * Copyright (C) 2013-2015  Manav Bhatia
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


// BOOST includes
#include <boost/test/unit_test.hpp>


// MAST includes
#include "examples/structural/beam_bending_thermal_stress_with_offset/beam_bending_thermal_stress.h"
#include "tests/base/test_comparisons.h"
#include "elasticity/structural_system_initialization.h"
#include "elasticity/structural_discipline.h"
#include "elasticity/structural_element_base.h"
#include "elasticity/stress_output_base.h"
#include "property_cards/solid_1d_section_element_property_card.h"
#include "base/parameter.h"
#include "base/constant_field_function.h"
#include "property_cards/isotropic_material_property_card.h"



// libMesh includes
#include "libmesh/numeric_vector.h"



//BOOST_FIXTURE_TEST_SUITE  (Structural1DBeamBending,
//                           MAST::BeamBendingThermalStress)
//
//BOOST_AUTO_TEST_CASE   (BeamBendingWithThermalStressSolution) {
//    
//    this->solve();
//    
//    // check the solution
//    // iterate over each node, and compare the nodal solution with the
//    // expected anlaytical value
//    unsigned int
//    dof_num = 0;
//    libMesh::MeshBase::const_node_iterator
//    it     =  _mesh->local_nodes_begin(),
//    end    =  _mesh->local_nodes_end();
//    
//    Real
//    temp        = (*_temp)(),
//    th_y        = (*_thy)(),
////    th_z        = (*_thz)(),
//    x           = 0.,
//    xi          = 0.,
//    eta         = 0.,
//    Eval        = (*_E)(),
//    alpha       = (*_alpha)(),
//    analytical  = 0.,
//    numerical   = 0.;
//    
//    // analytical solution to the simply supported problem is
//    // w(x)    = -alpha*T*A*th_y/2/Izz*(x^2/2 - L*x/2);
//    // dwdx(x) = -alpha*T*A*th_y/2/Izz*(x - L/2);
//    
//    BOOST_TEST_MESSAGE("  ** v-displacement and theta-z rotation **");
//    for ( ; it!=end; it++) {
//        const libMesh::Node* node = *it;
//        x            = (*node)(0);
//        
//        // v-displacement
//        analytical   = -alpha*temp/th_y*3./2.*(pow(x,2)/2. - _length*x/2.);
//        
//        dof_num      = node->dof_number(_sys->number(),
//                                        _structural_sys->vars()[1], // v-displ.
//                                        0);
//        numerical    =   _sys->solution->el(dof_num);
//        BOOST_CHECK(MAST::compare_value(analytical, numerical, tol));
//        
//        // theta-z rotation
//        analytical   = -alpha*temp/th_y*3./2.*(x - _length/2.);
//        
//        dof_num      = node->dof_number(_sys->number(),
//                                        _structural_sys->vars()[5], // tz-rotation
//                                        0);
//        numerical    =   _sys->solution->el(dof_num);
//        BOOST_CHECK(MAST::compare_value(analytical, numerical, tol));
//        
//    }
//    
//    
//    // make sure that each stress object has a single stored value
//    for (unsigned int i=0; i<_outputs.size(); i++) {
//        BOOST_CHECK(_outputs[i]->n_elem_in_storage() == 1);
//    }
//    
//    // now check the stress value in each element, which should be the same as
//    // the pressure value specified for the problem
//    BOOST_TEST_MESSAGE("  ** Stress **");
//    for (unsigned int i=0; i<_outputs.size(); i++) {
//        
//        // get the element and the nodes to evaluate the stress
//        const libMesh::Elem& e  = **(_outputs[i]->get_elem_subset().begin());
//        
//        const std::vector<MAST::StressStrainOutputBase::Data*>&
//        data = _outputs[i]->get_stress_strain_data_for_elem(&e);
//        
//        // find the location of quadrature point
//        for (unsigned int j=0; j<data.size(); j++) {
//            
//            // logitudinal strain for this location
//            numerical = data[j]->stress()(0);
//            
//            xi   = data[j]->point_location_in_element_coordinate()(0);
//            eta  = data[j]->point_location_in_element_coordinate()(1);
//            
//            // assuming linear Lagrange interpolation for elements
//            x =  e.point(0)(0) * (1.-xi)/2. +  e.point(1)(0) * (1.+xi)/2.;
//            // stress is a combination of the bending and compressive stress.
//            analytical   = Eval*alpha*temp*(0.75*(eta+1.) - 1.);
//            
//            
//            BOOST_CHECK(MAST::compare_value(analytical, numerical, tol));
//        }
//    }
//}
//
//
//BOOST_AUTO_TEST_CASE   (BeamBendingWithThermalStressSensitivity) {
//    
//    // verify the sensitivity solution of this system
//    RealVectorX
//    sol,
//    dsol;
//    
//    const libMesh::NumericVector<Real>& sol_vec = this->solve();
//    
//    // make sure that each stress object has a single stored value
//    for (unsigned int i=0; i<_outputs.size(); i++)
//        BOOST_CHECK((_outputs[i]->n_elem_in_storage() == 1));
//    
//    const unsigned int
//    n_dofs     = sol_vec.size(),
//    n_elems    = _mesh->n_elem();
//    
//    unsigned int
//    dof_num   = 0;
//    
//    const Real
//    p_val      = 2.;
//    
//    // store the stress values for sensitivity calculations
//    // make sure that the current setup has specified one stress output
//    // per element
//    libmesh_assert(_outputs.size() == n_elems);
//    RealVectorX
//    stress0       =  RealVectorX::Zero(n_elems),
//    dstressdp     =  RealVectorX::Zero(n_elems),
//    dstressdp_fd  =  RealVectorX::Zero(n_elems);
//    
//    for (unsigned int i=0; i<n_elems; i++) {
//        // the call to all elements should actually include a single element only
//        // the p-norm used is for p=2.
//        stress0(i) = _outputs[i]->von_Mises_p_norm_functional_for_all_elems(p_val);
//    }
//    
//    
//    sol      =   RealVectorX::Zero(n_dofs);
//    dsol     =   RealVectorX::Zero(n_dofs);
//    
//    // copy the solution for later use
//    for (unsigned int i=0; i<n_dofs; i++)  sol(i)  = sol_vec(i);
//    
//    // now clear the stress data structures
//    this->clear_stresss();
//    
//    // now iterate over all the parameters and calculate the analytical
//    // sensitivity and compare with the numerical sensitivity
//    
//
//
//    Real
//    temp        = (*_temp)(),
//    th_y        = (*_thy)(),
////    th_z        = (*_thz)(),
////    A           = th_y * th_z,
////    dAdth_y     = th_z,
////    dAdth_z     = th_y,
////    Izz         = th_z*pow(th_y,3)/12.+th_z*pow(th_y,3)/4.,
////    dIzzdth_y   = 3.*th_z*pow(th_y,2)/12.+3.*th_z*pow(th_y,2)/4.,
////    dIzzdth_z   = pow(th_y,3)/12.+pow(th_y,3)/4.,
//    x           = 0.,
//    xi          = 0.,
//    eta         = 0.,
//    analytical  = 0.,
//    numerical   = 0.,
////    Eval        = (*_E)(),
//    alpha       = (*_alpha)(),
//    p0          = 0.,
//    dp          = 0.;
//    
//    /////////////////////////////////////////////////////////
//    // now evaluate the direct sensitivity
//    /////////////////////////////////////////////////////////
//    
//    for (unsigned int i=0; i<_params_for_sensitivity.size(); i++ ) {
//        
//        MAST::Parameter& f = *this->_params_for_sensitivity[i];
//        
//        dsol         =   RealVectorX::Zero(n_dofs);
//        dstressdp    =   RealVectorX::Zero(n_elems);
//        dstressdp_fd =   RealVectorX::Zero(n_elems);
//        
//        // calculate the analytical sensitivity
//        // analysis is required at the baseline before sensitivity solution
//        // and the solution has changed after the previous perturbed solution
//        this->solve();
//        const libMesh::NumericVector<Real>& dsol_vec = this->sensitivity_solve(f);
//        
//        ////////////////////////////////////////////////////////
//        //   compare the displacement sensitivity
//        ////////////////////////////////////////////////////////
//        BOOST_TEST_MESSAGE("  ** dX/dp (total) wrt : " << f.name() << " **");
//        libMesh::MeshBase::const_node_iterator
//        it     =  _mesh->local_nodes_begin(),
//        end    =  _mesh->local_nodes_end();
//        for ( ; it!=end; it++) {
//            
//            const libMesh::Node* node = *it;
//            x            = (*node)(0);
//            
//            // v-displacement
//            analytical   = 0.;
//            if (f.name() == "nu")
//                analytical = 0.;
//            else if (f.name() == "E")
//                analytical = 0.;
//            else if (f.name() == "thy")
//                analytical   =
//                alpha*temp/pow(th_y,2)*3./2.*(pow(x,2)/2. - _length*x/2.);
//            else if (f.name() == "thz")
//                analytical   = 0.;
//            else
//                libmesh_error(); // should not get here
//
//            
//            dof_num      = node->dof_number(_sys->number(),
//                                            _structural_sys->vars()[1], // v-displ.
//                                            0);
//            numerical    =   dsol_vec.el(dof_num);
//            BOOST_CHECK(MAST::compare_value(analytical, numerical, tol));
//            
//            
//            
//            // theta-z rotation
//            analytical   = 0.;
//            if (f.name() == "nu")
//                analytical = 0.;
//            else if (f.name() == "E")
//                analytical = 0.;
//            else if (f.name() == "thy")
//                analytical   = alpha*temp/pow(th_y,2)*3./2.*(x - _length/2.);
//            else if (f.name() == "thz")
//                analytical   = 0.;
//            else
//                libmesh_error(); // should not get here
//            
//            dof_num      = node->dof_number(_sys->number(),
//                                            _structural_sys->vars()[5], // tz-rotation
//                                            0);
//            numerical    =   dsol_vec.el(dof_num);
//            BOOST_CHECK(MAST::compare_value(analytical, numerical, tol));
//        }
//        
//        
//        
//        ////////////////////////////////////////////////////////
//        //   compare the direct stress sensitivity with
//        //   analytical expressions
//        ////////////////////////////////////////////////////////
//        BOOST_TEST_MESSAGE("  ** dstress/dp (total) wrt : " << f.name() << " **");
//        for (unsigned int ii=0; ii<_outputs.size(); ii++) {
//            
//            // get the element and the nodes to evaluate the stress
//            const libMesh::Elem& e  = **(_outputs[ii]->get_elem_subset().begin());
//            
//            const std::vector<MAST::StressStrainOutputBase::Data*>&
//            data = _outputs[ii]->get_stress_strain_data_for_elem(&e);
//            
//            // find the location of quadrature point
//            for (unsigned int j=0; j<data.size(); j++) {
//                
//                // logitudinal strain for this location
//                numerical = data[j]->get_stress_sensitivity(&f)(0);
//                
//                xi         = data[j]->point_location_in_element_coordinate()(0);
//                eta        = data[j]->point_location_in_element_coordinate()(1);
//                
//                // assuming linear Lagrange interpolation for elements
//                x          =  e.point(0)(0) * (1.-xi)/2. +  e.point(1)(0) * (1.+xi)/2.;
//                
//                // this gives the stress sensitivity
//                analytical = 0.;
//                if (f.name() == "nu")
//                    analytical = 0.;
//                else if (f.name() == "E")
//                    analytical   = alpha*temp*(0.75*(eta+1.) - 1.);
//                else if (f.name() == "thy")
//                    analytical   = 0.;
//                else if (f.name() == "thz")
//                    analytical   = 0.;
//                else
//                    libmesh_error(); // should not get here
//                
//                BOOST_CHECK(MAST::compare_value(analytical, numerical, tol));
//            }
//        }
//        
//        
//        // make sure that each stress object has a single stored value
//        for (unsigned int i=0; i<_outputs.size(); i++)
//            BOOST_CHECK((_outputs[i]->n_elem_in_storage() == 1));
//        
//        // copy the sensitivity solution
//        for (unsigned int j=0; j<n_dofs; j++)  dsol(j)  = dsol_vec(j);
//        
//        // copy the analytical sensitivity of stress values
//        for (unsigned int j=0; j<n_elems; j++)
//            dstressdp(j)  =
//            _outputs[j]->von_Mises_p_norm_functional_sensitivity_for_all_elems
//            (p_val, &f);
//        
//        // now clear the stress data structures
//        this->clear_stresss();
//        
//        // now calculate the finite difference sensitivity
//        
//        // identify the perturbation in the parameter
//        p0           = f();
//        (p0 > 0)?  dp=delta*p0 : dp=delta;
//        f()         += dp;
//        
//        // solve at the perturbed parameter value
//        this->solve();
//        
//        // make sure that each stress object has a single stored value
//        for (unsigned int i=0; i<_outputs.size(); i++)
//            BOOST_CHECK((_outputs[i]->n_elem_in_storage() == 1));
//        
//        
//        ////////////////////////////////////////////////////////
//        //   compare the von Mises stress sensitivity
//        ////////////////////////////////////////////////////////
//        // copy the perturbed stress values
//        for (unsigned int j=0; j<n_elems; j++)
//            dstressdp_fd(j)  =
//            _outputs[j]->von_Mises_p_norm_functional_for_all_elems(p_val);
//        
//        // calculate the finite difference sensitivity for stress
//        dstressdp_fd  -= stress0;
//        dstressdp_fd  /= dp;
//        
//        // reset the parameter value
//        f()        = p0;
//        
//        // now compare the stress sensitivity
//        BOOST_TEST_MESSAGE("  ** dvm-stress/dp (total) wrt : " << f.name() << " **");
//        BOOST_CHECK(MAST::compare_vector(    dstressdp_fd,  dstressdp,  tol));
//        
//        // now clear the stress data structures
//        this->clear_stresss();
//        
//    }
//    
//    
//    /////////////////////////////////////////////////////////
//    // now evaluate the adjoint sensitivity
//    /////////////////////////////////////////////////////////
//    this->clear_stresss();
//    
//}
//
//
//BOOST_AUTO_TEST_SUITE_END()
