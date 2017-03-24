/*
 * MAST: Multidisciplinary-design Adaptation and Sensitivity Toolkit
 * Copyright (C) 2013-2017  Manav Bhatia
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


// MAST includes
#include "fluid/linearized_conservative_fluid_elem.h"
#include "fluid/primitive_fluid_solution.h"
#include "fluid/small_disturbance_primitive_fluid_solution.h"
#include "fluid/flight_condition.h"
#include "base/boundary_condition_base.h"
#include "base/system_initialization.h"
#include "aeroelasticity/frequency_function.h"
#include "elasticity/normal_rotation_function_base.h"



MAST::LinearizedConservativeFluidElem::
LinearizedConservativeFluidElem(MAST::SystemInitialization& sys,
                                const libMesh::Elem& elem,
                                const MAST::FlightCondition& f):
MAST::ConservativeFluidElementBase(sys, elem, f) {
    
    
}





MAST::LinearizedConservativeFluidElem::
~LinearizedConservativeFluidElem() {
    
    
}




bool
MAST::LinearizedConservativeFluidElem::
internal_residual (bool request_jacobian,
                   RealVectorX& f,
                   RealMatrixX& jac) {
    
    // first get the internal residual and Jacobian from the
    // conservative elems, and then add to it an extra dissipation
    // to control solution discontinuities emanating from the boundary.
    
    //const std::vector<Real>& JxW                  = _fe->get_JxW();
    //const std::vector<std::vector<Real> >& phi    = _fe->get_phi();
    const unsigned int
    dim    = _elem.dim(),
    n1     = dim+2,
    n2     = _fe->n_shape_functions()*n1;
    //nphi   = _fe->n_shape_functions();
    
    RealMatrixX
    f_jac_x         = RealMatrixX::Zero(   n2,    n2),
    fm_jac_xdot     = RealMatrixX::Zero(   n2,    n2);
    /*mat1_n1n1       = RealMatrixX::Zero(   n1,    n1),
     mat2_n1n1       = RealMatrixX::Zero(   n1,    n1),
     mat3_n1n2       = RealMatrixX::Zero(   n1,    n2),
     mat4_n2n2       = RealMatrixX::Zero(   n2,    n2),
     AiBi_adv        = RealMatrixX::Zero(   n1,    n2),
     A_sens          = RealMatrixX::Zero(   n1,    n2),
     LS              = RealMatrixX::Zero(   n1,    n2),
     LS_sens         = RealMatrixX::Zero(   n2,    n2);*/
    
    
    RealVectorX
    //vec1_n1  = RealVectorX::Zero(n1),
    //vec2_n1  = RealVectorX::Zero(n1),
    local_f    = RealVectorX::Zero(n2);
    //dc       = RealVectorX::Zero(dim);
    
    // df/dx. We always need the Jacobian, since it is used to calculate
    // the residual
    MAST::ConservativeFluidElementBase::internal_residual(true,
                                                          local_f,
                                                          f_jac_x);
    
    if (request_jacobian)
        jac      +=  f_jac_x;
    
    f        +=  f_jac_x * _delta_sol;
    
    return request_jacobian;
}




bool
MAST::LinearizedConservativeFluidElem::
side_external_residual (bool request_jacobian,
                        RealVectorX& f,
                        RealMatrixX& jac,
                        std::multimap<libMesh::boundary_id_type, MAST::BoundaryConditionBase*>& bc) {
    
    // the far-field and symmetry wall, which are assumed to be stationary will
    // only contribute to the real part of the complex Jacobian. Hence,
    // we will use the parent class's methods for these two. The
    // slip wall, which may be oscillating, is implemented for this element.
    
    const unsigned int
    dim     = _elem.dim(),
    n1      = dim+2,
    n2      = _fe->n_shape_functions()*n1;
    
    RealMatrixX
    f_jac_x = RealMatrixX::Zero(n2, n2);
    
    RealVectorX
    local_f = RealVectorX::Zero(n2);
    
    
    typedef std::multimap<libMesh::boundary_id_type, MAST::BoundaryConditionBase*> maptype;
    
    // iterate over the boundary ids given in the provided force map
    std::pair<maptype::const_iterator, maptype::const_iterator> it;
    
    const libMesh::BoundaryInfo& binfo = *_system.system().get_mesh().boundary_info;
    
    // for each boundary id, check if any of the sides on the element
    // has the associated boundary
    
    for (unsigned short int n=0; n<_elem.n_sides(); n++) {
        
        // if no boundary ids have been specified for the side, then
        // move to the next side.
        if (!binfo.n_boundary_ids(&_elem, n))
            continue;
        
        // check to see if any of the specified boundary ids has a boundary
        // condition associated with them
        std::vector<libMesh::boundary_id_type> bc_ids = binfo.boundary_ids(&_elem, n);
        std::vector<libMesh::boundary_id_type>::const_iterator bc_it = bc_ids.begin();
        
        for ( ; bc_it != bc_ids.end(); bc_it++) {
            
            // find the loads on this boundary and evaluate the f and jac
            it = bc.equal_range(*bc_it);
            
            for ( ; it.first != it.second; it.first++) {
                
                // apply all the types of loading
                switch (it.first->second->type()) {
                    case MAST::SYMMETRY_WALL: {
                        
                        f_jac_x.setZero();
                        local_f.setZero();
                        
                        // We always need the Jacobian, since it is used to
                        // calculate the residual
                        
                        MAST::ConservativeFluidElementBase::
                        symmetry_surface_residual(true,
                                                  local_f,
                                                  f_jac_x,
                                                  n,
                                                  *it.first->second);
                        
                        // multiply jacobian with the nondimensionalizing
                        // factor (V/b for flutter analysis)
                        if (request_jacobian)
                            jac  += f_jac_x;
                        f    += f_jac_x * _delta_sol;
                    }
                        break;
                        
                    case MAST::SLIP_WALL: {
                        
                        // this calculates the Jacobian and residual contribution
                        // including the nondimensionalizing factor.
                        this->slip_wall_surface_residual(request_jacobian,
                                                         f,
                                                         jac,
                                                         n,
                                                         *it.first->second);
                    }
                        break;
                        
                    case MAST::FAR_FIELD: {
                        
                        f_jac_x.setZero();
                        local_f.setZero();
                        
                        // We always need the Jacobian, since it is used to
                        // calculate the residual
                        
                        MAST::ConservativeFluidElementBase::
                        far_field_surface_residual(true,
                                                   local_f,
                                                   f_jac_x,
                                                   n,
                                                   *it.first->second);
                        
                        // multiply jacobian with the nondimensionalizing
                        // factor (V/b for flutter analysis)
                        if (request_jacobian)
                            jac  += f_jac_x;
                        
                        f    += f_jac_x * _delta_sol;
                    }
                        break;
                        
                    case MAST::DIRICHLET:
                        // nothing to be done here
                        break;
                        
                    default:
                        // not implemented yet
                        libmesh_error();
                        break;
                }
            }
        }
    }
    
    
    return request_jacobian;
}





bool
MAST::LinearizedConservativeFluidElem::
slip_wall_surface_residual(bool request_jacobian,
                           RealVectorX& f,
                           RealMatrixX& jac,
                           const unsigned int s,
                           MAST::BoundaryConditionBase& p) {
    
    // inviscid boundary condition without any diffusive component
    // conditions enforced are
    // vi ni = wi_dot (ni + dni) - ui dni   (moving slip wall with deformation)
    // tau_ij nj = 0   (because velocity gradient at wall = 0)
    // qi ni = 0       (since heat flux occurs only on no-slip wall and far-field bc)
    
    // prepare the side finite element
    libMesh::FEBase *fe_ptr    = nullptr;
    libMesh::QBase  *qrule_ptr = nullptr;
    _get_side_fe_and_qrule(_elem, s, &fe_ptr, &qrule_ptr, false);
    std::auto_ptr<libMesh::FEBase> fe(fe_ptr);
    std::auto_ptr<libMesh::QBase>  qrule(qrule_ptr);
    
    const std::vector<Real> &JxW                 = fe->get_JxW();
    const std::vector<libMesh::Point>& normals   = fe->get_normals();
    const std::vector<libMesh::Point>& qpoint    = fe->get_xyz();
    
    const unsigned int
    dim        = _elem.dim(),
    n1         = dim+2,
    n2         = _fe->n_shape_functions()*n1;
    
    RealVectorX
    vec1_n1    = RealVectorX::Zero(n1),
    uvec       = RealVectorX::Zero(3),
    dwdot_i    = RealVectorX::Zero(3),
    ni         = RealVectorX::Zero(3),
    dni        = RealVectorX::Zero(3),
    Dw_i       = RealVectorX::Zero(3),
    Dni        = RealVectorX::Zero(3),
    Duvec      = RealVectorX::Zero(3),
    vec2_n1    = RealVectorX::Zero(n1),
    vec2_n2    = RealVectorX::Zero(n2),
    tmp        = RealVectorX::Zero(6),
    flux       = RealVectorX::Zero(n1);
    
    RealMatrixX
    mat1_n1n1  = RealMatrixX::Zero( n1, n1),
    mat2_n1n2  = RealMatrixX::Zero( n1, n2),
    mat3_n2n2  = RealMatrixX::Zero( n2, n2);
    
    libMesh::Point pt;
    MAST::FEMOperatorMatrix Bmat;
    
    // create objects to calculate the primitive solution, flux, and Jacobian
    MAST::PrimitiveSolution                            primitive_sol;
    MAST::SmallPerturbationPrimitiveSolution<Real>  sd_primitive_sol;
    
    Real
    Dvi_ni              = 0.,
    ui_ni_steady        = 0.;
    
    
    // get the surface motion object from the boundary condition object
    MAST::FieldFunction<RealVectorX>
    *displ   = nullptr;
    MAST::NormalRotationFunctionBase<RealVectorX>
    *n_rot = nullptr;
    
    if (p.contains("displacement"))
        displ = &p.get<MAST::FieldFunction<RealVectorX> >("displacement");
    if (p.contains("normal_rotation")) {
        
        MAST::FieldFunction<RealVectorX>&
        tmp = p.get<MAST::FieldFunction<RealVectorX> >("normal_rotation");
        n_rot = dynamic_cast<MAST::NormalRotationFunctionBase<RealVectorX>*>(&tmp);
    }
    
    
    for (unsigned int qp=0; qp<JxW.size(); qp++) {
        
        // initialize the Bmat operator for this term
        _initialize_fem_interpolation_operator(qp, dim, *fe, Bmat);
        Bmat.right_multiply(vec1_n1,         _sol);  // conservative sol
        Bmat.right_multiply(vec2_n1,   _delta_sol);  // perturbation sol
        
        // initialize the primitive solution
        primitive_sol.zero();
        primitive_sol.init(dim,
                           vec1_n1,
                           flight_condition->gas_property.cp,
                           flight_condition->gas_property.cv,
                           if_viscous());
        
        // initialize the small-disturbance primitive sol
        sd_primitive_sol.zero();
        sd_primitive_sol.init(primitive_sol, vec2_n1);
        
        ////////////////////////////////////////////////////////////
        //   Calculation of the surface velocity term. For a
        //   steady-state system,
        //        vi (ni + dni)  =  wdot_i (ni + dni)
        //  or    vi ni = wdot_i (ni + dni) - vi dni
        //
        //    The first order perturbation, D(.), of this is
        //        (vi + Dvi) (ni + dni + Dni) =
        //          (wdot_i + Dwdot_i) (ni + dni + Dni)
        //  or    Dvi ni = Dwdot_i (ni + dni + Dni) +
        //                  wdot_i (ni + dni + Dni) -
        //                  vi (ni + dni + Dni) -
        //                  Dvi (dni + Dni)
        //     Neglecting HOT, this becomes
        //        Dvi ni = Dwdot_i (ni + dni) +
        //                  wdot_i (ni + dni + Dni) -
        //                  vi (ni + dni + Dni) -
        //                  Dvi dni
        //
        ////////////////////////////////////////////////////////////
        
        // copy the surface normal
        for (unsigned int i_dim=0; i_dim<dim; i_dim++)
            ni(i_dim) = normals[qp](i_dim);
        
        // now check if the surface deformation is defined and
        // needs to be applied through transpiration boundary
        // condition
        primitive_sol.get_uvec(uvec);
        sd_primitive_sol.get_duvec(Duvec);
        
        flux.setZero();
        
        //////////////////////////////////////////////////////////////
        // contribution from the base-flow boundary condition
        //////////////////////////////////////////////////////////////
        if (displ) {
            
            (*displ)(qpoint[qp], _time, tmp);
            dwdot_i = tmp.topRows(3);
            (*n_rot)(qpoint[qp], normals[qp], _time, dni);
            ui_ni_steady  =  dwdot_i.dot(ni+dni) - uvec.dot(dni);
            
            flux         += ui_ni_steady * vec2_n1;               // vi_ni  dcons_flux
            flux(n1-1)   += ui_ni_steady * sd_primitive_sol.dp;   // vi_ni {0,0,0,0,Dp}
            
            //////////////////////////////////////////////////////////////
            // contribution from the small-disturbance boundary condition
            //////////////////////////////////////////////////////////////
            displ->perturbation(qpoint[qp], _time, tmp);
            Dw_i = tmp.topRows(3);
            n_rot->perturbation(qpoint[qp], normals[qp], _time, Dni);
        }
        
        
        Dvi_ni   = (Dw_i.dot(ni+dni) +
                    dwdot_i.dot(ni+dni+Dni) -
                    uvec.dot(ni+dni+Dni) -
                    Duvec.dot(dni));
        
        
        flux             += Dvi_ni * vec1_n1;              // Dvi_ni cons_flux
        flux(n1-1)       += Dvi_ni * primitive_sol.p;      // Dvi_ni {0,0,0,0,p}
        
        // ni Dp  (only for the momentun eq)
        flux.segment(1, dim) += (sd_primitive_sol.dp * ni.segment(0,dim));
        
        Bmat.vector_mult_transpose(vec2_n2, flux);
        f += JxW[qp] * vec2_n2;
        
        if ( request_jacobian ) {
            
            // the Jacobian contribution comes only from the dp term in the
            // residual
            this->calculate_advection_flux_jacobian_for_moving_solid_wall_boundary
            (primitive_sol,
             ui_ni_steady,
             normals[qp],
             dni,
             mat1_n1n1);
            
            Bmat.left_multiply(mat2_n1n2, mat1_n1n1);
            Bmat.right_multiply_transpose(mat3_n2n2, mat2_n1n2);
            jac += JxW[qp] * mat3_n2n2;
        }
    }
    
    return request_jacobian;
}



