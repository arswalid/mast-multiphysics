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

// C++ includes
#include <iomanip>

// MAST includes
#include "examples/base/input_wrapper.h"
#include "level_set/level_set_discipline.h"
#include "level_set/level_set_system_initialization.h"
#include "level_set/level_set_eigenproblem_assembly.h"
#include "level_set/level_set_nonlinear_implicit_assembly.h"
#include "level_set/level_set_volume_output.h"
#include "level_set/level_set_perimeter_output.h"
#include "level_set/level_set_boundary_velocity.h"
#include "level_set/indicator_function_constrain_dofs.h"
#include "level_set/level_set_constrain_dofs.h"
#include "level_set/level_set_intersection.h"
#include "level_set/filter_base.h"
#include "level_set/level_set_parameter.h"
#include "level_set/sub_elem_mesh_refinement.h"
#include "elasticity/structural_nonlinear_assembly.h"
#include "elasticity/structural_modal_eigenproblem_assembly.h"
#include "elasticity/ks_stress_output.h"
#include "elasticity/smooth_ramp_stress_output.h"
#include "elasticity/level_set_stress_assembly.h"
#include "elasticity/compliance_output.h"
#include "elasticity/structural_system_initialization.h"
#include "elasticity/structural_near_null_vector_space.h"
#include "heat_conduction/heat_conduction_system_initialization.h"
#include "heat_conduction/heat_conduction_nonlinear_assembly.h"
#include "base/constant_field_function.h"
#include "base/nonlinear_system.h"
#include "base/transient_assembly.h"
#include "base/boundary_condition_base.h"
#include "boundary_condition/dirichlet_boundary_condition.h"
#include "solver/slepc_eigen_solver.h"
#include "property_cards/isotropic_material_property_card.h"
#include "property_cards/solid_2d_section_element_property_card.h"
#include "optimization/gcmma_optimization_interface.h"
#include "optimization/npsol_optimization_interface.h"
#include "optimization/function_evaluation.h"
#include "examples/structural/base/bracket_2d_model.h"
#include "examples/structural/base/inplane_2d_model.h"
#include "examples/structural/base/truss_2d_model.h"
#include "examples/structural/base/eyebar_2d_model.h"


// libMesh includes
#include "libmesh/fe_type.h"
#include "libmesh/serial_mesh.h"
#include "libmesh/equation_systems.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/dof_map.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/petsc_nonlinear_solver.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/error_vector.h"
#include "libmesh/parallel.h"


void
_optim_obj(int*    mode,
           int*    n,
           double* x,
           double* f,
           double* g,
           int*    nstate);
void
_optim_con(int*    mode,
           int*    ncnln,
           int*    n,
           int*    ldJ,
           int*    needc,
           double* x,
           double* c,
           double* cJac,
           int*    nstate);

//
// BEGIN_TRANSLATE Level-set topology optimization
//
//   \tableofcontents
//
//  This example computes the optimal topology of a structure subject to
//  specified boundary conditions (Dirichlet and Neumann). A level-set function
//  is used to implicitly define the geometry inside a mesh using the
//  immersed boundary approach.
//
//  Level Set Mesh Function
class PhiMeshFunction:
public MAST::FieldFunction<Real> {
public:
    PhiMeshFunction():
    MAST::FieldFunction<Real>("phi"), _phi(nullptr) { }
    virtual ~PhiMeshFunction(){ if (_phi) delete _phi;}
    
    void init(MAST::SystemInitialization& sys, const libMesh::NumericVector<Real>& sol) {
        if (!_phi) _phi = new MAST::MeshFieldFunction(sys, "phi", libMesh::SERIAL);
        else _phi->clear();
        _phi->init(sol, true);
    }

    void init_sens(const MAST::FunctionBase& f,
                   const libMesh::NumericVector<Real>& dsol) {
        
        libmesh_assert(_phi);
        _phi->init_sens(f, dsol, true);
    }

    MAST::MeshFieldFunction& get_mesh_function() {return *_phi;}
    
    virtual void operator() (const libMesh::Point& p, const Real t, Real& v) const {
        libmesh_assert(_phi);
        RealVectorX v1;
        (*_phi)(p, t, v1);
        v = v1(0);
    }
    
protected:
    MAST::MeshFieldFunction *_phi;
};


class ElementParameterDependence:
public MAST::AssemblyBase::ElemParameterDependence {
public:
    ElementParameterDependence(const MAST::FilterBase& filter):
    MAST::AssemblyBase::ElemParameterDependence(true), _filter(filter) {}
    virtual ~ElementParameterDependence() {}
    virtual bool if_elem_depends_on_parameter(const libMesh::Elem& e,
                                              const MAST::FunctionBase& p) const {
        const MAST::LevelSetParameter
        &p_ls = dynamic_cast<const MAST::LevelSetParameter&>(p);
        
        return _filter.if_elem_in_domain_of_influence(e, *p_ls.level_set_node());
    }
    
private:
    const MAST::FilterBase& _filter;
};


template <typename T>
class TopologyOptimizationLevelSet:
public MAST::FunctionEvaluation {
    
public:
    
    bool                                      _initialized;
    MAST::Examples::GetPotWrapper&            _input;

    std::string                               _problem;
    Real                                      _volume;
    Real                                      _obj_scaling;
    Real                                      _stress_penalty;
    Real                                      _perimeter_penalty;
    Real                                      _stress_lim;
    Real                                      _p_val, _vm_rho;
    Real                                      _ref_eig_val;
    unsigned int                              _n_eig_vals;
    
    libMesh::UnstructuredMesh*                _mesh;
    libMesh::UnstructuredMesh*                _level_set_mesh;
    
    MAST::SubElemMeshRefinement*              _mesh_refinement;
    
    libMesh::EquationSystems*                 _eq_sys;
    libMesh::EquationSystems*                 _level_set_eq_sys;
    
    MAST::NonlinearSystem*                    _sys;
    MAST::NonlinearSystem*                    _level_set_sys;
    MAST::NonlinearSystem*                    _level_set_sys_on_str_mesh;
    MAST::NonlinearSystem*                    _indicator_sys;
    
    MAST::StructuralSystemInitialization*     _sys_init;
    MAST::LevelSetSystemInitialization*       _level_set_sys_init_on_str_mesh;
    MAST::LevelSetSystemInitialization*       _level_set_sys_init;
    MAST::HeatConductionSystemInitialization* _indicator_sys_init;
     
    MAST::StructuralNearNullVectorSpace*      _nsp;

    MAST::PhysicsDisciplineBase*              _discipline;
    MAST::PhysicsDisciplineBase*              _indicator_discipline;
    MAST::LevelSetDiscipline*                 _level_set_discipline;
    
    MAST::FilterBase*                         _filter;
    
    MAST::MaterialPropertyCardBase            *_m_card;
    MAST::ElementPropertyCardBase             *_p_card;
    
    PhiMeshFunction*                          _level_set_function;
    MAST::LevelSetBoundaryVelocity*           _level_set_vel;
    libMesh::ExodusII_IO*                     _output;
    
    libMesh::FEType                           _fetype;
    libMesh::FEType                           _level_set_fetype;
    
    MAST::BoundaryConditionBase*              _shifted_boundary_load;
    std::vector<MAST::Parameter*>             _params_for_sensitivity;
    std::map<std::string, MAST::Parameter*>   _parameters;
    std::set<MAST::FunctionBase*>             _field_functions;
    std::set<MAST::BoundaryConditionBase*>    _boundary_conditions;
    std::set<unsigned int>                    _dv_dof_ids;
    std::set<unsigned int>                    _dirichlet_bc_ids;
    
    std::vector<std::pair<unsigned int, MAST::Parameter*>>  _dv_params;

    
    //
    //  \section ex_5_system_discipline  System and Discipline
    //
    void _init_system_and_discipline() {
        
        //
        // make sure that the mesh has been initialized
        //
        libmesh_assert(_mesh);
        
        //
        // create the equation system
        //
        _eq_sys    = new  libMesh::EquationSystems(*_mesh);
        
        //
        // create the libmesh system and set the preferences for structural
        // eigenvalue problems
        //
        _sys       = &(_eq_sys->add_system<MAST::NonlinearSystem>("structural"));
        _sys->set_eigenproblem_type(libMesh::GHEP);

        _mesh_refinement = new MAST::SubElemMeshRefinement(*_mesh, *_sys);
        _sys->attach_constraint_object(*_mesh_refinement);
        
        //
        // initialize the system to the right set of variables
        //
        _sys_init       = new MAST::StructuralSystemInitialization(*_sys,
                                                                   _sys->name(),
                                                                   _fetype);
        _discipline     = new MAST::PhysicsDisciplineBase(*_eq_sys);
        
        //
        // Initialize the system for level set function.
        // A level set function is defined on a coarser mesh than the structural
        // mesh.
        // A level set function is assumed to be a first-order Lagrange finite element
        //
        _level_set_fetype      = libMesh::FEType(libMesh::FIRST, libMesh::LAGRANGE);
        _level_set_eq_sys      = new libMesh::EquationSystems(*_level_set_mesh);
        _level_set_sys         = &(_level_set_eq_sys->add_system<MAST::NonlinearSystem>("level_set"));
        _level_set_sys->extra_quadrature_order = 2;
        _level_set_sys_init    = new MAST::LevelSetSystemInitialization(*_level_set_sys,
                                                                        _level_set_sys->name(),
                                                                        _level_set_fetype);
        _level_set_discipline  = new MAST::LevelSetDiscipline(*_eq_sys);
        
        //
        // A system with level set function is defined on the strucutral mesh
        // for the purpose of plotting.
        //
        _level_set_sys_on_str_mesh      = &(_eq_sys->add_system<MAST::NonlinearSystem>("level_set"));
        _level_set_sys_init_on_str_mesh = new MAST::LevelSetSystemInitialization(*_level_set_sys_on_str_mesh,
                                                                                 _level_set_sys->name(),
                                                                                 _level_set_fetype);
        
        //
        //  an indicator function is used to locate unconnected free-floating
        // domains of material. The indicator function solves a heat condution
        // problem. Regions with uniformly zero temperature are marked as
        // unconnected domains.
        //
        _indicator_sys                  = &(_eq_sys->add_system<MAST::NonlinearSystem>("indicator"));
        _indicator_sys_init             = new MAST::HeatConductionSystemInitialization(*_indicator_sys,
                                                                                       _indicator_sys->name(),
                                                                                       _fetype);
        _indicator_discipline           = new MAST::PhysicsDisciplineBase(*_eq_sys);
    }

    
    void _init_eq_sys() {
        
        _eq_sys->init();
        _sys->eigen_solver->set_position_of_spectrum(libMesh::LARGEST_MAGNITUDE);
        _sys->set_exchange_A_and_B(true);
        
        _level_set_eq_sys->init();
    }
    

    //
    //   variables added to the mesh
    //
    void _init_fetype() {
        
        // FEType to initialize the system. Get the order and type of element.
        std::string
        order_str   = _input("fe_order", "order of finite element shape basis functions",    "first"),
        family_str  = _input("fe_family",      "family of finite element shape functions", "lagrange");
        
        libMesh::Order
        o  = libMesh::Utility::string_to_enum<libMesh::Order>(order_str);
        libMesh::FEFamily
        fe = libMesh::Utility::string_to_enum<libMesh::FEFamily>(family_str);
        _fetype = libMesh::FEType(o, fe);
    }
    
    
    //
    //   \section ex_5_properties Properties
    //
    //
    //
    //   \subsection ex_5_material_properties Material Properties
    //

    void _init_material() {
        
        Real
        Eval      = _input("E", "modulus of elasticity", 72.e9),
        rhoval    = _input("rho", "material density", 2700.),
        nu_val    = _input("nu", "Poisson's ratio",  0.33),
        kval      = _input("k", "thermal conductivity",  1.e-2),
        cpval     = _input("cp", "thermal capacitance",  864.);
        
        
        MAST::Parameter
        *E         = new MAST::Parameter("E",          Eval),
        *E_v       = new MAST::Parameter("E_v",          0.),
        *rho       = new MAST::Parameter("rho",      rhoval),
        *nu        = new MAST::Parameter("nu",       nu_val),
        *k         = new MAST::Parameter("k",          kval),
        *cp        = new MAST::Parameter("cp",        cpval);
        
        MAST::ConstantFieldFunction
        *E_f     = new MAST::ConstantFieldFunction(    "E",      *E),
        *E_v_f   = new MAST::ConstantFieldFunction(    "E",    *E_v),
        *rho_f   = new MAST::ConstantFieldFunction(  "rho",    *rho),
        *nu_f    = new MAST::ConstantFieldFunction(   "nu",     *nu),
        *k_f     = new MAST::ConstantFieldFunction( "k_th",      *k),
        *cp_f    = new MAST::ConstantFieldFunction(   "cp",     *cp);
        
        _parameters[    E->name()]     = E;
        _parameters[  E_v->name()]     = E_v;
        _parameters[  rho->name()]     = rho;
        _parameters[   nu->name()]     = nu;
        _parameters[    k->name()]     = k;
        _parameters[   cp->name()]     = cp;
        _field_functions.insert(E_f);
        _field_functions.insert(E_v_f);
        _field_functions.insert(rho_f);
        _field_functions.insert(nu_f);
        _field_functions.insert(k_f);
        _field_functions.insert(cp_f);

        _m_card  = new MAST::IsotropicMaterialPropertyCard;
        _m_card->add(*E_f);
        _m_card->add(*rho_f);
        _m_card->add(*nu_f);
        _m_card->add(*k_f);
        _m_card->add(*cp_f);
    }

    
    //
    //   \subsection ex_5_section_properties Section Properties
    //

    void _init_section_property(){
        
        
        
        Real
        kappa_val = _input("kappa", "shear correction factor",  5./6.),
        th_v      =  _input("th", "thickness of 2D element",  0.001);
        
        MAST::Parameter
        *th       = new MAST::Parameter("th", th_v),
        *kappa    = new MAST::Parameter("kappa", kappa_val),
        *zero     = new MAST::Parameter("zero", 0.);
        
        MAST::ConstantFieldFunction
        *th_f     = new MAST::ConstantFieldFunction("h",       *th),
        *kappa_f  = new MAST::ConstantFieldFunction("kappa",  *kappa),
        *hoff_f   = new MAST::ConstantFieldFunction("off",   *zero);
        
        
        _parameters[th->name()]    = th;
        _parameters[kappa->name()] = kappa;
        _parameters[zero->name()]  = zero;
        _field_functions.insert(th_f);
        _field_functions.insert(kappa_f);
        _field_functions.insert(hoff_f);
        
         typename T::SectionPropertyCardType
         *p_card   = new typename T::SectionPropertyCardType;

        _p_card   = p_card;

        // set nonlinear strain if requested
        bool
        nonlinear = _input("if_nonlinear", "flag to turn on/off nonlinear strain", false);
        if (nonlinear) _p_card->set_strain(MAST::NONLINEAR_STRAIN);

        // property card for void
        p_card->add(*th_f);
        p_card->add(*kappa_f);
        p_card->add(*hoff_f);
        p_card->set_material(*_m_card);
        
        _discipline->set_property_for_subdomain(0, *p_card);


        _indicator_discipline->set_property_for_subdomain(0, *p_card);
    }
        
    //
    //   \subsection ex_5_design_variable_init   Design Variables
    //
    //   initializes the design variable vector, called by the
    //   optimization interface.
    //
    void init_dvar(std::vector<Real>& x,
                   std::vector<Real>& xmin,
                   std::vector<Real>& xmax) {
        
        x.resize(_n_vars);
        xmin.resize(_n_vars);
        xmax.resize(_n_vars);
        
        std::fill(xmin.begin(), xmin.end(),   -1.e0);
        std::fill(xmax.begin(), xmax.end(),    1.e0);

        //
        // now, check if the user asked to initialize dvs from a previous file
        //
        std::string
        nm    =  _input("restart_optimization_file", "filename with optimization history for restart", "");
        
        if (nm.length()) {
            
            unsigned int
            iter = _input("restart_optimization_iter", "restart iteration number from file", 0);
            this->initialize_dv_from_output_file(nm, iter, x);
        }
        else {
            
            for (unsigned int i=0; i<_n_vars; i++)
                x[i] = (*_dv_params[i].second)();
        }
    }

    //
    //  \section  ex_5_analysis Function Evaluation and Sensitivity
    //
    //
    //   \subsection ex_5_element_error_metric Element Error Metric
    //
    void
    _compute_element_errors(libMesh::ErrorVector& error) {
        
        MAST::LevelSetIntersection intersection;
        
        libMesh::MeshBase::const_element_iterator
        it  = _mesh->active_elements_begin(),
        end = _mesh->active_elements_end();
        
        for ( ; it != end; it++) {
            
            const libMesh::Elem* elem = *it;
            intersection.init( *_level_set_function, *elem, _sys->time,
                              _mesh->max_elem_id(),
                              _mesh->max_node_id());
            if (intersection.if_intersection_through_elem())
                error[elem->id()] = 1.-intersection.get_positive_phi_volume_fraction();
            intersection.clear();
        }
    }
    
    
    
    class ElemFlag: public libMesh::MeshRefinement::ElementFlagging {
    public:
        ElemFlag(libMesh::MeshBase& mesh, MAST::FieldFunction<Real>& phi, unsigned int max_h):
        _mesh(mesh), _phi(phi), _max_h(max_h) {}
        virtual ~ElemFlag() {}
        virtual void flag_elements () {
            
            MAST::LevelSetIntersection intersection;
            
            libMesh::MeshBase::element_iterator
            it  = _mesh.active_elements_begin(),
            end = _mesh.active_elements_end();
            
            for ( ; it != end; it++) {
                
                libMesh::Elem* elem = *it;
                intersection.init( _phi, *elem, 0.,
                                  _mesh.max_elem_id(),
                                  _mesh.max_node_id());
                if (intersection.if_intersection_through_elem()) {
                    
                    Real vol_frac = intersection.get_positive_phi_volume_fraction();
                    if (vol_frac < 0.5 && elem->level() < _max_h)
                        elem->set_refinement_flag(libMesh::Elem::REFINE);
                    else if (vol_frac > 0.90)
                        elem->set_refinement_flag(libMesh::Elem::COARSEN);
                }
                else
                    elem->set_refinement_flag(libMesh::Elem::COARSEN);
                intersection.clear();
            }
        }
        
    protected:
        libMesh::MeshBase& _mesh;
        MAST::FieldFunction<Real>& _phi;
        unsigned int _max_h;
    };

    
    
    void mark_shifted_boundary(unsigned int b_id) {

        // remove the previous information for boundary id
        _mesh->boundary_info->remove_id(b_id);
        
        MAST::LevelSetIntersection intersection;
        
        libMesh::MeshBase::element_iterator
        it  = _mesh->active_local_elements_begin(),
        end = _mesh->active_local_elements_end();

        std::set<unsigned int> sides;
        std::vector<libMesh::boundary_id_type> bids;

        std::map<unsigned int, unsigned int> neighbor_side_pairs;
        neighbor_side_pairs[0] = 2;
        neighbor_side_pairs[1] = 3;
        neighbor_side_pairs[2] = 0;
        neighbor_side_pairs[3] = 1;
        
        for ( ; it != end; it++) {
            
            libMesh::Elem* elem = *it;
            
            // begin by setting all subdomain id to 0.
            elem->subdomain_id() = 0;
            
            intersection.init(*_level_set_function, *elem, 0.,
                              _mesh->max_elem_id(),
                              _mesh->max_node_id());
            if (intersection.if_intersection_through_elem()) {
                // set the shifted boundary to one which is
                // completely inside the material without any intersection
                // on the edge
                 
                sides.clear();
                intersection.get_material_sides_without_intersection(sides);
                
                // add this side in the boundary info object
                std::set<unsigned int>::const_iterator
                it  = sides.begin(),
                end = sides.end();
                for (; it != end; it++) {

                    bids.clear();
                    _mesh->boundary_info->boundary_ids(elem, *it, bids);
                    
                    // if this side has been identied as a dirichlet condition
                    // then we do not include it in the set
                    bool set_id = true;
                    //for (unsigned int i=0; i<bids.size(); i++)
                    //    if (_dirichlet_bc_ids.count(bids[i])) {
                    //        set_id = false;
                    //        break;
                    //    }
                            
                    if (set_id) {
                        
                        // find the topological neighbor and the side number
                        // for the neighbor
                        libMesh::Elem* e = elem->neighbor_ptr(*it);
                        // the side may be on the boundary, in which case
                        // no boundary should be set.
                        if (e)
                            _mesh->boundary_info->add_side(e, neighbor_side_pairs[*it], b_id);
                    }
                }
                
                // any element with an intersection will be included in the void
                // set since its interaction is included using the sifted boundary
                // method.
                elem->subdomain_id() = 3;
            }
            else if (intersection.if_elem_on_negative_phi())
                elem->subdomain_id() = 3;
            intersection.clear();
        }
    }

    //
    //  \subsection ex_5_function_evaluation Function Evaluation
    //
    void evaluate(const std::vector<Real>& dvars,
                  Real& obj,
                  bool eval_obj_grad,
                  std::vector<Real>& obj_grad,
                  std::vector<Real>& fvals,
                  std::vector<bool>& eval_grads,
                  std::vector<Real>& grads) {
        
        libMesh::out << "New Evaluation" << std::endl;
        
        // copy DVs to level set function
        libMesh::NumericVector<Real>
        &base_phi = _level_set_sys->get_vector("base_values");
        
        for (unsigned int i=0; i<_n_vars; i++)
            if (_dv_params[i].first >= base_phi.first_local_index() &&
                _dv_params[i].first <  base_phi.last_local_index())
                base_phi.set(_dv_params[i].first, dvars[i]);
        base_phi.close();
        _filter->compute_filtered_values(base_phi, *_level_set_sys->solution);
        // this will create a localized vector in _level_set_sys->curret_local_solution
        _level_set_sys->update();
        
        // create a serialized vector for use in interpolation
        std::unique_ptr<libMesh::NumericVector<Real>>
        serial_level_set_sol(libMesh::NumericVector<Real>::build(_sys->comm()).release());
        serial_level_set_sol->init(_level_set_sys->solution->size(), false, libMesh::SERIAL);
        _level_set_sys->solution->localize(*serial_level_set_sol);

        _level_set_function->init(*_level_set_sys_init, *serial_level_set_sol);
        _sys->solution->zero();

        //////////////////////////////////////////////////////////////////////
        // check to see if the sensitivity of constraint is requested
        //////////////////////////////////////////////////////////////////////
        bool if_grad_sens = false;
        for (unsigned int i=0; i<eval_grads.size(); i++)
            if_grad_sens = (if_grad_sens || eval_grads[i]);

        // if sensitivity analysis is requested, then initialize the vectors
        std::vector<libMesh::NumericVector<Real>*> sens_vecs;
        if (eval_obj_grad || if_grad_sens)
            _initialize_sensitivity_data(sens_vecs);
        

        _level_set_vel->init(*_level_set_sys_init, _level_set_function->get_mesh_function());
        
        /*if (_mesh_refinement->initialized()) {
            
            _mesh_refinement->clear_mesh();
            _eq_sys->reinit();
        }

        if (_mesh_refinement->process_mesh(*_level_set_function,
                                           true, // strong discontinuity
                                           0.,
                                           6,   // negative_level_set_subdomain_offset
                                           3,   // inactive_subdomain_offset
                                           8))  // level_set_boundary_id
            _eq_sys->reinit();*/

        
        //*********************************************************************
        // DO NOT zero out the gradient vector, since GCMMA needs it for the  *
        // subproblem solution                                                *
        //*********************************************************************
        MAST::LevelSetNonlinearImplicitAssembly                  nonlinear_assembly(true);
        MAST::LevelSetNonlinearImplicitAssembly                  level_set_assembly(false);
        MAST::LevelSetEigenproblemAssembly                       eigen_assembly;
        MAST::LevelSetStressAssembly                             stress_assembly;
        MAST::StructuralNonlinearAssemblyElemOperations          nonlinear_elem_ops;
        MAST::HeatConductionNonlinearAssemblyElemOperations      conduction_elem_ops;
        MAST::StructuralModalEigenproblemAssemblyElemOperations  modal_elem_ops;
        
        //
        // reinitialize the dof constraints before solution of the linear system
        // FIXME: we should be able to clear the constraint object from the
        // system before it goes out of scope, but libMesh::System does not
        // have a clear method. So, we are going to leave it as is, hoping
        // that libMesh::System will not attempt to use it (most likely, we
        // shoudl be ok).
        //
        /////////////////////////////////////////////////////////////////////
        // first constrain the indicator function and solve
        /////////////////////////////////////////////////////////////////////
        SNESConvergedReason r;
        /*{
            libMesh::out << "Indicator Function" << std::endl;
            nonlinear_assembly.set_discipline_and_system(*_indicator_discipline, *_indicator_sys_init);
            conduction_elem_ops.set_discipline_and_system(*_indicator_discipline, *_indicator_sys_init);
            nonlinear_assembly.set_level_set_function(*_level_set_function);
            
            MAST::LevelSetConstrainDofs constrain(*_indicator_sys_init, *_level_set_function);
            constrain.constrain_all_negative_indices(true);
            _indicator_sys->attach_constraint_object(constrain);
            _indicator_sys->reinit_constraints();
            _indicator_sys->solve(conduction_elem_ops, nonlinear_assembly);
            r = dynamic_cast<libMesh::PetscNonlinearSolver<Real>&>
            (*_indicator_sys->nonlinear_solver).get_converged_reason();
            nonlinear_assembly.clear_level_set_function();
            nonlinear_assembly.clear_discipline_and_system();
        }
        // if the solver diverged due to linear solve, then there is a problem with
        // this geometry and we need to return with a high value set for the
        // constraints
        if (r == SNES_DIVERGED_LINEAR_SOLVE) {
            
            obj = 1.e10;
            for (unsigned int i=0; i<_n_ineq; i++)
                fvals[i] = 1.e10;
            return;
        }
        
        
        /////////////////////////////////////////////////////////////////////
        // now, use the indicator function to constrain dofs in the structural
        // system
        /////////////////////////////////////////////////////////////////////
        MAST::MeshFieldFunction indicator(*_indicator_sys_init, "indicator");
        indicator.init(*_indicator_sys->solution);
        MAST::IndicatorFunctionConstrainDofs constrain(*_sys_init, *_level_set_function, indicator);
        MAST::LevelSetConstrainDofs constrain(*_sys_init, *_level_set_function);
        _sys->attach_constraint_object(constrain);
        _sys->reinit_constraints();
        _sys->initialize_condensed_dofs(*_discipline);*/
        
        /////////////////////////////////////////////////////////////////////
        // first constrain the indicator function and solve
        /////////////////////////////////////////////////////////////////////
        nonlinear_assembly.set_discipline_and_system(*_discipline, *_sys_init);
        nonlinear_assembly.set_level_set_function(*_level_set_function, *_filter);
        nonlinear_assembly.set_level_set_velocity_function(*_level_set_vel);
        //nonlinear_assembly.set_indicator_function(indicator);
        eigen_assembly.set_discipline_and_system(*_discipline, *_sys_init);
        eigen_assembly.set_level_set_function(*_level_set_function);
        eigen_assembly.set_level_set_velocity_function(*_level_set_vel);
        stress_assembly.set_discipline_and_system(*_discipline, *_sys_init);
        stress_assembly.init(*_level_set_function, nonlinear_assembly.if_use_dof_handler()?&nonlinear_assembly.get_dof_handler():nullptr);
        level_set_assembly.set_discipline_and_system(*_level_set_discipline, *_level_set_sys_init);
        level_set_assembly.set_level_set_function(*_level_set_function, *_filter);
        level_set_assembly.set_level_set_velocity_function(*_level_set_vel);
        nonlinear_elem_ops.set_discipline_and_system(*_discipline, *_sys_init);
        modal_elem_ops.set_discipline_and_system(*_discipline, *_sys_init);
        
        
        libMesh::MeshRefinement refine(*_mesh);
        
        libMesh::out << "before refinement" << std::endl;
        _mesh->print_info();

        bool
        continue_refining    = true;
        Real
        threshold            = _input("refinement_threshold","threshold for element to be refined", 0.1);
        unsigned int
        n_refinements        = 0,
        max_refinements      = _input("max_refinements","maximum refinements", 3);
        
        while (n_refinements < max_refinements && continue_refining) {
            
            // The ErrorVector is a particular StatisticsVector
            // for computing error information on a finite element mesh.
            libMesh::ErrorVector error(_mesh->max_elem_id(), _mesh);
            _compute_element_errors(error);
            libMesh::out
            << "After refinement: " << n_refinements << std::endl
            << "max error:    " << error.maximum()
            << ",  mean error: " << error.mean() << std::endl;

            if (error.maximum() > threshold) {
                
                ElemFlag flag(*_mesh, *_level_set_function, max_refinements);
                refine.max_h_level()      = max_refinements;
                refine.refine_fraction()  = 1.;
                refine.coarsen_fraction() = 0.5;
                refine.flag_elements_by (flag);
                if (refine.refine_and_coarsen_elements())
                    _eq_sys->reinit ();

                _mesh->print_info();
                
                n_refinements++;
            }
            else
                continue_refining = false;
        }
        /*if (_mesh_refinement->process_mesh(*_level_set_function,
                                           true, // strong discontinuity
                                           0.,
                                           6,   // negative_level_set_subdomain_offset
                                           3,   // inactive_subdomain_offset
                                           8))  // level_set_boundary_id
            _eq_sys->reinit();*/

        
        MAST::LevelSetVolume                            volume;
        MAST::LevelSetPerimeter                         perimeter;
        MAST::StressStrainOutputBase                    stress;
        MAST::ComplianceOutput                          compliance;
        volume.set_discipline_and_system(*_level_set_discipline, *_level_set_sys_init);
        perimeter.set_discipline_and_system(*_level_set_discipline, *_level_set_sys_init);
        stress.set_discipline_and_system(*_discipline, *_sys_init);
        volume.set_participating_elements_to_all();
        perimeter.set_participating_elements_to_all();
        stress.set_participating_elements_to_all();
        stress.set_aggregation_coefficients(_p_val, 1., _vm_rho, _stress_lim) ;
        compliance.set_participating_elements_to_all();
        compliance.set_discipline_and_system(*_discipline, *_sys_init);

        //////////////////////////////////////////////////////////////////////
        // evaluate the stress constraint
        //////////////////////////////////////////////////////////////////////
        // tell the thermal jacobian scaling object about the assembly object
        
        libMesh::out << "Static Solve" << std::endl;
        _sys->solve(nonlinear_elem_ops, nonlinear_assembly);
        r = dynamic_cast<libMesh::PetscNonlinearSolver<Real>&>
        (*_sys->nonlinear_solver).get_converged_reason();
        
        // if the solver diverged due to linear solve, then there is a problem with
        // this geometry and we need to return with a high value set for the
        // constraints
        if (r == SNES_DIVERGED_LINEAR_SOLVE ||
            _sys->final_nonlinear_residual() > 1.e-1) {
            
            obj = 1.e10;
            for (unsigned int i=0; i<_n_ineq; i++)
                fvals[i] = 1.e10;
            return;
        }
        
        // this will localize the solution vector for later use
        _sys->update();
        
        //////////////////////////////////////////////////////////////////////
        // evaluate the functions
        //////////////////////////////////////////////////////////////////////

        
        level_set_assembly.set_evaluate_output_on_negative_phi(false);
        level_set_assembly.calculate_output(*_level_set_sys->solution, true, volume);
        level_set_assembly.set_evaluate_output_on_negative_phi(true);
        level_set_assembly.calculate_output(*_level_set_sys->solution, true, perimeter);
        level_set_assembly.set_evaluate_output_on_negative_phi(false);

        Real
        max_vm = 0.,
        vm_agg = 0.,
        vol    = volume.output_total(),
        per    = perimeter.output_total(),
        comp   = 0.;

        libMesh::out << "volume: " << vol    << "  perim: "  << per    << std::endl;
        
        // evaluate the output based on specified problem type
        if (_problem == "compliance_volume") {
            
            Real
            vf     = _input("volume_fraction", "volume fraction", 0.3);

            // if the shifted boundary is implementing a traction-free condition
            // compliance does not need contribution from shifted boundary load
            nonlinear_assembly.calculate_output(*_sys->current_local_solution, false, compliance);
            comp      = compliance.output_total();
            obj       = _obj_scaling * (comp + _perimeter_penalty * per);
            fvals[0]  = vol/_volume - vf; // vol/vol0 - a <=
            libMesh::out << "compliance: " << comp << std::endl;
        }
        else if (_problem == "volume_stress") {
            
            // set the elasticity penalty for stress evaluation
            nonlinear_assembly.calculate_output(*_sys->current_local_solution, false, stress);
            max_vm    = stress.get_maximum_von_mises_stress();
            vm_agg    = stress.output_total();
            obj       = _obj_scaling * (vol + _perimeter_penalty * per);
            fvals[0]  =  stress.output_total()/_stress_lim - 1.;  // g = sigma/sigma0-1 <= 0
            //fvals[0]  =  stress.output_total()/_length/_height;  // g <= 0 for the smooth ramp function
            libMesh::out
            << "  max: "    << max_vm
            << "  constr: " << vm_agg
            << std::endl;
        }
        else if (_problem == "volume_eigenvalue" && _n_eig_vals) {
            
            //////////////////////////////////////////////////////////////////////
            // evaluate the eigenvalue constraint
            //////////////////////////////////////////////////////////////////////
            libMesh::out << "Eigen Solve" << std::endl;
            _sys->eigenproblem_solve(modal_elem_ops, eigen_assembly);
            Real eig_imag = 0.;
            //
            // hopefully, the solver found the requested number of eigenvalues.
            // if not, then we will set zero values for the ones it did not.
            //
            unsigned int n_conv = std::min(_n_eig_vals, _sys->get_n_converged_eigenvalues());
            std::vector<Real> eig(_n_eig_vals, 0.);
            
            // get the converged eigenvalues
            for (unsigned int i=0; i<n_conv; i++)      _sys->get_eigenvalue(0, eig[i], eig_imag);
            //
            //  eig > eig0
            //  -eig < -eig0
            //  -eig/eig0 < -1
            // -eig/eig0 + 1 < 0
            //
            for (unsigned int i=0; i<_n_eig_vals; i++)
                fvals[i+1] = -eig[i]/_ref_eig_val + 1.;
        }
        else
            libmesh_error();

        //////////////////////////////////////////////////////////////////////
        // evaluate the objective sensitivities, if requested
        //////////////////////////////////////////////////////////////////////
        if (eval_obj_grad) {
            
            if (_problem == "compliance_volume") {
                
                std::vector<Real>
                grad1(obj_grad.size(), 0.);

                _evaluate_volume_sensitivity(sens_vecs, nullptr, &perimeter, level_set_assembly, obj_grad);

                _evaluate_compliance_sensitivity(compliance,
                                                 nonlinear_elem_ops,
                                                 nonlinear_assembly,
                                                 grad1);

                for (unsigned int i=0; i<obj_grad.size(); i++) {
                    obj_grad[i] += grad1[i];
                    obj_grad[i] *= _obj_scaling;
                }
            }
            else if (_problem == "volume_stress") {
                
                _evaluate_volume_sensitivity(sens_vecs, &volume, &perimeter, level_set_assembly, obj_grad);
                for (unsigned int i=0; i<obj_grad.size(); i++)
                    obj_grad[i] *= _obj_scaling;
            }
            else
                libmesh_error();
        }
                
        //////////////////////////////////////////////////////////////////////
        // evaluate the sensitivities for constraints
        //////////////////////////////////////////////////////////////////////
        if (if_grad_sens) {

            if (_problem == "compliance_volume") {
                
                _evaluate_volume_sensitivity(sens_vecs, &volume, nullptr, level_set_assembly, grads);
                for (unsigned int i=0; i<grads.size(); i++)
                    grads[i] /= _volume;
            }
            else if (_problem == "volume_stress") {
                
                _evaluate_stress_sensitivity(stress,
                                             nonlinear_elem_ops,
                                             nonlinear_assembly,
                                             modal_elem_ops,
                                             eigen_assembly,
                                             grads);
            }
            else
                libmesh_error();
        }
        
        //
        // also the stress data for plotting
        //
        stress_assembly.update_stress_strain_data(stress, *_sys->solution);
        _clear_sensitivity_data(sens_vecs);
    }

    //
    // \subsection ex_5_sensitivity_vectors Initialize sensitivity data
    //
    void _initialize_sensitivity_data(std::vector<libMesh::NumericVector<Real>*>& dphi_vecs) {

        libmesh_assert_equal_to(dphi_vecs.size(), 0);
        
        dphi_vecs.resize(_n_vars, nullptr);
        
        // Serial vectors are used for the level
        // set mesh function since it uses a different mesh than the analysis mesh
        // and the two can have different partitionings in the paralle environment.
        for (unsigned int i=0; i<_n_vars; i++) {

            libMesh::NumericVector<Real>
            *vec = nullptr;

            // non-zero value of the DV perturbation
            std::map<unsigned int, Real> nonzero_val;
            nonzero_val[_dv_params[i].first] = 1.;
            
            vec = libMesh::NumericVector<Real>::build(_sys->comm()).release();
            vec->init(_level_set_sys->solution->size(), false, libMesh::SERIAL);
            _filter->compute_filtered_values(nonzero_val, *vec, false);

            dphi_vecs[i] = vec;
        }

        for ( unsigned int i=0; i<_n_vars; i++)
            dphi_vecs[i]->close();

        for ( unsigned int i=0; i<_n_vars; i++) {

            // we will use this serialized vector to initialize the mesh function,
            // which is setup to reuse this vector, so we have to store it
            _level_set_function->init_sens(*_dv_params[i].second, *dphi_vecs[i]);
        }
    }

    
    void _clear_sensitivity_data(std::vector<libMesh::NumericVector<Real>*>& dphi_vecs) {

        // delete the vectors that we do not need any more
        for (unsigned int i=0; i<dphi_vecs.size(); i++)
            delete dphi_vecs[i];
        dphi_vecs.clear();
        _level_set_vel->clear();
    }
    

    //
    //  \subsection ex_5_volume_sensitivity Sensitivity of Material Volume
    //
    void _evaluate_volume_sensitivity(const std::vector<libMesh::NumericVector<Real>*>& dphi_vecs,
                                      MAST::LevelSetVolume*    volume,
                                      MAST::LevelSetPerimeter* perimeter,
                                      MAST::LevelSetNonlinearImplicitAssembly& assembly,
                                      std::vector<Real>& grad) {
        
        std::fill(grad.begin(), grad.end(), 0.);
        
        ElementParameterDependence dep(*_filter);
        assembly.attach_elem_parameter_dependence_object(dep);
        
        if (volume)    volume->set_skip_comm_sum(true);
        if (perimeter) perimeter->set_skip_comm_sum(true);
        
        
        for (unsigned int i=0; i<_n_vars; i++) {
            
            // if the volume output was specified then compute the sensitivity
            // and add to the grad vector
            if (volume) {

                assembly.set_evaluate_output_on_negative_phi(false);
                assembly.calculate_output_direct_sensitivity(*_level_set_sys->current_local_solution,
                                                             false,
                                                             dphi_vecs[i],
                                                             false,
                                                             *_dv_params[i].second,
                                                             *volume);
                
                grad[i] = volume->output_sensitivity_total(*_dv_params[i].second);
            }
            
            // if the perimeter output was specified then compute the sensitivity
            // and add to the grad vector
            if (perimeter) {
                assembly.set_evaluate_output_on_negative_phi(true);
                assembly.calculate_output_direct_sensitivity(*_level_set_sys->current_local_solution,
                                                             false,
                                                             dphi_vecs[i],
                                                             false,
                                                             *_dv_params[i].second,
                                                             *perimeter);
                assembly.set_evaluate_output_on_negative_phi(false);
                
                grad[i] += _perimeter_penalty *
                perimeter->output_sensitivity_total(*_dv_params[i].second);
            }
        }

        _sys->comm().sum(grad);
        
        if (volume)    volume->set_skip_comm_sum(false);
        if (perimeter) perimeter->set_skip_comm_sum(false);

        assembly.clear_elem_parameter_dependence_object();
    }
    
    
    
    //
    //  \subsection ex_5_stress_sensitivity Sensitivity of Stress and Eigenvalues
    //
    void
    _evaluate_stress_sensitivity
    (MAST::StressStrainOutputBase& stress,
     MAST::AssemblyElemOperations& nonlinear_elem_ops,
     MAST::NonlinearImplicitAssembly& nonlinear_assembly,
     MAST::StructuralModalEigenproblemAssemblyElemOperations& eigen_elem_ops,
     MAST::LevelSetEigenproblemAssembly& eigen_assembly,
     std::vector<Real>& grads) {
        
        unsigned int n_conv = std::min(_n_eig_vals, _sys->get_n_converged_eigenvalues());
        
        _sys->adjoint_solve(*_sys->current_local_solution,
                            false,
                            nonlinear_elem_ops,
                            stress,
                            nonlinear_assembly,
                            false);
        
        ElementParameterDependence dep(*_filter);
        nonlinear_assembly.attach_elem_parameter_dependence_object(dep);

        //////////////////////////////////////////////////////////////////
        // indices used by GCMMA follow this rule:
        // grad_k = dfi/dxj  ,  where k = j*NFunc + i
        //////////////////////////////////////////////////////////////////
        // first compute the sensitivity contribution from dot product of adjoint vector
        // and residual sensitivity
        std::vector<Real>
        g1(_n_vars, 0.),
        g2(_n_vars, 0.);
        std::vector<const MAST::FunctionBase*>
        p_vec(_n_vars, nullptr);
        for (unsigned int i=0; i<_n_vars; i++)
            p_vec[i] = _dv_params[i].second;
        
        //////////////////////////////////////////////////////////////////////
        // stress sensitivity
        //////////////////////////////////////////////////////////////////////
        // set the elasticity penalty for solution, which is needed for
        // computation of the residual sensitivity
        nonlinear_assembly.calculate_output_adjoint_sensitivity_multiple_parameters_no_direct
        (*_sys->current_local_solution,
         false,
         _sys->get_adjoint_solution(),
         p_vec,
         nonlinear_elem_ops,
         stress,
         g1);

        // we will skip the summation of sensitivity inside the stress object to minimize
        // communication cost. Instead, we will do it at the end for the constraint vector
        stress.set_skip_comm_sum(true);
        for (unsigned int i=0; i<_n_vars; i++) {
            
            nonlinear_assembly.calculate_output_adjoint_sensitivity(*_sys->current_local_solution,
                                                                    false,
                                                                    _sys->get_adjoint_solution(),
                                                                    *_dv_params[i].second,
                                                                    nonlinear_elem_ops,
                                                                    stress);
            g2[i] = stress.output_sensitivity_total(*_dv_params[i].second);
            stress.clear_sensitivity_data();
            
            //////////////////////////////////////////////////////////////////////
            // eigenvalue sensitivity, only if the values were requested
            //////////////////////////////////////////////////////////////////////
            if (_n_eig_vals) {
                
                std::vector<Real> sens;
                _sys->eigenproblem_sensitivity_solve(eigen_elem_ops,
                                                     eigen_assembly,
                                                     *_dv_params[i].second,
                                                     sens);
                for (unsigned int j=0; j<n_conv; j++)
                    grads[_n_ineq*i+j+1] = -sens[j]/_ref_eig_val;
            }
        }
        stress.set_skip_comm_sum(false);

        // now sum the values across processors to sum the partial derivatives for
        // each parameter
        _sys->comm().sum(g2);
        
        // now compute contribution to the stress constraint
        for (unsigned int i=0; i<_n_vars; i++)
            grads[1*i+0] = 1./_stress_lim * (g1[i] + g2[i]);

        nonlinear_assembly.clear_elem_parameter_dependence_object();
    }

    
    void
    _evaluate_compliance_sensitivity
    (MAST::ComplianceOutput& compliance,
     MAST::AssemblyElemOperations& nonlinear_elem_ops,
     MAST::NonlinearImplicitAssembly& nonlinear_assembly,
     std::vector<Real>& grads) {
        
        // Adjoint solution for compliance = - X
        // if the shifted boundary is implementing a traction-free condition
        // compliance does not need contribution from shifted boundary load
        _sys->adjoint_solve(*_sys->current_local_solution,
                            false,
                            nonlinear_elem_ops,
                            compliance,
                            nonlinear_assembly,
                            false);

        ElementParameterDependence dep(*_filter);
        nonlinear_assembly.attach_elem_parameter_dependence_object(dep);

        //////////////////////////////////////////////////////////////////
        // indices used by GCMMA follow this rule:
        // grad_k = dfi/dxj  ,  where k = j*NFunc + i
        //////////////////////////////////////////////////////////////////
        // first compute the sensitivity contribution from dot product of adjoint vector
        // and residual sensitivity
        std::vector<Real>
        g1(_n_vars, 0.),
        g2(_n_vars, 0.);
        std::vector<const MAST::FunctionBase*>
        p_vec(_n_vars, nullptr);
        for (unsigned int i=0; i<_n_vars; i++)
            p_vec[i] = _dv_params[i].second;
        
        //////////////////////////////////////////////////////////////////////
        // compliance sensitivity
        //////////////////////////////////////////////////////////////////////
        // set the elasticity penalty for solution, which is needed for
        // computation of the residual sensitivity
        nonlinear_assembly.calculate_output_adjoint_sensitivity_multiple_parameters_no_direct
        (*_sys->current_local_solution,
         false,
         _sys->get_adjoint_solution(),
         p_vec,
         nonlinear_elem_ops,
         compliance,
         g1);

        compliance.set_skip_comm_sum(true);
        for (unsigned int i=0; i<_n_vars; i++) {
            
            nonlinear_assembly.calculate_output_direct_sensitivity(*_sys->current_local_solution,
                                                                   false,
                                                                   nullptr,
                                                                   false,
                                                                   *_dv_params[i].second,
                                                                   compliance);
            g2[i] = compliance.output_sensitivity_total(*_dv_params[i].second);
        }
        compliance.set_skip_comm_sum(false);

        // now sum the values across processors to sum the partial derivatives for
        // each parameter
        _sys->comm().sum(g2);

        for (unsigned int i=0; i<_n_vars; i++)
            grads[i] = (g1[i] + g2[i]);

        nonlinear_assembly.clear_elem_parameter_dependence_object();
    }
    
    
    void set_n_vars(const unsigned int n_vars) {_n_vars = n_vars;}

    //
    //  \subsection ex_5_design_output  Output of Design Iterate
    //
    void output(unsigned int iter,
                const std::vector<Real>& x,
                Real obj,
                const std::vector<Real>& fval,
                bool if_write_to_optim_file) {
        
        libmesh_assert_equal_to(x.size(), _n_vars);
        
        Real
        sys_time     = _sys->time;
        
        std::string
        output_name  = _input("output_file_root", "prefix of output file names", "output"),
        modes_name   = output_name + "modes.exo";
        
        std::ostringstream oss;
        oss << "output_optim.e-s." << std::setfill('0') << std::setw(5) << iter ;
        
        //
        // copy DVs to level set function
        //
        libMesh::NumericVector<Real>
        &base_phi = _level_set_sys->get_vector("base_values");
        
        for (unsigned int i=0; i<_n_vars; i++)
            if (_dv_params[i].first >= base_phi.first_local_index() &&
                _dv_params[i].first <  base_phi.last_local_index())
                base_phi.set(_dv_params[i].first, x[i]);
        base_phi.close();
        _filter->compute_filtered_values(base_phi, *_level_set_sys->solution);
        // create a serialized vector for use in interpolation
        std::unique_ptr<libMesh::NumericVector<Real>>
        serial_level_set_sol(libMesh::NumericVector<Real>::build(_sys->comm()).release());
        serial_level_set_sol->init(_level_set_sys->solution->size(), false, libMesh::SERIAL);
        _level_set_sys->solution->localize(*serial_level_set_sol);

        _level_set_function->init(*_level_set_sys_init, *serial_level_set_sol);
        _level_set_sys_init_on_str_mesh->initialize_solution(_level_set_function->get_mesh_function());
        
        std::vector<bool> eval_grads(this->n_ineq(), false);
        std::vector<Real> f(this->n_ineq(), 0.), grads;
        this->evaluate(x, obj, false, grads, f, eval_grads, grads);
        
        _sys->time = iter;
        _sys_init->get_stress_sys().time = iter;
        // "1" is the number of time-steps in the file, as opposed to the time-step number.
        libMesh::ExodusII_IO(*_mesh).write_timestep(oss.str(), *_eq_sys, 1, (1.*iter));
        
        if (_n_eig_vals) {
            
            //////////////////////////////////////////////////////////////////////////
            // eigenvalue analysis: write modes to file
            //////////////////////////////////////////////////////////////////////////
            libMesh::ExodusII_IO writer(*_mesh);
            Real eig_r, eig_i;
            for (unsigned int i=0; i<_sys->get_n_converged_eigenvalues(); i++) {
                _sys->get_eigenpair(i, eig_r, eig_i, *_sys->solution);
                writer.write_timestep(modes_name, *_eq_sys, i+1, i);
            }
            _sys->solution->zero();
        }
        
        //
        // set the value of time back to its original value
        //
        _sys->time = sys_time;
        
        //
        // increment the parameter values
        //
        unsigned int
        update_freq = _input("update_freq_optim_params", "number of iterations after which the optimization parameters are updated", 50),
        factor = iter/update_freq ;
        if (factor > 0 && iter%update_freq == 0) {
            
            Real
            p_val           = _input("constraint_aggregation_p_val", "value of p in p-norm stress aggregation", 2.0),
            vm_rho          = _input("constraint_aggregation_rho_val", "value of rho in p-norm stress aggregation", 2.0),
            constr_penalty  = _input("constraint_penalty", "constraint penalty in GCMMA",      50.),
            max_penalty     = _input("max_constraint_penalty", "maximum constraint penalty in GCMMA",      1.e7),
            initial_step    = _input("initial_rel_step", "initial relative step length in GCMMA",      0.5),
            min_step        = _input("minimum_rel_step", "minimum relative step length in GCMMA",      0.001);
            
            constr_penalty = std::min(constr_penalty*pow(10, factor), max_penalty);
            initial_step   = std::max(initial_step-0.01*factor, min_step);
            _p_val         = std::min(p_val+2*factor, 10.);
            _vm_rho        = std::min(vm_rho+factor*0.5, 2.);
            libMesh::out
            << "Updated values: c = " << constr_penalty
            << "  step = " << initial_step
            << "  p = " << _p_val
            << "  rho = " << _vm_rho << std::endl;
            
            _optimization_interface->set_real_parameter   ( "constraint_penalty",   constr_penalty);
            _optimization_interface->set_real_parameter   ("initial_rel_step",        initial_step);
        }

        MAST::FunctionEvaluation::output(iter, x, obj/_obj_scaling, f, if_write_to_optim_file);
    }

#if MAST_ENABLE_SNOPT == 1
    MAST::FunctionEvaluation::funobj
    get_objective_evaluation_function() {
    
        return _optim_obj;
    }

    MAST::FunctionEvaluation::funcon
    get_constraint_evaluation_function() {
    
        return _optim_con;
    }
#endif
    
    
    //
    // \section  ex_5_initialization  Initialization
    //
    //   \subsection ex_5_constructor  Constructor
    //
    
    TopologyOptimizationLevelSet(const libMesh::Parallel::Communicator& comm_in,
                                   MAST::Examples::GetPotWrapper& input):
    MAST::FunctionEvaluation             (comm_in),
    _initialized                         (false),
    _input                               (input),
    _problem                             (),
    _volume                              (0.),
    _obj_scaling                         (0.),
    _stress_penalty                      (0.),
    _perimeter_penalty                   (0.),
    _stress_lim                          (0.),
    _p_val                               (0.),
    _vm_rho                              (0.),
    _ref_eig_val                         (0.),
    _n_eig_vals                          (0),
    _mesh                                (nullptr),
    _level_set_mesh                      (nullptr),
    _mesh_refinement                     (nullptr),
    _eq_sys                              (nullptr),
    _level_set_eq_sys                    (nullptr),
    _sys                                 (nullptr),
    _level_set_sys                       (nullptr),
    _level_set_sys_on_str_mesh           (nullptr),
    _indicator_sys                       (nullptr),
    _sys_init                            (nullptr),
    _level_set_sys_init_on_str_mesh      (nullptr),
    _level_set_sys_init                  (nullptr),
    _indicator_sys_init                  (nullptr),
    _nsp                                 (nullptr),
    _discipline                          (nullptr),
    _indicator_discipline                (nullptr),
    _level_set_discipline                (nullptr),
    _filter                              (nullptr),
    _m_card                              (nullptr),
    _p_card                              (nullptr),
    _level_set_function                  (nullptr),
    _level_set_vel                       (nullptr),
    _output                              (nullptr),
    _shifted_boundary_load               (nullptr) {
        
        libmesh_assert(!_initialized);
        
        //
        // call the initialization routines for each component
        //
        
        std::string
        s  = _input("mesh", "type of mesh to be analyzed {inplane, bracket, truss, eyebar}",
                    "inplane");

        _mesh           = new libMesh::SerialMesh(this->comm());
        _level_set_mesh = new libMesh::SerialMesh(this->comm());

        
        _init_fetype();
        T::init_analysis_mesh(*this, *_mesh);
        T::init_level_set_mesh(*this, *_level_set_mesh);
        _init_system_and_discipline();
        T::init_analysis_dirichlet_conditions(*this);
        T::init_indicator_dirichlet_conditions(*this);
        _init_eq_sys();
        _init_material();
        T::init_structural_loads(*this);
        T::init_indicator_loads(*this);
        _init_section_property();
        _initialized = true;
        
        _nsp  = new MAST::StructuralNearNullVectorSpace;
        _sys->nonlinear_solver->nearnullspace_object = _nsp;

        /////////////////////////////////////////////////
        // now initialize the design data.
        /////////////////////////////////////////////////
        
        //
        // first, initialize the level set functions over the domain
        //
        T::initialize_level_set_solution(*this);
        
        //
        // next, define a new parameter to define design variable for nodal level-set
        // function value
        //
        T::init_level_set_dvs(*this);
        
        Real
        filter_radius          = _input("filter_radius", "radius of geometric filter for level set field", 0.015);
        _filter                = new MAST::FilterBase(*_level_set_sys, filter_radius, _dv_dof_ids);
        libMesh::NumericVector<Real>& vec = _level_set_sys->add_vector("base_values");
        vec = *_level_set_sys->solution;
        vec.close();

        
        _problem               = _input("problem_type", "{compliance_volume, volume_stress}", "compliance_volume");
        _volume                = T::reference_volume(*this);
        _obj_scaling           = 1./_volume;
        _stress_penalty        = _input("stress_penalty", "penalty value for stress_constraint", 0.);
        _perimeter_penalty     = _input("perimeter_penalty", "penalty value for perimeter in the objective function", 0.);
        _stress_lim            = _input("vm_stress_limit", "limit von-mises stress value", 2.e8);
        _p_val                 = _input("constraint_aggregation_p_val", "value of p in p-norm stress aggregation", 2.0);
        _vm_rho                = _input("constraint_aggregation_rho_val", "value of rho in p-norm stress aggregation", 2.0);
        _level_set_vel         = new MAST::LevelSetBoundaryVelocity(2);
        _level_set_function    = new PhiMeshFunction;
        _output                = new libMesh::ExodusII_IO(*_mesh);
        
        MAST::BoundaryConditionBase
        *bc = new MAST::BoundaryConditionBase(MAST::BOUNDARY_VELOCITY);
        bc->add(*_level_set_vel);
        _discipline->add_side_load(8, *bc);
        _boundary_conditions.insert(bc);

        _n_eig_vals            = _input("n_eig", "number of eigenvalues to constrain", 0);
        if (_n_eig_vals) {
            //
            // set only if the user requested eigenvalue constraints
            //
            _ref_eig_val           = _input("eigenvalue_low_bound", "lower bound enforced on eigenvalue constraints", 1.e3);
            _sys->set_n_requested_eigenvalues(_n_eig_vals);
        }
        
        //
        // two inequality constraints: stress and eigenvalue.
        //
        _n_ineq = 1+_n_eig_vals;
        
        std::string
        output_name = _input("output_file_root", "prefix of output file names", "output");
        output_name += "_optim_history.txt";
        this->set_output_file(output_name);
        
    }
    
    //
    //   \subsection ex_5_destructor  Destructor
    //
    ~TopologyOptimizationLevelSet() {
        
        {
            std::set<MAST::BoundaryConditionBase*>::iterator
            it   = _boundary_conditions.begin(),
            end  = _boundary_conditions.end();
            for ( ; it!=end; it++)
                delete *it;
        }
        
        {
            std::set<MAST::FunctionBase*>::iterator
            it   = _field_functions.begin(),
            end  = _field_functions.end();
            for ( ; it!=end; it++)
                delete *it;
        }
        
        {
            std::map<std::string, MAST::Parameter*>::iterator
            it   = _parameters.begin(),
            end  = _parameters.end();
            for ( ; it!=end; it++)
                delete it->second;
        }
        
        if (!_initialized)
            return;
        
        delete _nsp;

        delete _m_card;
        delete _p_card;

        delete _eq_sys;
        delete _mesh_refinement;
        delete _mesh;
        
        delete _discipline;
        delete _sys_init;
        
        delete _level_set_function;
        delete _level_set_vel;
        delete _level_set_sys_init;
        delete _indicator_sys_init;
        delete _indicator_discipline;
        delete _level_set_discipline;
        delete _filter;
        delete _level_set_eq_sys;
        delete _level_set_mesh;
        delete _output;
        delete _level_set_sys_init_on_str_mesh;
        
        for (unsigned int i=0; i<_dv_params.size(); i++)
            delete _dv_params[i].second;
    }
    

};


//
//   \subsection ex_5_wrappers_snopt  Wrappers for SNOPT
//

MAST::FunctionEvaluation* _my_func_eval = nullptr;

#if MAST_ENABLE_SNOPT == 1

unsigned int
it_num = 0;

void
_optim_obj(int*    mode,
           int*    n,
           double* x,
           double* f,
           double* g,
           int*    nstate) {

    //
    // make sure that the global variable has been setup
    //
    libmesh_assert(_my_func_eval);

    //
    // initialize the local variables
    //
    Real
    obj = 0.;

    unsigned int
    n_vars  =  _my_func_eval->n_vars(),
    n_con   =  _my_func_eval->n_eq()+_my_func_eval->n_ineq();

    libmesh_assert_equal_to(*n, n_vars);

    std::vector<Real>
    dvars   (*n,    0.),
    obj_grad(*n,    0.),
    fvals   (n_con, 0.),
    grads   (0);

    std::vector<bool>
    eval_grads(n_con);
    std::fill(eval_grads.begin(), eval_grads.end(), false);
    
    //
    // copy the dvars
    //
    for (unsigned int i=0; i<n_vars; i++)
        dvars[i] = x[i];


    _my_func_eval->_evaluate_wrapper(dvars,
                                     obj,
                                     *mode>0,       // request the derivatives of obj
                                     obj_grad,
                                     fvals,
                                     eval_grads,
                                     grads);

    //
    // now copy them back as necessary
    //
    *f  = obj;
    if (*mode > 0) {
        
        // output data to the file
        _my_func_eval->_output_wrapper(it_num, dvars, obj, fvals, true);
        it_num++;

        for (unsigned int i=0; i<n_vars; i++)
            g[i] = obj_grad[i];
    }

    if (obj > 1.e5) *mode = -1;
}






void
_optim_con(int*    mode,
           int*    ncnln,
           int*    n,
           int*    ldJ,
           int*    needc,
           double* x,
           double* c,
           double* cJac,
           int*    nstate) {

    //
    // make sure that the global variable has been setup
    //
    libmesh_assert(_my_func_eval);

    //
    // initialize the local variables
    //
    Real
    obj = 0.;

    unsigned int
    n_vars  =  _my_func_eval->n_vars(),
    n_con   =  _my_func_eval->n_eq()+_my_func_eval->n_ineq();

    libmesh_assert_equal_to(    *n, n_vars);
    libmesh_assert_equal_to(*ncnln, n_con);

    std::vector<Real>
    dvars   (*n,    0.),
    obj_grad(*n,    0.),
    fvals   (n_con, 0.),
    grads   (n_vars*n_con, 0.);

    std::vector<bool>
    eval_grads(n_con);
    std::fill(eval_grads.begin(), eval_grads.end(), *mode>0);

    //
    // copy the dvars
    //
    for (unsigned int i=0; i<n_vars; i++)
        dvars[i] = x[i];


    _my_func_eval->_evaluate_wrapper(dvars,
                                     obj,
                                     false,       // request the derivatives of obj
                                     obj_grad,
                                     fvals,
                                     eval_grads,
                                     grads);

    //
    // now copy them back as necessary
    //
    // first the constraint functions
    //
    for (unsigned int i=0; i<n_con; i++)
        c[i] = fvals[i];

    if (*mode > 0) {
        //
        // next, the constraint gradients
        //
        for (unsigned int i=0; i<n_con*n_vars; i++)
            cJac[i] = grads[i];
    }
    
    if (obj > 1.e5) *mode = -1;
}
#endif

//
//   \subsection ex_5_main Main function
//

int main(int argc, char* argv[]) {

    libMesh::LibMeshInit init(argc, argv);

    MAST::Examples::GetPotWrapper
    input(argc, argv, "input");


    std::unique_ptr<MAST::FunctionEvaluation>
    top_opt;
    
    std::string
    mesh = input("mesh", "inplane2d, bracket2d, truss2d, eyebar2d", "inplane2d");
    
    if (mesh == "inplane2d") {
        top_opt.reset
        (new TopologyOptimizationLevelSet<MAST::Examples::Inplane2DModel>
         (init.comm(), input));
    }
    else if (mesh == "bracket2d") {
        top_opt.reset
        (new TopologyOptimizationLevelSet<MAST::Examples::Bracket2DModel>
         (init.comm(), input));
    }
    else if (mesh == "eyebar2d") {
        top_opt.reset
        (new TopologyOptimizationLevelSet<MAST::Examples::Eyebar2DModel>
         (init.comm(), input));
    }
    else if (mesh == "truss2d") {
        top_opt.reset
        (new TopologyOptimizationLevelSet<MAST::Examples::Truss2DModel>
         (init.comm(), input));
    }
    else
        libmesh_error();
    
    _my_func_eval = top_opt.get();

    
    std::unique_ptr<MAST::OptimizationInterface> optimizer;
    
    std::string
    s          = input("optimizer", "optimizer to use in the example", "gcmma");

    if (s == "gcmma") {

        optimizer.reset(new MAST::GCMMAOptimizationInterface);
        
        unsigned int
        max_inner_iters        = input("max_inner_iters", "maximum inner iterations in GCMMA", 15);
        
        Real
        constr_penalty         = input("constraint_penalty", "constraint penalty in GCMMA", 50.),
        initial_rel_step       = input("initial_rel_step", "initial step size in GCMMA", 1.e-2),
        asymptote_reduction    = input("asymptote_reduction", "reduction of aymptote in GCMMA", 0.7),
        asymptote_expansion    = input("asymptote_expansion", "expansion of asymptote in GCMMA", 1.2);
        
        optimizer->set_real_parameter   ("constraint_penalty",  constr_penalty);
        optimizer->set_real_parameter   ("initial_rel_step",  initial_rel_step);
        optimizer->set_real_parameter   ("asymptote_reduction",  asymptote_reduction);
        optimizer->set_real_parameter   ("asymptote_expansion",  asymptote_expansion);
        optimizer->set_integer_parameter(   "max_inner_iters", max_inner_iters);
    }
    else if (s == "snopt") {
        
        //optimizer.reset(new MAST::NPSOLOptimizationInterface);
    }
    else {
        
        libMesh::out
        << "Unrecognized optimizer specified: " << s << std::endl;
        libmesh_error();
    }
    
    if (optimizer.get()) {
        
        optimizer->attach_function_evaluation_object(*top_opt);

        bool
        verify_grads = input("verify_gradients", "If true, the gradients of objective and constraints will be verified without optimization", false);
        if (verify_grads) {
            
            std::vector<Real> xx1(top_opt->n_vars()), xx2(top_opt->n_vars());
            top_opt->init_dvar(xx1, xx2, xx2);
            top_opt->verify_gradients(xx1);
        }
        else
            optimizer->optimize();
    }
    
    // END_TRANSLATE
    return 0;
}
