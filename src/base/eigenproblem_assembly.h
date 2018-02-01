/*
 * MAST: Multidisciplinary-design Adaptation and Sensitivity Toolkit
 * Copyright (C) 2013-2018  Manav Bhatia
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

#ifndef __mast__eigenproblem_assembly__
#define __mast__eigenproblem_assembly__

// MAST includes
#include "base/assembly_base.h"

// libMesh includes
#include "libmesh/sparse_matrix.h"

namespace MAST {
    
    // Forward declerations
    class EigenproblemAssemblyElemOperations;
    
    
    /*!
     *   Assembles the system of equations for an eigenproblem of type
     *   \f$ {\bf A} {\bf x} = \lambda {\bf B} {\bf x} \f$. If the
     */
    
    class EigenproblemAssembly:
    public MAST::AssemblyBase {
        
    public:
        
        /*!
         *   constructor associates the eigen system with this assembly object
         */
        EigenproblemAssembly();
        
        /*!
         *   destructor resets the association with the eigen system
         *   from this assembly object
         */
        virtual ~EigenproblemAssembly();
        

        /*!
         *   attaches a system to this discipline, and vice-a-versa
         */
        virtual void
        attach_discipline_and_system(MAST::EigenproblemAssemblyElemOperations& elem_ops,
                                     MAST::PhysicsDisciplineBase& discipline,
                                     MAST::SystemInitialization& system);
        
        
        /*!
         *   Reattaches to the same system that was attached earlier.
         *
         *   This cannot be called if the clear_discipline_and_system() method
         *   has been called.
         */
        virtual void
        reattach_to_system();

        /*!
         *   clears association with a system to this discipline, and vice-a-versa
         */
        virtual void
        clear_discipline_and_system( );

        /*!
         *    @returns a reference to the A matrix of the EigenSystem
         */
        libMesh::SparseMatrix<Real>& A_matrix();
        
        
        /*!
         *    @returns a reference to the B matrix of the EigenSystem. Note
         *    that this matrix exists only for the Generalized Eigen
         *    problem.
         */
        libMesh::SparseMatrix<Real>& B_matrix();
        
        
        /*!
         *    assembles the matrices for eigenproblem depending on the analysis type
         */
        virtual void
        eigenproblem_assemble(libMesh::SparseMatrix<Real>* A,
                              libMesh::SparseMatrix<Real>* B);
        
        /**
         * Assembly function.  This function will be called
         * to assemble the sensitivity of eigenproblem matrices.
         * The method provides dA/dp and dB/dp for \par f parameter.
         *
         * If the routine is not able to provide sensitivity for this parameter,
         * then it should return false, and the system will attempt to use
         * finite differencing.
         */
        virtual bool
        eigenproblem_sensitivity_assemble (const MAST::FunctionBase& f,
                                           libMesh::SparseMatrix<Real>* sensitivity_A,
                                           libMesh::SparseMatrix<Real>* sensitivity_B);
        
        
        /*!
         *   if the eigenproblem is defined about a non-zero base solution,
         *   then this method provides the object with the base solution.
         *   The flag \par if_sens tells the method if \par sol
         *   is the sensitivity of the base solution for the current parameter
         *   being solved for
         */
        void set_base_solution(const libMesh::NumericVector<Real>& sol,
                               bool if_sens = false);

        
        /*!
         *   Clears the pointer to the solution.
         *   The flag \par if_sens tells the method to clear the pointer 
         *   the solution sensitivity vector.
         */
        void clear_base_solution(bool if_sens = false);

        
        /*!
         *   @returns true if a nonzero base solution is used to linearize the
         *   Eigen problem, false otherwise
         */
        bool if_linearized_about_nonzero_solution() const;
        
        
        /*!
         *   @returns a const reference to the base solution (or
         *   its sensitivity when \par if_sens is true) about which
         *   the Eigen problem was linearized.
         */
        const libMesh::NumericVector<Real>&
        base_sol(bool if_sens = false) const;
        

        /*!
         *   @returns a non-const reference to the base solution (or
         *   its sensitivity when \par if_sens is true) about which
         *   the Eigen problem was linearized.
         */
        libMesh::NumericVector<Real>& base_sol(bool if_sens = false);
        
        
    protected:
        
        /*!
         *    object to provide transient element operations
         */
        MAST::EigenproblemAssemblyElemOperations* _eigenproblem_elem_ops;

        /*!
         *   base solution about which this eigenproblem is defined. This
         *   vector stores the localized values necessary to perform element
         *   calculations.
         */
        const libMesh::NumericVector<Real> * _base_sol;
        
        /*!
         *   sensitivity of base solution may be needed for sensitivity
         *   analysis. This vector stores the localized values necessary to
         *   perform element calculations.
         */
        const libMesh::NumericVector<Real> * _base_sol_sensitivity;
        
    };
    
}

#endif // __mast__eigenproblem_assembly__
