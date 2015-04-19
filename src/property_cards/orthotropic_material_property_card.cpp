/*
 * MAST: Multidisciplinary-design Adaptation and Sensitivity Toolkit
 * Copyright (C) 2013-2014  Manav Bhatia
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
#include "property_cards/orthotropic_material_property_card.h"
#include "base/field_function_base.h"


namespace MAST {
    namespace OrthotropicMaterialProperty {
        
        
        class StiffnessMatrix1D: public MAST::FieldFunction<RealMatrixX> {
        public:
            
            StiffnessMatrix1D( MAST::FieldFunction<Real>* E,
                              MAST::FieldFunction<Real>* nu);
            
            StiffnessMatrix1D(const MAST::OrthotropicMaterialProperty::StiffnessMatrix1D& f):
            MAST::FieldFunction<RealMatrixX>(f),
            _E(f._E->clone().release()),
            _nu(f._nu->clone().release()){
                _functions.insert(_E->master());
                _functions.insert(_nu->master());
            }
            
            /*!
             *   @returns a clone of the function
             */
            virtual std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
            clone() const {
                return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
                (new MAST::OrthotropicMaterialProperty::StiffnessMatrix1D(*this));
            }
            
            virtual ~StiffnessMatrix1D() {
                delete _E;
                delete _nu;
            }
            
            virtual void operator() (const libMesh::Point& p,
                                     const Real t,
                                     RealMatrixX& m) const;
            
            virtual void derivative(const MAST::DerivativeType d,
                                    const MAST::FunctionBase& f,
                                    const libMesh::Point& p,
                                    const Real t,
                                    RealMatrixX& m) const;
            
        protected:
            
            MAST::FieldFunction<Real>* _E;
            MAST::FieldFunction<Real>* _nu;
        };
        
        
        
        class TransverseShearStiffnessMatrix: public MAST::FieldFunction<RealMatrixX> {
        public:
            TransverseShearStiffnessMatrix( MAST::FieldFunction<Real>* E,
                                           MAST::FieldFunction<Real>* nu,
                                           MAST::FieldFunction<Real>* kappa);
            
            TransverseShearStiffnessMatrix(const MAST::OrthotropicMaterialProperty::TransverseShearStiffnessMatrix& f):
            MAST::FieldFunction<RealMatrixX>(f),
            _E(f._E->clone().release()),
            _nu(f._nu->clone().release()),
            _kappa(f._kappa->clone().release()) {
                _functions.insert(_E->master());
                _functions.insert(_nu->master());
                _functions.insert(_kappa->master());
            }
            
            /*!
             *   @returns a clone of the function
             */
            virtual std::auto_ptr<MAST::FieldFunction<RealMatrixX> > clone() const  {
                return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
                (new MAST::OrthotropicMaterialProperty::TransverseShearStiffnessMatrix(*this));
            }
            
            virtual ~TransverseShearStiffnessMatrix() {
                delete _E;
                delete _nu;
                delete _kappa;
            }
            
            virtual void operator() (const libMesh::Point& p,
                                     const Real t,
                                     RealMatrixX& m) const;
            
            virtual void derivative(const MAST::DerivativeType d,
                                    const MAST::FunctionBase& f,
                                    const libMesh::Point& p,
                                    const Real t,
                                    RealMatrixX& m) const;
            
        protected:
            
            MAST::FieldFunction<Real>* _E;
            MAST::FieldFunction<Real>* _nu;
            MAST::FieldFunction<Real>* _kappa;
        };
        
        
        class StiffnessMatrix2D: public MAST::FieldFunction<RealMatrixX> {
        public:
            StiffnessMatrix2D(MAST::FieldFunction<Real>* E11,
                              MAST::FieldFunction<Real>* E22,
                              MAST::FieldFunction<Real>* nu12,
                              MAST::FieldFunction<Real>* G12,
                              bool plane_stress);
            
            StiffnessMatrix2D(const MAST::OrthotropicMaterialProperty::StiffnessMatrix2D& f):
            MAST::FieldFunction<RealMatrixX>(f),
            _E11(f._E11->clone().release()),
            _E22(f._E22->clone().release()),
            _nu12(f._nu12->clone().release()),
            _G12(f._G12->clone().release()),
            _plane_stress(f._plane_stress) {
                _functions.insert(_E11->master());
                _functions.insert(_E22->master());
                _functions.insert(_nu12->master());
                _functions.insert(_G12->master());
            }
            
            /*!
             *   @returns a clone of the function
             */
            virtual std::auto_ptr<MAST::FieldFunction<RealMatrixX> > clone() const {
                return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
                (new MAST::OrthotropicMaterialProperty::StiffnessMatrix2D(*this));
            }
            
            virtual ~StiffnessMatrix2D() {
                delete _E11;
                delete _E22;
                delete _nu12;
                delete _G12;
            }
            
            
            virtual void operator() (const libMesh::Point& p,
                                     const Real t,
                                     RealMatrixX& m) const;
            
            virtual void derivative(const MAST::DerivativeType d,
                                    const MAST::FunctionBase& f,
                                    const libMesh::Point& p,
                                    const Real t,
                                    RealMatrixX& m) const;
            
        protected:
            
            MAST::FieldFunction<Real> *_E11, *_E22;
            MAST::FieldFunction<Real>* _nu12;
            MAST::FieldFunction<Real>* _G12;
            bool _plane_stress;
        };
        
        
        
        class StiffnessMatrix3D: public MAST::FieldFunction<RealMatrixX> {
        public:
            StiffnessMatrix3D(MAST::FieldFunction<Real>* E11,
                              MAST::FieldFunction<Real>* E22,
                              MAST::FieldFunction<Real>* E33,
                              MAST::FieldFunction<Real>* nu12,
                              MAST::FieldFunction<Real>* nu13,
                              MAST::FieldFunction<Real>* nu23,
                              MAST::FieldFunction<Real>* G12,
                              MAST::FieldFunction<Real>* G13,
                              MAST::FieldFunction<Real>* G23);
            
            StiffnessMatrix3D(const MAST::OrthotropicMaterialProperty::StiffnessMatrix3D &f):
            MAST::FieldFunction<RealMatrixX>(f),
            _E11(f._E11->clone().release()),
            _E22(f._E22->clone().release()),
            _E33(f._E33->clone().release()),
            _nu12(f._nu12->clone().release()),
            _nu13(f._nu13->clone().release()),
            _nu23(f._nu23->clone().release()) {
                _functions.insert(_E11->master());
                _functions.insert(_E22->master());
                _functions.insert(_E33->master());
                _functions.insert(_nu12->master());
                _functions.insert(_nu13->master());
                _functions.insert(_nu23->master());
                _functions.insert(_G12->master());
                _functions.insert(_G13->master());
                _functions.insert(_G23->master());
            }
            
            /*!
             *   @returns a clone of the function
             */
            virtual std::auto_ptr<MAST::FieldFunction<RealMatrixX> > clone() const {
                return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
                (new MAST::OrthotropicMaterialProperty::StiffnessMatrix3D(*this));
            }
            
            virtual ~StiffnessMatrix3D() {
                delete _E11;
                delete _E22;
                delete _E33;
                delete _nu12;
                delete _nu13;
                delete _nu23;
                delete _G12;
                delete _G13;
                delete _G23;
            }
            
            virtual void operator() (const libMesh::Point& p,
                                     const Real t,
                                     RealMatrixX& m) const;
            
            virtual void derivative(const MAST::DerivativeType d,
                                    const MAST::FunctionBase& f,
                                    const libMesh::Point& p,
                                    const Real t,
                                    RealMatrixX& m) const;
            
        protected:
            
            MAST::FieldFunction<Real> *_E11, *_E22, *_E33;
            MAST::FieldFunction<Real> *_nu12, *_nu13, *_nu23;
            MAST::FieldFunction<Real> *_G12, *_G13, *_G23;

        };
        
        
        
        class ThermalExpansionMatrix: public MAST::FieldFunction<RealMatrixX> {
        public:
            
            ThermalExpansionMatrix(unsigned int dim,
                                   MAST::FieldFunction<Real>* alpha11,
                                   MAST::FieldFunction<Real>* alpha22 = NULL,
                                   MAST::FieldFunction<Real>* alpha33 = NULL):
            MAST::FieldFunction<RealMatrixX>("ThermalExpansionMatrix"),
            _dim(dim),
            _alpha(dim) {

                switch (dim-1) {
                    case 2:
                        libmesh_assert_msg(alpha33, "alpha33 cannot be NULL for 3D matrix");
                        _alpha[2] = alpha33;
                    case 1:
                        libmesh_assert_msg(alpha22, "alpha22 cannot be NULL for 2D/3D matrix");
                        _alpha[1] = alpha22;
                    case 0:
                        libmesh_assert_msg(alpha11, "alpha11 cannot be NULL for 1D/2D/3D matrix");
                        _alpha[0] = alpha11;
                        break;
                    default:
                        libmesh_error();
                }
                
                for (unsigned int i=0; i<dim; i++)
                    _functions.insert(_alpha[i]->master());
            }
            
            ThermalExpansionMatrix(const MAST::OrthotropicMaterialProperty::ThermalExpansionMatrix& f):
            MAST::FieldFunction<RealMatrixX>(f),
            _dim(f._dim),
            _alpha(f._dim) {
                
                for (unsigned int i=0; i<f._dim; i++) {
                    _alpha[i] = f._alpha[i]->clone().release();
                    _functions.insert(_alpha[i]->master());
                }
            }
            
            /*!
             *   @returns a clone of the function
             */
            virtual std::auto_ptr<MAST::FieldFunction<RealMatrixX> > clone() const {
                return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
                (new MAST::OrthotropicMaterialProperty::ThermalExpansionMatrix(*this));
            }
            
            virtual ~ThermalExpansionMatrix() {
                
                for (unsigned int i=0; i<_dim; i++)
                    delete _alpha[i];
            }
            
            virtual void operator() (const libMesh::Point& p,
                                     const Real t,
                                     RealMatrixX& m) const;
            
            virtual void derivative(const MAST::DerivativeType d,
                                    const MAST::FunctionBase& f,
                                    const libMesh::Point& p,
                                    const Real t,
                                    RealMatrixX& m) const;
            
        protected:
            
            const unsigned int _dim;
            
            std::vector<MAST::FieldFunction<Real>*> _alpha;
        };
        
        
        
        
        class ThermalConductanceMatrix:
        public MAST::FieldFunction<RealMatrixX> {
        public:
            
            ThermalConductanceMatrix(unsigned int dim,
                                     MAST::FieldFunction<Real>* k11,
                                     MAST::FieldFunction<Real>* k22 = NULL,
                                     MAST::FieldFunction<Real>* k33 = NULL):
            MAST::FieldFunction<RealMatrixX>("ThermalConductanceMatrix"),
            _dim(dim),
            _k(dim) {

                switch (dim-1) {
                    case 2:
                        libmesh_assert_msg(k33, "k33 cannot be NULL for 3D matrix");
                        _k[2] = k33;
                    case 1:
                        libmesh_assert_msg(k22, "k22 cannot be NULL for 2D/3D matrix");
                        _k[1] = k22;
                    case 0:
                        libmesh_assert_msg(k11, "k11 cannot be NULL for 1D/2D/3D matrix");
                        _k[0] = k11;
                        break;
                    default:
                        libmesh_error();
                }

                for (unsigned int i=0; i<dim; i++)
                    _functions.insert(_k[i]->master());
            }
            
            ThermalConductanceMatrix(const MAST::OrthotropicMaterialProperty::ThermalConductanceMatrix& f):
            MAST::FieldFunction<RealMatrixX>(f),
            _dim(f._dim),
            _k(f._dim) {
                
                for (unsigned int i=0; i<f._dim; i++) {
                    _k[i] = f._k[i]->clone().release();
                    _functions.insert(_k[i]->master());
                }
            }
            
            /*!
             *   @returns a clone of the function
             */
            virtual std::auto_ptr<MAST::FieldFunction<RealMatrixX> > clone() const {
                return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
                (new MAST::OrthotropicMaterialProperty::ThermalConductanceMatrix(*this));
            }
            
            virtual ~ThermalConductanceMatrix() {

                for (unsigned int i=0; i<_dim; i++)
                    delete _k[i];
                
            }
            
            virtual void operator() (const libMesh::Point& p,
                                     const Real t,
                                     RealMatrixX& m) const;
            
            virtual void derivative(const MAST::DerivativeType d,
                                    const MAST::FunctionBase& f,
                                    const libMesh::Point& p,
                                    const Real t,
                                    RealMatrixX& m) const;
            
        protected:
            
            const unsigned int _dim;
            
            std::vector<MAST::FieldFunction<Real>*> _k;
        };
        
        
        
        class ThermalCapacitanceMatrix:
        public MAST::FieldFunction<RealMatrixX> {
        public:
            
            ThermalCapacitanceMatrix(unsigned int dim,
                                     MAST::FieldFunction<Real>* rho,
                                     MAST::FieldFunction<Real>* cp):
            MAST::FieldFunction<RealMatrixX>("ThermalCapacitanceMatrix"),
            _dim(dim),
            _rho(rho),
            _cp(cp) {
                
                _functions.insert(_rho->master());
                _functions.insert(_cp->master());
            }
            
            ThermalCapacitanceMatrix(const MAST::OrthotropicMaterialProperty::ThermalCapacitanceMatrix& f):
            MAST::FieldFunction<RealMatrixX>(f),
            _dim(f._dim),
            _rho(f._rho->clone().release()),
            _cp(f._cp->clone().release()) {
                
                _functions.insert(_rho->master());
                _functions.insert(_cp->master());
            }
            
            /*!
             *   @returns a clone of the function
             */
            virtual std::auto_ptr<MAST::FieldFunction<RealMatrixX> > clone() const {
                return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
                (new MAST::OrthotropicMaterialProperty::ThermalCapacitanceMatrix(*this));
            }
            
            virtual ~ThermalCapacitanceMatrix() {
                
                delete _rho;
                delete _cp;
            }
            
            virtual void operator() (const libMesh::Point& p,
                                     const Real t,
                                     RealMatrixX& m) const;
            
            virtual void derivative(const MAST::DerivativeType d,
                                    const MAST::FunctionBase& f,
                                    const libMesh::Point& p,
                                    const Real t,
                                    RealMatrixX& m) const;
            
        protected:
            
            const unsigned int _dim;
            
            MAST::FieldFunction<Real>* _rho;
            
            MAST::FieldFunction<Real>* _cp;
        };
        
        
    }
}



MAST::OrthotropicMaterialProperty::
StiffnessMatrix1D::StiffnessMatrix1D(MAST::FieldFunction<Real>* E,
                                     MAST::FieldFunction<Real>* nu ):
MAST::FieldFunction<RealMatrixX>("StiffnessMatrix1D"),
_E(E),
_nu(nu)
{
    _functions.insert(E->master());
    _functions.insert(nu->master());
}




void
MAST::OrthotropicMaterialProperty::
StiffnessMatrix1D::operator() (const libMesh::Point& p,
                               const Real t,
                               RealMatrixX& m) const {
    m  = RealMatrixX::Zero(2,2);
    Real E, nu, G;
    (*_E)(p, t, E); (*_nu)(p, t, nu);
    G = E/2./(1.+nu);
    m(0,0) = E;
    m(1,1) = G;
}


void
MAST::OrthotropicMaterialProperty::
StiffnessMatrix1D::derivative(const MAST::DerivativeType d,
                              const MAST::FunctionBase &f,
                              const libMesh::Point &p,
                              const Real t,
                              RealMatrixX &m) const {
    
    
    RealMatrixX dm;
    m = RealMatrixX::Zero(2,2); dm = RealMatrixX::Zero(2,2);
    Real E, nu, dEdf, dnudf;
    (*_E)(p, t, E);     _E->derivative(d, f, p, t, dEdf);
    (*_nu)(p, t, nu);  _nu->derivative(d, f, p, t, dnudf);
    
    // parM/parE * parE/parf
    dm(0,0) = 1.;
    dm(1,1) = 1./2./(1.+nu);
    m += dEdf * dm;
    
    
    // parM/parnu * parnu/parf
    dm(0,0) = 0.;
    dm(1,1) = -E/2./pow(1.+nu,2);
    m+= dnudf*dm;
}


MAST::OrthotropicMaterialProperty::
TransverseShearStiffnessMatrix::TransverseShearStiffnessMatrix(MAST::FieldFunction<Real> * E,
                                                               MAST::FieldFunction<Real> * nu,
                                                               MAST::FieldFunction<Real> * kappa):
MAST::FieldFunction<RealMatrixX>("TransverseShearStiffnessMatrix"),
_E(E),
_nu(nu),
_kappa(kappa)
{
    _functions.insert(E->master());
    _functions.insert(nu->master());
    _functions.insert(kappa->master());
}



void
MAST::OrthotropicMaterialProperty::
TransverseShearStiffnessMatrix::operator() (const libMesh::Point& p,
                                            const Real t,
                                            RealMatrixX& m) const {
    m = RealMatrixX::Zero(2,2);
    Real E, nu, kappa, G;
    (*_E)(p, t, E); (*_nu)(p, t, nu); (*_kappa)(p, t, kappa);
    G = E/2./(1.+nu);
    m(0,0) = G*kappa;
    m(1,1) = m(0,0);
}



void
MAST::OrthotropicMaterialProperty::
TransverseShearStiffnessMatrix::derivative(const MAST::DerivativeType d,
                                           const MAST::FunctionBase& f,
                                           const libMesh::Point& p,
                                           const Real t,
                                           RealMatrixX& m) const {
    RealMatrixX dm;
    m = RealMatrixX::Zero(2,2); dm = RealMatrixX::Zero(2, 2);
    Real E, nu, kappa, dEdf, dnudf, dkappadf, G;
    (*_E)    (p, t, E);         _E->derivative(d, f, p, t, dEdf);
    (*_nu)   (p, t, nu);       _nu->derivative(d, f, p, t, dnudf);
    (*_kappa)(p, t, kappa); _kappa->derivative(d, f, p, t, dkappadf);
    G = E/2./(1.+nu);
    
    
    // parM/parE * parE/parf
    dm(0,0) = 1./2./(1.+nu)*kappa;
    dm(1,1) = dm(0,0);
    m += dEdf * dm;
    
    
    // parM/parnu * parnu/parf
    dm(0,0) = -E/2./pow(1.+nu,2)*kappa;
    dm(1,1) = dm(0,0);
    m+= dnudf*dm;
    
    // parM/parnu * parkappa/parf
    
    dm(0,0) = G; dm(1,1) = G;
    dm += dkappadf*dm;
}




MAST::OrthotropicMaterialProperty::
StiffnessMatrix2D::StiffnessMatrix2D(MAST::FieldFunction<Real> * E11,
                                     MAST::FieldFunction<Real> * E22,
                                     MAST::FieldFunction<Real> * nu12,
                                     MAST::FieldFunction<Real> * G12,
                                     bool plane_stress ):
MAST::FieldFunction<RealMatrixX>("StiffnessMatrix2D"),
_E11(E11),
_E22(E22),
_nu12(nu12),
_G12(G12),
_plane_stress(plane_stress)
{
    _functions.insert(E11->master());
    _functions.insert(E22->master());
    _functions.insert(nu12->master());
    _functions.insert(G12->master());
}




void
MAST::OrthotropicMaterialProperty::
StiffnessMatrix2D::operator() (const libMesh::Point& p,
                               const Real t,
                               RealMatrixX& m) const {
    libmesh_assert(_plane_stress); // currently only implemented for plane stress
    m = RealMatrixX::Zero(3,3);
    Real E11, E22, nu12, nu21, G12, D;
    (*_E11) (p, t,  E11);
    (*_E22) (p, t,  E22);
    (*_nu12)(p, t, nu12);
    (*_G12) (p, t,  G12);
    nu21 = nu12 * E22/E11;
    D = (1. - nu12*nu21);

    m(0,0)  =      E11;
    m(0,1)  = nu21*E11;
    
    m(1,0)  = nu12*E22;
    m(1,1)  =      E22;
    
    m.topLeftCorner(2, 2) /= D;
    
    m(2,2)  = G12;
}




void
MAST::OrthotropicMaterialProperty::
StiffnessMatrix2D::derivative (const MAST::DerivativeType d,
                               const MAST::FunctionBase& f,
                               const libMesh::Point& p,
                               const Real t,
                               RealMatrixX& m) const {
    libmesh_assert(_plane_stress); // currently only implemented for plane stress
    RealMatrixX dm;
    m = RealMatrixX::Zero(3,3); dm = RealMatrixX::Zero(3, 3);
    Real E11, E22, nu12, nu21, dE11df, dE22df, dnu12df, dnu21df, D, dDdf, dG12df;
    (*_E11)  (p, t,  E11);   _E11->derivative(d, f, p, t, dE11df);
    (*_E22)  (p, t,  E22);   _E22->derivative(d, f, p, t, dE22df);
    (*_nu12) (p, t, nu12);  _nu12->derivative(d, f, p, t, dnu12df);
    _G12->derivative(d, f, p, t, dG12df);
    
    nu21    = nu12 * E22/E11;
    dnu21df = dnu12df * E22/E11 + nu12 * dE22df/E11 - nu12 * E22/E11/E11*dE11df;
    D    = (1. - nu12*nu21);
    dDdf = (- dnu12df*nu21 - nu12*dnu21df);
    
    m(0,0)  =      E11;
    m(0,1)  = nu21*E11;
    
    m(1,0)  = nu12*E22;
    m(1,1)  =      E22;
    
    m.topLeftCorner(2, 2) *= -dDdf/D/D;
    m(2,2)  = dG12df;
    
    dm(0,0)  =      dE11df;
    dm(0,1)  = nu21*dE11df + dnu21df*E11;
    
    dm(1,0)  = nu12*dE22df + dnu12df*E22;
    dm(1,1)  =      dE22df;

    m.topLeftCorner(2, 2) += dm.topLeftCorner(3, 3)/D;
}



MAST::OrthotropicMaterialProperty::
StiffnessMatrix3D::StiffnessMatrix3D(MAST::FieldFunction<Real> * E11,
                                     MAST::FieldFunction<Real> * E22,
                                     MAST::FieldFunction<Real> * E33,
                                     MAST::FieldFunction<Real> * nu12,
                                     MAST::FieldFunction<Real> * nu13,
                                     MAST::FieldFunction<Real> * nu23,
                                     MAST::FieldFunction<Real> * G12,
                                     MAST::FieldFunction<Real> * G13,
                                     MAST::FieldFunction<Real> * G23):
MAST::FieldFunction<RealMatrixX>("StiffnessMatrix3D"),
_E11(E11),
_E22(E22),
_E33(E33),
_nu12(nu12),
_nu13(nu13),
_nu23(nu23),
_G12(G12),
_G13(G13),
_G23(G23)
{
    _functions.insert(E11->master());
    _functions.insert(E22->master());
    _functions.insert(E33->master());
    _functions.insert(nu12->master());
    _functions.insert(nu13->master());
    _functions.insert(nu23->master());
    _functions.insert(G12->master());
    _functions.insert(G13->master());
    _functions.insert(G23->master());
}





void
MAST::OrthotropicMaterialProperty::
StiffnessMatrix3D::operator() (const libMesh::Point& p,
                               const Real t,
                               RealMatrixX& m) const {
    m = RealMatrixX::Zero(6,6);
    Real E11, E22, E33, G12, G13, G23, nu12, nu13, nu23, nu21, nu31, nu32, D;
    (*_E11) (p, t,  E11);
    (*_E22) (p, t,  E22);
    (*_E33) (p, t,  E33);
    (*_nu12)(p, t, nu12);
    (*_nu13)(p, t, nu13);
    (*_nu23)(p, t, nu23);
    (*_G12) (p, t,  G12);
    (*_G13) (p, t,  G13);
    (*_G23) (p, t,  G23);
    nu21 = nu12 * E22/E11;
    nu31 = nu13 * E33/E11;
    nu32 = nu23 * E33/E22;
    D = 1.- nu12*nu21 - nu13*nu31 - nu23*nu32 - nu12*nu23*nu31 - nu13*nu21*nu32;
    

    m(0,0) = (  1. - nu23*nu32)*E11;
    m(0,1) = (nu21 + nu23*nu31)*E11;
    m(0,2) = (nu31 + nu21*nu32)*E11;

    m(1,0) = (nu12 + nu13*nu32)*E22;
    m(1,1) = (  1. - nu13*nu31)*E22;
    m(1,2) = (nu32 + nu12*nu31)*E22;

    m(2,0) = (nu13 + nu12*nu23)*E33;
    m(2,1) = (nu23 + nu13*nu21)*E33;
    m(2,2) = (  1. - nu12*nu21)*E33;

    m.topLeftCorner(3, 3) /= D;
    m(3,3) = G12;
    m(4,4) = G23;
    m(5,5) = G13;
}




void
MAST::OrthotropicMaterialProperty::
StiffnessMatrix3D::derivative (const MAST::DerivativeType d,
                               const MAST::FunctionBase& f,
                               const libMesh::Point& p,
                               const Real t,
                               RealMatrixX& m) const {
    RealMatrixX dm;
    m = RealMatrixX::Zero(6,6); dm = RealMatrixX::Zero(6,6);
    Real
    E11, dE11df,
    E22, dE22df,
    E33, dE33df,
    dG12df,
    dG13df,
    dG23df,
    nu12, dnu12df,
    nu13, dnu13df,
    nu23, dnu23df,
    nu21, dnu21df,
    nu31, dnu31df,
    nu32, dnu32df,
    D, dDdf;
    (*_E11)  (p, t,  E11);   _E11->derivative(d, f, p, t, dE11df);
    (*_E22)  (p, t,  E22);   _E22->derivative(d, f, p, t, dE22df);
    (*_E33)  (p, t,  E33);   _E33->derivative(d, f, p, t, dE33df);
    (*_nu12) (p, t, nu12); _nu12->derivative(d, f, p, t, dnu12df);
    (*_nu13) (p, t, nu13); _nu13->derivative(d, f, p, t, dnu13df);
    (*_nu23) (p, t, nu23); _nu23->derivative(d, f, p, t, dnu23df);
    _G12->derivative(d, f, p, t,  dG12df);
    _G13->derivative(d, f, p, t,  dG13df);
    _G23->derivative(d, f, p, t,  dG23df);
    nu21    = nu12 * E22/E11;
    dnu21df = dnu12df * E22/E11 + nu12 * dE22df/E11 - nu12 * E22/E11/E11*dE11df;
    nu31    = nu13 * E33/E11;
    dnu31df = dnu13df * E33/E11 + nu13 * dE33df/E11 - nu13 * E33/E11/E11*dE11df;
    nu32    = nu23 * E33/E22;
    dnu32df = dnu23df * E33/E22 + nu23 * dE33df/E22 - nu23 * E33/E22/E22*dE22df;
    D    = 1.- nu12*nu21 - nu13*nu31 - nu23*nu32 - nu12*nu23*nu31 - nu13*nu21*nu32;
    dDdf =
    - dnu12df*nu21 - nu12*dnu21df
    - dnu13df*nu31 - nu13*dnu31df
    - dnu23df*nu32 - nu23*dnu32df
    - dnu12df*nu23*nu31
    - nu12*dnu23df*nu31
    - nu12*nu23*dnu31df
    - dnu13df*nu21*nu32
    - nu13*dnu21df*nu32
    - nu13*nu21*dnu32df;
    
    m(0,0) = (  1. - nu23*nu32)*E11;
    m(0,1) = (nu21 + nu23*nu31)*E11;
    m(0,2) = (nu31 + nu21*nu32)*E11;
    
    m(1,0) = (nu12 + nu13*nu32)*E22;
    m(1,1) = (  1. - nu13*nu31)*E22;
    m(1,2) = (nu32 + nu12*nu31)*E22;
    
    m(2,0) = (nu13 + nu12*nu23)*E33;
    m(2,1) = (nu23 + nu13*nu21)*E33;
    m(2,2) = (  1. - nu12*nu21)*E33;
    m *= -dDdf/D/D;

    m(3,3) = dG12df;
    m(4,4) = dG23df;
    m(5,5) = dG13df;


    dm(0,0) = (- dnu23df*nu32 - nu23*dnu32df)*E11 + (  1. - nu23*nu32)*dE11df;
    dm(0,1) = (dnu21df + dnu23df*nu31 + nu23*dnu31df)*E11 + (nu21 + nu23*nu31)*dE11df;
    dm(0,2) = (dnu31df + dnu21df*nu32 + nu21*dnu32df)*E11 + (nu31 + nu21*nu32)*dE11df;
    
    dm(1,0) = (dnu12df + dnu13df*nu32 + nu13*dnu32df)*E22 + (nu12 + nu13*nu32)*dE22df;
    dm(1,1) = (- dnu13df*nu31 - nu13*dnu31df)*E22 + (  1. - nu13*nu31)*dE22df;
    dm(1,2) = (dnu32df + dnu12df*nu31 + nu12*dnu31df)*E22 + (nu32 + nu12*nu31)*dE22df;
    
    dm(2,0) = (dnu13df + dnu12df*nu23 + nu12*dnu23df)*E33 + (nu13 + nu12*nu23)*dE33df;
    dm(2,1) = (dnu23df + dnu13df*nu21 + nu13*dnu21df)*E33 + (nu23 + nu13*nu21)*dE33df;
    dm(2,2) = (- dnu12df*nu21 - nu12*dnu21df)*E33 + (  1. - nu12*nu21)*dE33df;

    m += dm/D;
}




void
MAST::OrthotropicMaterialProperty::ThermalExpansionMatrix::
operator() (const libMesh::Point& p,
            const Real t,
            RealMatrixX& m) const {
    

    Real alpha;
    switch (_dim) {
        case 1:
            m.setZero(2, 1);
            break;
            
        case 2:
            m.setZero(3, 1);
            break;
            
        case 3:
            m.setZero(6, 1);
            break;
    }
    
    for (unsigned int i=0; i<_dim; i++) {
        (*_alpha[i])  (p, t, alpha);
        m(i,0) = alpha;
    }
}





void
MAST::OrthotropicMaterialProperty::ThermalExpansionMatrix::
derivative (const MAST::DerivativeType d,
            const MAST::FunctionBase& f,
            const libMesh::Point& p,
            const Real t,
            RealMatrixX& m) const {
    
    
    
    Real alpha;
    switch (_dim) {
        case 1:
            m.setZero(2, 1);
            break;
            
        case 2:
            m.setZero(3, 1);
            break;
            
        case 3:
            m.setZero(6, 1);
            break;
    }
    
    for (unsigned int i=0; i<_dim; i++) {
        _alpha[i]->derivative(d, f, p, t, alpha);
        m(i,0) = alpha;
    }
}





void
MAST::OrthotropicMaterialProperty::ThermalCapacitanceMatrix::
operator() (const libMesh::Point& p,
            const Real t,
            RealMatrixX& m) const {
    
    Real cp, rho;
    (*_cp)  (p, t, cp);
    (*_rho) (p, t, rho);
    
    m.setZero(1,1);
    
    m(0,0) = cp*rho;
}





void
MAST::OrthotropicMaterialProperty::ThermalCapacitanceMatrix::
derivative (const MAST::DerivativeType d,
            const MAST::FunctionBase& f,
            const libMesh::Point& p,
            const Real t,
            RealMatrixX& m) const {
    
    
    Real cp, dcp, rho, drho;
    (*_cp)  (p, t, cp);    _cp->derivative(d, f, p, t, dcp);
    (*_rho) (p, t, rho);  _rho->derivative(d, f, p, t, drho);
    
    m.setZero(1,1);
    
    m(0,0) = dcp*rho + cp*drho;
}




void
MAST::OrthotropicMaterialProperty::ThermalConductanceMatrix::
operator() (const libMesh::Point& p,
            const Real t,
            RealMatrixX& m) const {
    
    Real k;
    m.setZero(_dim, _dim);
    for (unsigned int i=0; i<_dim; i++) {
        (*_k[i])  (p, t, k);
        m(i,i) = k;
    }
    
}





void
MAST::OrthotropicMaterialProperty::ThermalConductanceMatrix::
derivative (const MAST::DerivativeType d,
            const MAST::FunctionBase& f,
            const libMesh::Point& p,
            const Real t,
            RealMatrixX& m) const {
    
    Real k;
    m.setZero(_dim, _dim);
    for (unsigned int i=0; i<_dim; i++) {
        _k[i]->derivative(d, f, p, t, k);
        m(i,i) = k;
    }
}





std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
MAST::OrthotropicMaterialPropertyCard::stiffness_matrix(const unsigned int dim,
                                                      const bool plane_stress) {
    
    MAST::FieldFunction<RealMatrixX> *rval = NULL;
    
    switch (dim) {
        case 1:
            rval = new MAST::OrthotropicMaterialProperty::StiffnessMatrix1D
            (this->get<MAST::FieldFunction<Real> >("E").clone().release(),
             this->get<MAST::FieldFunction<Real> >("nu").clone().release());
            break;
            
        case 2:
            rval = new MAST::OrthotropicMaterialProperty::StiffnessMatrix2D
            (this->get<MAST::FieldFunction<Real> >("E11").clone().release(),
             this->get<MAST::FieldFunction<Real> >("E22").clone().release(),
             this->get<MAST::FieldFunction<Real> >("nu12").clone().release(),
             this->get<MAST::FieldFunction<Real> >("G12").clone().release(),
             plane_stress);
            break;
            
        case 3:
            rval = new MAST::OrthotropicMaterialProperty::StiffnessMatrix3D
            (this->get<MAST::FieldFunction<Real> >("E11").clone().release(),
             this->get<MAST::FieldFunction<Real> >("E22").clone().release(),
             this->get<MAST::FieldFunction<Real> >("E33").clone().release(),
             this->get<MAST::FieldFunction<Real> >("nu12").clone().release(),
             this->get<MAST::FieldFunction<Real> >("nu13").clone().release(),
             this->get<MAST::FieldFunction<Real> >("nu23").clone().release(),
             this->get<MAST::FieldFunction<Real> >("G12").clone().release(),
             this->get<MAST::FieldFunction<Real> >("G13").clone().release(),
             this->get<MAST::FieldFunction<Real> >("G23").clone().release());
            break;
            
        default:
            // should not get here
            libmesh_error();
    }
    
    return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >(rval);
}




std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
MAST::OrthotropicMaterialPropertyCard::damping_matrix(const unsigned int dim) {
    
    MAST::FieldFunction<RealMatrixX> *rval = NULL;
    
    // make sure that this is not null
    libmesh_assert(rval);
    
    return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >(rval);
}




std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
MAST::OrthotropicMaterialPropertyCard::inertia_matrix(const unsigned int dim) {
    
    MAST::FieldFunction<RealMatrixX> *rval = NULL;
    
    // make sure that this is not null
    libmesh_assert(rval);
    
    return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >(rval);
}




std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
MAST::OrthotropicMaterialPropertyCard::thermal_expansion_matrix(const unsigned int dim) {

    MAST::FieldFunction<RealMatrixX> *rval = NULL;
    
    switch (dim) {
        case 1:
            rval = new MAST::OrthotropicMaterialProperty::ThermalExpansionMatrix
            (dim,
             this->get<MAST::FieldFunction<Real> >("alpha11_expansion").clone().release());
            break;
            
        case 2:
            rval = new MAST::OrthotropicMaterialProperty::ThermalExpansionMatrix
            (dim,
             this->get<MAST::FieldFunction<Real> >("alpha11_expansion").clone().release(),
             this->get<MAST::FieldFunction<Real> >("alpha22_expansion").clone().release());
            break;
            
        case 3:
            rval = new MAST::OrthotropicMaterialProperty::ThermalExpansionMatrix
            (dim,
             this->get<MAST::FieldFunction<Real> >("alpha11_expansion").clone().release(),
             this->get<MAST::FieldFunction<Real> >("alpha22_expansion").clone().release(),
             this->get<MAST::FieldFunction<Real> >("alpha33_expansion").clone().release());
            break;
            
        default:
            libmesh_error();
            break;
    }
    
    return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >(rval);
}




std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
MAST::OrthotropicMaterialPropertyCard::transverse_shear_stiffness_matrix() {
    
    MAST::FieldFunction<RealMatrixX> *rval =
    new MAST::OrthotropicMaterialProperty::TransverseShearStiffnessMatrix
    (this->get<MAST::FieldFunction<Real> >("E11").clone().release(),
     this->get<MAST::FieldFunction<Real> >("nu12").clone().release(),
     this->get<MAST::FieldFunction<Real> >("kappa").clone().release());
    
    return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >(rval);
}



std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
MAST::OrthotropicMaterialPropertyCard::capacitance_matrix(const unsigned int dim) {
    
    MAST::FieldFunction<RealMatrixX> *rval =
    new MAST::OrthotropicMaterialProperty::ThermalCapacitanceMatrix
    (dim,
     this->get<MAST::FieldFunction<Real> >("rho").clone().release(),
     this->get<MAST::FieldFunction<Real> >("cp").clone().release());
    
    return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >(rval);
}





std::auto_ptr<MAST::FieldFunction<RealMatrixX> >
MAST::OrthotropicMaterialPropertyCard::conductance_matrix(const unsigned int dim) {
    
    MAST::FieldFunction<RealMatrixX> *rval = NULL;
    
    switch (dim) {
        case 1:
            rval = new MAST::OrthotropicMaterialProperty::ThermalConductanceMatrix
            (dim,
             this->get<MAST::FieldFunction<Real> >("k11_th").clone().release());
            break;

        case 2:
            rval = new MAST::OrthotropicMaterialProperty::ThermalConductanceMatrix
            (dim,
             this->get<MAST::FieldFunction<Real> >("k11_th").clone().release(),
             this->get<MAST::FieldFunction<Real> >("k22_th").clone().release());
            break;

        case 3:
            rval = new MAST::OrthotropicMaterialProperty::ThermalConductanceMatrix
            (dim,
             this->get<MAST::FieldFunction<Real> >("k11_th").clone().release(),
             this->get<MAST::FieldFunction<Real> >("k22_th").clone().release(),
             this->get<MAST::FieldFunction<Real> >("k33_th").clone().release());
            break;

        default:
            libmesh_error();
            break;
    }
    
    return std::auto_ptr<MAST::FieldFunction<RealMatrixX> >(rval);
}



