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

#include "catch.hpp"

#include "base/parameter.h"
#include "base/constant_field_function.h"
#include "base/boundary_condition_base.h"

TEST_CASE("boundary_condition_base",
          "[boundary_condition][base]")
{
    MAST::Parameter param1("dummy_param", 482.222);
    
    MAST::ConstantFieldFunction dummy_f("dummy", param1);
    
    // Ensure that this is a derived calss of MAST::FunctionSetBase
    // This is important becasue we assume that if the FunctionSetBase tests 
    // pass, then all methods that BoundaryConditionBase inherits from it
    // will automatically pass as well.  If this changes, we need to detect
    // this so we can rewrite these tests for MAST::BoundaryConditionBase.
    REQUIRE( std::is_base_of<MAST::FunctionSetBase, MAST::BoundaryConditionBase>::value );
    
    MAST::BoundaryConditionBase bc1(MAST::SURFACE_PRESSURE);   
    REQUIRE( bc1.type() == MAST::SURFACE_PRESSURE ); 
    
    MAST::BoundaryConditionBase bc2(MAST::POINT_LOAD);
    MAST::BoundaryConditionBase bc3(MAST::POINT_MOMENT);
    MAST::BoundaryConditionBase bc4(MAST::PISTON_THEORY);
    MAST::BoundaryConditionBase bc5(MAST::DIRICHLET);
    
    MAST::BoundaryConditionBase bc6(MAST::TEMPERATURE);
    REQUIRE( bc6.type() == MAST::TEMPERATURE );
    
    MAST::BoundaryConditionBase bc7(MAST::HEAT_FLUX);
    MAST::BoundaryConditionBase bc8(MAST::CONVECTION_HEAT_FLUX);
    MAST::BoundaryConditionBase bc9(MAST::SURFACE_RADIATION_HEAT_FLUX);
    MAST::BoundaryConditionBase bc10(MAST::HEAT_SOURCE);
    REQUIRE( bc10.type() == MAST::HEAT_SOURCE );
    
    MAST::BoundaryConditionBase bc11(MAST::NO_SLIP_WALL);
    MAST::BoundaryConditionBase bc12(MAST::SYMMETRY_WALL);
    MAST::BoundaryConditionBase bc13(MAST::SLIP_WALL);
    MAST::BoundaryConditionBase bc14(MAST::FAR_FIELD);
    MAST::BoundaryConditionBase bc15(MAST::EXHAUST);
    MAST::BoundaryConditionBase bc16(MAST::ISOTHERMAL);
    MAST::BoundaryConditionBase bc17(MAST::ADIABATIC);
    //MAST::BoundaryConditionBase bc18(MAST::DOF_DIRICHLET);
}
