/*
 * MAST: Multidisciplinary-design Adaptation and Sensitivity Toolkit
 * Copyright (C) 2013-2019  Manav Bhatia
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
#include "examples/structural/beam_piston_theory_flutter/beam_piston_theory_flutter.h"



MAST::Examples::BeamPistonTheoryFlutterAnalysis::
BeamPistonTheoryFlutterAnalysis(const libMesh::Parallel::Communicator& comm_in):
MAST::Examples::StructuralExample1D(comm_in) {
    
}





MAST::Examples::BeamPistonTheoryFlutterAnalysis::~BeamPistonTheoryFlutterAnalysis() {
    
}


void
MAST::Examples::BeamPistonTheoryFlutterAnalysis::_init_loads() {
    
    this->_init_piston_theory_load();
}

