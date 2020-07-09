//
// Created by walid arsalane on 6/23/20.
//

#ifndef __mast_foam_mesh__
#define __mast_foam_mesh__

// C++ includes
#include <map>

// MAST includes
#include "base/mast_data_types.h"
#include "base/constant_field_function.h"
#include "base/physics_discipline_base.h"

// libMesh includes
#include "libmesh/libmesh.h"
#include "libmesh/replicated_mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/elem.h"
#include "libmesh/boundary_info.h"

namespace MAST {


/*!
 *    builds the mesh for a foam structure
 */
class FoamMesh {
public:
    FoamMesh() { }
    
    ~FoamMesh() { }
    
    void
    init(const unsigned int divs_in_x,
         const unsigned int divs_in_y,
         const Real length,
         const Real width,
         const Real height,
         libMesh::ReplicatedMesh& mesh,
         const unsigned int n_refine);
    
protected:
    
    enum Component {
        PANEL,
        STIFFENER_X,
        STIFFENER_Y
    };
    
    void
    process_quad4_elems(int sub_id,
                        libMesh::Elem* e,
                        const std::vector<unsigned int>& node_ids,
                        std::map<std::pair<unsigned int, unsigned int>, libMesh::Elem*>& node_pair_to_elem_map,
                        std::vector<libMesh::Elem*>& elems,
                        std::map<const libMesh::Node*, libMesh::Node*>& node_map,
                        libMesh::MeshBase& mesh);
    
    void
    process_elems_in_plane(libMesh::Elem* e,
                           const std::vector<unsigned int>& node_ids_in_plane,
                           std::map<std::pair<unsigned int, unsigned int>, libMesh::Elem*>& node_pair_to_elem_map,
                           std::vector<libMesh::Elem*>& elems,
                           std::map<const libMesh::Node*, libMesh::Node*>& node_map,
                           libMesh::MeshBase& mesh);
    
    void
    add_elem_to_map(std::map<std::pair<unsigned int, unsigned int>, libMesh::Elem*>& node_pair_to_elem_map,
                    libMesh::Elem* e);
    
    libMesh::Node*
    new_node_for_ref_node(const libMesh::Node* ref_node,
                          std::map<const libMesh::Node*, libMesh::Node*>& node_map,
                          libMesh::MeshBase& mesh);
    
    bool
    check_if_nodes_connected(const std::map<std::pair<unsigned int, unsigned int>, libMesh::Elem*>& node_pair_to_elem_map,
                             unsigned int id1, unsigned int id2);
    
};
}


#endif // __mast_blade_stiffened_panel_mesh__
