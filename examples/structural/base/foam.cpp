//
// Created by walid arsalane on 6/23/20.
//

// MAST includes
#include "foam.h"


// libMesh includes
#include "libmesh/libmesh.h"
#include "libmesh/replicated_mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/elem.h"
#include "libmesh/boundary_info.h"
#include "libmesh/mesh_refinement.h"

void
MAST::FoamMesh::init(const unsigned int divs_in_x,
                     const unsigned int divs_in_y,
                     const Real length,
                     const Real width,
                     const Real height,
                     libMesh::ReplicatedMesh& mesh,
                     const unsigned int n_refine) {
    
    
    libMesh::ReplicatedMesh ref_mesh(mesh.comm());

    libMesh::MeshTools::Generation::build_cube(ref_mesh, divs_in_x, divs_in_y, 1,
                                               0., length, 0., width, 0., height,
                                               libMesh::HEX27);
    
    // iterate over all elements and create line elements between centroid and
    // corner nodes
    libMesh::MeshBase::element_iterator
    it = ref_mesh.elements_begin(),
    end = ref_mesh.elements_end();
    
    std::vector<libMesh::Elem *> elems;
    
    std::vector<unsigned int>
    nodes_connected_to_center = {12, 13, 14, 15, 9, 11, 19, 17, 8, 10, 18, 16},
    right_plane = {13, 9, 14, 17},
    left_plane = {12, 11, 15, 19},
    front_plane = {12, 8, 13, 16},
    rear_plane = {15, 10, 14, 18},
    top_plane = {16, 17, 18, 19},
    bottom_plane = {8, 9, 10, 11},
    top_quads = {4, 16, 25, 19, 16, 5, 17, 25, 25, 17, 6, 18, 19, 25, 18, 7},
    bottom_quads = {0, 8, 20, 11, 8, 1, 9, 20, 11, 20, 10, 3, 20, 9, 2, 10};
    elems.reserve((nodes_connected_to_center.size() + right_plane.size() * 6 + 1) * ref_mesh.n_elem());
    
    std::map<std::pair<unsigned int, unsigned int>, libMesh::Elem *> node_pair_to_elem_map;
    std::map<const libMesh::Node *, libMesh::Node *> ref_to_new_node_map;
    
    for (; it != end; it++) {
        
        libMesh::Elem *e = *it;
        
        libMesh::Node
        *center_node = new_node_for_ref_node(e->node_ptr(26), ref_to_new_node_map, mesh),
        *nd;
        
        // connections to the center node
        for (unsigned int i = 0; i < nodes_connected_to_center.size(); i++) {
            
            nd = new_node_for_ref_node(e->node_ptr(nodes_connected_to_center[i]), ref_to_new_node_map, mesh);
            libMesh::Elem *edge = libMesh::Elem::build(libMesh::EDGE2).release();
            edge->set_node(0) = center_node;
            edge->set_node(1) = nd;
            edge->subdomain_id() = 3;
            elems.push_back(edge);
            add_elem_to_map(node_pair_to_elem_map, edge);
        }
        
        // connections in the right plane
        /*   HEX27:     7              18             6
         *              o--------------o--------------o
         *             /:             /              /|
         *            / :            /              / |
         *           /  :           /              /  |
         *        19/   :        25/            17/   |
         *         o--------------o--------------o    |
         *        /     :        /              /|    |
         *       /    15o       /    23o       / |  14o
         *      /       :      /              /  |   /|           zeta
         *    4/        :   16/             5/   |  / |            ^   eta (into page)
         *    o--------------o--------------o    | /  |            | /
         *    |         :    |   26         |    |/   |            |/
         *    |  24o    :    |    o         |  22o    |            o---> xi
         *    |         :    |       10     |   /|    |
         *    |        3o....|.........o....|../.|....o
         *    |        .     |              | /  |   / 2
         *    |       .    21|            13|/   |  /
         * 12 o--------------o--------------o    | /
         *    |     .        |              |    |/
         *    |  11o         | 20o          |    o
         *    |   .          |              |   / 9
         *    |  .           |              |  /
         *    | .            |              | /
         *    |.             |              |/
         *    o--------------o--------------o
         *    0              8              1
         */
        process_elems_in_plane(e, right_plane, node_pair_to_elem_map, elems, ref_to_new_node_map, mesh);
        process_elems_in_plane(e, left_plane, node_pair_to_elem_map, elems, ref_to_new_node_map, mesh);
        process_elems_in_plane(e, front_plane, node_pair_to_elem_map, elems, ref_to_new_node_map, mesh);
        process_elems_in_plane(e, rear_plane, node_pair_to_elem_map, elems, ref_to_new_node_map, mesh);
        process_elems_in_plane(e, top_plane, node_pair_to_elem_map, elems, ref_to_new_node_map, mesh);
        process_elems_in_plane(e, bottom_plane, node_pair_to_elem_map, elems, ref_to_new_node_map, mesh);
        
        // if the element has no neighbor on the bottom and top faces, then
        // we need to add shell elements at these locations
        /*
         *   HEX8: 7        6
         *         o--------z
         *        /:       /|         zeta
         *       / :      / |          ^   eta (into page)
         *    4 /  :   5 /  |          | /
         *     o--------o   |          |/
         *     |   o....|...o 2        o---> xi
         *     |  .3    |  /
         *     | .      | /
         *     |.       |/
         *     o--------o
         *     0        1
         *
         *   libMesh side numbering:
         *    {0, 3, 2, 1}, // Side 0 : back
         *    {0, 1, 5, 4}, // Side 1 : bottom
         *    {1, 2, 6, 5}, // Side 2 : right
         *    {2, 3, 7, 6}, // Side 3 : top
         *    {3, 0, 4, 7}, // Side 4 : left
         *    {4, 5, 6, 7}  // Side 5 : front
         */
        
        if (e->neighbor_ptr(0) == nullptr)
            process_quad4_elems(0, e, bottom_quads, node_pair_to_elem_map, elems, ref_to_new_node_map, mesh);
        
        if (e->neighbor_ptr(5) == nullptr)
            process_quad4_elems(1, e, top_quads, node_pair_to_elem_map, elems, ref_to_new_node_map, mesh);
    }
    
    for (unsigned int i = 0; i < elems.size(); i++)
        mesh.add_elem(elems[i]);
    
    
    
    
    // boundary_id
    int bc_ids = 0;
    libMesh::BoundaryInfo &mesh_bd_info = *mesh.boundary_info;
    
    libMesh::MeshBase::const_element_iterator
    el_it = mesh.elements_begin(),
    el_end = mesh.elements_end();
    
    std::map<libMesh::Node *, libMesh::Node *> old_to_new;
    libMesh::Node *old_node, *new_node;
    
    for (; el_it != el_end; el_it++) {
        
        libMesh::Elem *old_elem = *el_it;
        
        
        // add boundary condition tags for the panel boundary
        if (old_elem->subdomain_id() == 0 || old_elem->subdomain_id() == 1){
            
            
            if (abs((old_elem->point(0))(0) - 0.) < 1.e-8 &&
                abs((old_elem->point(3))(0) - 0.) < 1.e-8 )  { // check if side is on bnoudary x = 0
                if (old_elem->subdomain_id() == 0)
                    mesh_bd_info.add_side(old_elem, 3, 3);
                else if (old_elem->subdomain_id() == 1)
                    mesh_bd_info.add_side(old_elem, 3, 3+ 4);
            }
            if (abs((old_elem->point(1))(0) - length) < 1.e-8 &&
                abs((old_elem->point(2))(0) - length) < 1.e-8) { // check if side is on bnoudary x = length
                if (old_elem->subdomain_id() == 0)
                    mesh_bd_info.add_side(old_elem, 1, 1);
                else if (old_elem->subdomain_id() == 1)
                    mesh_bd_info.add_side(old_elem, 1, 1+ 4);
            }
            
            if (abs((old_elem->point(0))(1) - 0.) < 1.e-8 &&
                abs((old_elem->point(1))(1) - 0.) < 1.e-8 )  { // check if side is on bnoudary y = 0
                if (old_elem->subdomain_id() == 0)
                    mesh_bd_info.add_side(old_elem, 0, 0);
                else if (old_elem->subdomain_id() == 1)
                    mesh_bd_info.add_side(old_elem, 0, 0+ 4);
            }
            if (abs((old_elem->point(2))(1) - width) < 1.e-8 &&
                abs((old_elem->point(3))(1) - width) < 1.e-8) { // check if side is on bnoudary x = 0
                if (old_elem->subdomain_id() == 0)
                    mesh_bd_info.add_side(old_elem, 2, 2);
                else if (old_elem->subdomain_id() == 1)
                    mesh_bd_info.add_side(old_elem, 2, 2+ 4);
            }
        }
    }
    
    mesh.prepare_for_use();
    
    mesh.write("mesh_bfr.exo");

    if (n_refine) {
        
        libMesh::MeshRefinement mesh_refinement(mesh);
        mesh_refinement.uniformly_refine(n_refine);
    }
    
    mesh.write("mesh_aft.exo");
}


void
MAST::FoamMesh::process_quad4_elems
 (int sub_id,
  libMesh::Elem* e,
  const std::vector<unsigned int>& node_ids,
  std::map<std::pair<unsigned int, unsigned int>, libMesh::Elem*>& node_pair_to_elem_map,
  std::vector<libMesh::Elem*>& elems,
  std::map<const libMesh::Node*, libMesh::Node*>& node_map,
  libMesh::MeshBase& mesh) {
    
    libMesh::Node *node1, *node2, *node3, *node4;
    for (unsigned int i=0; i<4; i++) {
        
        node1 = new_node_for_ref_node(e->node_ptr(node_ids[i*4+0]), node_map, mesh);
        node2 = new_node_for_ref_node(e->node_ptr(node_ids[i*4+1]), node_map, mesh);
        node3 = new_node_for_ref_node(e->node_ptr(node_ids[i*4+2]), node_map, mesh);
        node4 = new_node_for_ref_node(e->node_ptr(node_ids[i*4+3]), node_map, mesh);
        
        libMesh::Elem *quad4 = libMesh::Elem::build(libMesh::QUAD4).release();
        quad4->set_node(0) = node1;
        quad4->set_node(1) = node2;
        quad4->set_node(2) = node3;
        quad4->set_node(3) = node4;
        quad4->subdomain_id() = sub_id;
        elems.push_back(quad4);
    }
}



void
MAST::FoamMesh::process_elems_in_plane
 (libMesh::Elem* e,
  const std::vector<unsigned int>& node_ids_in_plane,
  std::map<std::pair<unsigned int, unsigned int>, libMesh::Elem*>& node_pair_to_elem_map,
  std::vector<libMesh::Elem*>& elems,
  std::map<const libMesh::Node*, libMesh::Node*>& node_map,
  libMesh::MeshBase& mesh) {
    
    libMesh::Node *node1, *node2;
    for (unsigned int i=0; i<node_ids_in_plane.size(); i++) {
        
        node1 = new_node_for_ref_node(    e->node_ptr(node_ids_in_plane[i%4]), node_map, mesh);
        node2 = new_node_for_ref_node(e->node_ptr(node_ids_in_plane[(i+1)%4]), node_map, mesh);
        
        if (!check_if_nodes_connected(node_pair_to_elem_map, node1->id(), node2->id())) {
            
            libMesh::Elem *edge = libMesh::Elem::build(libMesh::EDGE2).release();
            edge->set_node(0) = node1;
            edge->set_node(1) = node2;
            edge->subdomain_id() = 3;
            elems.push_back(edge);
            add_elem_to_map(node_pair_to_elem_map, edge);
        }
    }
}

void
MAST::FoamMesh::add_elem_to_map
 (std::map<std::pair<unsigned int, unsigned int>, libMesh::Elem*>& node_pair_to_elem_map,
  libMesh::Elem* e) {
    
    unsigned int
    id1   = e->node_ptr(0)->id(),
    id2   = e->node_ptr(1)->id();
    
    libmesh_assert(!check_if_nodes_connected(node_pair_to_elem_map, id1, id2));
    
    std::pair<unsigned int, unsigned int> ids(std::min(id1, id2), std::max(id1, id2));
    
    node_pair_to_elem_map[ids] = e;
}


libMesh::Node*
MAST::FoamMesh::new_node_for_ref_node(const libMesh::Node* ref_node,
                                      std::map<const libMesh::Node*, libMesh::Node*>& node_map,
                                      libMesh::MeshBase& mesh) {
    
    std::map<const libMesh::Node*, libMesh::Node*>::iterator
    it  = node_map.find(ref_node);
    
    // create and add a new node if it does not exist
    if (it == node_map.end()) {
        
        libMesh::Node* node = new libMesh::Node(libMesh::Point(*ref_node));
        node->set_id() = mesh.max_node_id();
        mesh.add_node(node);
        it = node_map.insert(std::map<const libMesh::Node*, libMesh::Node*>::value_type(ref_node, node)).first;
    }
    
    return it->second;
}


bool
MAST::FoamMesh::check_if_nodes_connected
 (const std::map<std::pair<unsigned int, unsigned int>, libMesh::Elem*>& node_pair_to_elem_map,
  unsigned int id1, unsigned int id2) {
     
    std::pair<unsigned int, unsigned int> ids(std::min(id1, id2), std::max(id1, id2));
    return node_pair_to_elem_map.count(ids);
}
