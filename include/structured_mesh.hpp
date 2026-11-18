/*
 * Copyright (C) 2025 André Löfgren
 *
 * This file is part of Biceps.
 *
 * Biceps is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Biceps is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Biceps. If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once
#include <vector>
#include <Eigen/Dense>

class StructuredMesh {
private:
    int _degree;
    int _nof_dofs;

    int _nof_verts;
    int _nof_cells;
    int _nof_edges;

    int _dofs_per_cell;
    int _edges_per_cell;
    int _dofs_per_edge;

    int _cell_type;

    int _nx;
    int _nz;
    int _hl;
    int _vl;

    std::vector<int> vertex_dof_inds;

    void assemble_pmat();
    void assemble_cmat();
    void assemble_emat();

    void assemble_cimat();
    void assemble_eimat();
    void assemble_dimat();

    void assemble_cimat_quadrilateral();
    void assemble_cimat_triangle_left();
    void assemble_cimat_triangle_right();

    void compute_vertex_dof_inds();

    void compute_edge_normals_and_tangents();
    void compute_vertex_normals_and_tangents();
    void compute_dof_normals_and_tangents();
    void compute_normals_and_tangents();

    void triangle_left_connectivity(int ci, int i, int j);
    void triangle_right_connectivity(int ci, int i, int j);
    void quadrilateral_connectivity(int ci, int i, int j);

    const int corner_dof_SW();
    const int corner_dof_SE();
    const int corner_dof_NE();
    const int corner_dof_NW();

    bool is_corner_dof(int vi);

public:
    Eigen::MatrixXd pmat_unit_box;

    Eigen::MatrixXd pmat;
    Eigen::MatrixXi cmat;
    Eigen::MatrixXi emat;

    Eigen::MatrixXi eimat;
    Eigen::VectorXi dimat;
    Eigen::VectorXi cimat;

    Eigen::MatrixXi edge_list;
    Eigen::MatrixXd edge_tangents;

    Eigen::MatrixXd edge_normals;

    Eigen::MatrixXd vertex_tangents;
    Eigen::MatrixXd vertex_normals;

    Eigen::MatrixXd dof_tangents;
    Eigen::MatrixXd dof_normals;

    Eigen::VectorXi v2s_map;
    Eigen::VectorXi v2b_map;

    StructuredMesh(int nx, int nz, int degree, int cell_type);

    void extrude_x(const Eigen::VectorXd &xvec_p1);
    void extrude_x(double x0, double x1);

    void extrude_z(
        std::function<double (double)> zb_expr,
        std::function<double (double)> zs_expr
    );
    void extrude_z(
        const Eigen::VectorXd &zb_vec_p1,
        const Eigen::VectorXd &zs_vec_p1
    );
    void extrude_z(const Eigen::VectorXd &zs_vec_p1);

    std::vector<int> extract_cell_inds(int id);
    std::vector<int> extract_edge_inds(int id);
    std::vector<int> extract_dof_inds(int id);

    std::vector<int> extract_vertex_dof_inds(int id);

    int nx();
    int nz();
    int hl();
    int vl();

    int degree();

    int nof_verts();
    int nof_dofs();
    int nof_cells();
    int nof_edges();

    int dofs_per_cell();
    int dofs_per_edge();

    int cell_type();

    double area();
};
