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
#include <functional>
#include <tuple>
#include <set>
#include <boost/format.hpp>

#include <enums.hpp>
#include <fem_2d.hpp>


StructuredMesh::StructuredMesh(int nx, int nz, int degree, int cell_type): 
    _nx(nx), _nz(nz), _degree(degree), _cell_type(cell_type)
{
    if (nx < 1)
        throw std::invalid_argument("nx must be greater than 0");
    if (nz < 1)
        throw std::invalid_argument("nz must be greater than 0");

    if (degree < 1)
        throw std::invalid_argument("degree must be greater than 0");
    else if (degree > 2)
        throw std::invalid_argument("degree greater than 2 is currently not supported");

    if (cell_type == MESH2D::TRIANGLE_LEFT || cell_type == MESH2D::TRIANGLE_RIGHT) {
        _dofs_per_cell = (_degree+1)*(_degree+2) >> 1;
        _nof_cells = 2*_nx*_nz;
        _nof_edges = 3*_nx*_nz + _nx + _nz;
        _dofs_per_edge = degree+1;
        _edges_per_cell = 3;
    } else if (cell_type == MESH2D::QUADRILATERAL) {
        _dofs_per_cell = (_degree+1)*(_degree+1);
        _nof_cells = _nx*_nz;
        _dofs_per_edge = degree+1;
        _nof_edges = 2*_nx*_nz + _nx + _nz;
        _edges_per_cell = 4;
    } else {
        throw std::invalid_argument(
            (boost::format{"Unrecognized cell_type with id '%d'"} %cell_type).str()
        );
    }

    _hl = _degree*_nx;
    _vl = _degree*_nz;
    _nof_verts = (_nx+1)*(_nz+1);
    _nof_dofs = (_hl+1)*(_vl+1);

    pmat = Eigen::MatrixXd(_nof_dofs, 2);
    pmat_unit_box = Eigen::MatrixXd(_nof_dofs, 2);
    cmat = Eigen::MatrixXi(_nof_cells, _dofs_per_cell);
    dimat = Eigen::VectorXi(_nof_dofs);
    cimat = Eigen::VectorXi(_nof_cells);

    assemble_pmat();
    assemble_cmat();
    assemble_emat();
    assemble_cimat();
    assemble_dimat();
    assemble_eimat();
    compute_vertex_dof_inds();
    compute_normals_and_tangents();

    v2s_map = Eigen::VectorXi::Zero(_nof_dofs);
    v2b_map = Eigen::VectorXi::Zero(_nof_dofs);
    for (int vi = 0; vi < _nof_dofs; vi++) {
        v2b_map[vi] = vi % (_hl+1);
        v2s_map[vi] = v2b_map[vi] + _vl*(_hl+1);
    }
}

void StructuredMesh::assemble_pmat()
{
    Eigen::VectorXd xrange = Eigen::VectorXd::LinSpaced(_hl + 1, 0.0, 1.0);
    Eigen::VectorXd zrange = Eigen::VectorXd::LinSpaced(_vl + 1, 0.0, 1.0);

    int dof = 0;
    for (double z : zrange) {
        for (double x : xrange) {
            pmat(dof, 0) = x;
            pmat(dof, 1) = z;
            pmat_unit_box(dof, 0) = x;
            pmat_unit_box(dof, 1) = z;
            dof++;
        }
    }
}

void StructuredMesh::assemble_cmat()
{
    int ci = 0;  // Cell index
    for (int i = 0; i < _vl; i += _degree) {
        for (int j = 0; j < _hl; j += _degree) {
            if (_cell_type == MESH2D::TRIANGLE_RIGHT) {
                triangle_right_connectivity(ci, i, j);
                ci += 2;
            } else if (_cell_type == MESH2D::TRIANGLE_LEFT) {
                triangle_left_connectivity(ci, i, j);
                ci += 2;
            } else if (_cell_type == MESH2D::QUADRILATERAL) {
                quadrilateral_connectivity(ci, i, j);
                ci += 1;
            }
        }
    }
}

void StructuredMesh::assemble_emat()
{
    emat = Eigen::MatrixXi(_nof_edges, _dofs_per_edge);

    std::set<std::tuple<int, int>> visited_edges;
    std::tuple<int, int> edge;

    int first_ind, last_ind, ei = 0;
    for (int ci = 0; ci < cmat.rows(); ++ci) {
        Eigen::VectorXi cell = cmat.row(ci);
        for (int i = 0; i < _edges_per_cell; ++i) {
            first_ind = i * _degree;
            if (i != _edges_per_cell - 1)
                last_ind = (i + 1) * _degree;
            else
                last_ind = 0;

            if (cell(first_ind) < cell(last_ind))
                edge = std::tuple<int, int>(cell(first_ind), cell(last_ind));
            else
                edge = std::tuple<int, int>(cell(last_ind), cell(first_ind));

            if (visited_edges.find(edge) == visited_edges.end()) {
                visited_edges.insert(edge);
                for (int j = 0; j < _dofs_per_edge - 1; ++j) {
                    emat(ei, j) = cell(first_ind + j);
                }
                emat(ei, _dofs_per_edge - 1) = cell(last_ind);
                ei++;
            }
        }
    }
}

inline void StructuredMesh::triangle_right_connectivity(int ci, int i, int j)
{
    if (_degree == 1) {
        cmat(ci, 0) = (_hl + 1) * i + j;
        cmat(ci, 1) = (_hl + 1) * i + j + 1;
        cmat(ci, 2) = (_hl + 1) * (i + 1) + j;

        cmat(ci + 1, 0) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci + 1, 1) = (_hl + 1) * (i + 1) + j;
        cmat(ci + 1, 2) = (_hl + 1) * i + (j + 1);
    } else if (_degree == 2) {
        cmat(ci, 0) = (_hl + 1) * i + j;
        cmat(ci, 1) = (_hl + 1) * i + j + 1;
        cmat(ci, 2) = (_hl + 1) * i + j + 2;
        cmat(ci, 3) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci, 4) = (_hl + 1) * (i + 2) + j;
        cmat(ci, 5) = (_hl + 1) * (i + 1) + j;

        cmat(ci + 1, 0) = (_hl + 1) * (i + 2) + j + 2;
        cmat(ci + 1, 1) = (_hl + 1) * (i + 2) + j + 1;
        cmat(ci + 1, 2) = (_hl + 1) * (i + 2) + j;
        cmat(ci + 1, 3) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci + 1, 4) = (_hl + 1) * i + j + 2;
        cmat(ci + 1, 5) = (_hl + 1) * (i + 1) + j + 2;
    }
}

inline void StructuredMesh::triangle_left_connectivity(int ci, int i, int j)
{
    if (_degree == 1) {
        cmat(ci, 0) = (_hl + 1) * (i + 1) + j;
        cmat(ci, 1) = (_hl + 1) * i + j;
        cmat(ci, 2) = (_hl + 1) * (i + 1) + j + 1;

        cmat(ci + 1, 0) = (_hl + 1) * i + j + 1;
        cmat(ci + 1, 1) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci + 1, 2) = (_hl + 1) * i + j;
    } else if (_degree == 2) {
        cmat(ci, 0) = (_hl + 1) * (i + 2) + j;
        cmat(ci, 1) = (_hl + 1) * (i + 1) + j;
        cmat(ci, 2) = (_hl + 1) * i + j;
        cmat(ci, 3) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci, 4) = (_hl + 1) * (i + 2) + j + 2;
        cmat(ci, 5) = (_hl + 1) * (i + 2) + j + 1;

        cmat(ci + 1, 0) = (_hl + 1) * i + j + 2;
        cmat(ci + 1, 1) = (_hl + 1) * (i + 1) + j + 2;
        cmat(ci + 1, 2) = (_hl + 1) * (i + 2) + j + 2;
        cmat(ci + 1, 3) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci + 1, 4) = (_hl + 1) * i + j;
        cmat(ci + 1, 5) = (_hl + 1) * i + j + 1;
    }
}

inline void StructuredMesh::quadrilateral_connectivity(int ci, int i, int j)
{
    if (_degree == 1) {
        cmat(ci, 0) = (_hl + 1) * i + j;
        cmat(ci, 1) = (_hl + 1) * i + j + 1;
        cmat(ci, 2) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci, 3) = (_hl + 1) * (i + 1) + j;
    } else if (_degree == 2) {
        cmat(ci, 0) = (_hl + 1) * i + j;
        cmat(ci, 1) = (_hl + 1) * i + j + 1;
        cmat(ci, 2) = (_hl + 1) * i + j + 2;
        cmat(ci, 3) = (_hl + 1) * (i + 1) + j + 2;
        cmat(ci, 4) = (_hl + 1) * (i + 2) + j + 2;
        cmat(ci, 5) = (_hl + 1) * (i + 2) + j + 1;
        cmat(ci, 6) = (_hl + 1) * (i + 2) + j;
        cmat(ci, 7) = (_hl + 1) * (i + 1) + j;
        cmat(ci, 8) = (_hl + 1) * (i + 1) + j + 1;
    }
}

void StructuredMesh::assemble_cimat() 
{
    if (_cell_type == MESH2D::QUADRILATERAL)
        assemble_cimat_quadrilateral();
    else if (_cell_type == MESH2D::TRIANGLE_LEFT)
        assemble_cimat_triangle_left();
    else if (_cell_type == MESH2D::TRIANGLE_RIGHT)
        assemble_cimat_triangle_right();
}

void StructuredMesh::assemble_cimat_triangle_left() 
{
    // Placeholder (not implemented yet)
}

void StructuredMesh::assemble_cimat_triangle_right() 
{
    // Placeholder (not implemented yet())
}

void StructuredMesh::assemble_cimat_quadrilateral()
{
    int i, j, ci;

    cimat(0) = MESH2D::SOUTH_WEST_ID;
    cimat(_nx - 1) = MESH2D::SOUTH_EAST_ID;
    cimat(_nof_cells - _nx) = MESH2D::NORTH_WEST_ID;
    cimat(_nof_cells - 1) = MESH2D::NORTH_EAST_ID;

    i = _nz - 1;
    for (j = 1; j < _nx - 1; ++j) {
        ci = i * _nx + j;
        cimat(ci) = MESH2D::NORTH_ID;
    }

    j = 0;
    for (i = 1; i < _nz - 1; ++i) {
        ci = i * _nx + j;
        cimat(ci) = MESH2D::WEST_ID;
    }

    j = _nx - 1;
    for (i = 1; i < _nz - 1; ++i) {
        ci = i * _nx + j;
        cimat(ci) = MESH2D::EAST_ID;
    }

    i = 0;
    for (j = 1; j < _nx - 1; ++j) {
        ci = i * (_nx - 1) + j;
        cimat(ci) = MESH2D::SOUTH_ID;
    }

    for (i = 1; i < _nz - 1; ++i) {
        for (j = 1; j < _nx - 1; ++j) {
            ci = i * _nx + j;
            cimat(ci) = MESH2D::INTERIOR_ID;
        }
    }
}

void StructuredMesh::assemble_dimat()
{
    double domain_width = pmat(Eigen::placeholders::last, 0) - pmat(0, 0);

    double xtol = domain_width * 1e-3 / _hl;
    double ztol = 1e-3 / _vl;

    for (int i = 0; i < _nof_dofs; ++i) {

        if (pmat(i, 1) > 1.0 - ztol && pmat(i, 0) < xtol)
            dimat(i) = MESH2D::NORTH_WEST_ID;

        else if (pmat(i, 1) < ztol && pmat(i, 0) < xtol)
            dimat(i) = MESH2D::SOUTH_WEST_ID;

        else if (pmat(i, 1) < ztol && pmat(i, 0) > 1.0 - xtol)
            dimat(i) = MESH2D::SOUTH_EAST_ID;

        else if (pmat(i, 1) > 1.0 - ztol && pmat(i, 0) > 1.0 - xtol)
            dimat(i) = MESH2D::NORTH_EAST_ID;

        else if (pmat(i, 1) > 1.0 - ztol)
            dimat(i) = MESH2D::NORTH_ID;

        else if (pmat(i, 1) < ztol)
            dimat(i) = MESH2D::SOUTH_ID;

        else if (pmat(i, 0) < xtol)
            dimat(i) = MESH2D::WEST_ID;

        else if (pmat(i, 0) > 1.0 - xtol)
            dimat(i) = MESH2D::EAST_ID;

        else
            dimat(i) = MESH2D::INTERIOR_ID;
    }
}

void StructuredMesh::assemble_eimat()
{
    int dpe = _dofs_per_edge;

    eimat = Eigen::MatrixXi::Zero(_nof_edges, dpe + 1);

    double domain_width = pmat(Eigen::placeholders::last, 0) - pmat(0, 0);

    double xtol = domain_width * 1e-3 / _hl;
    double ztol = 1e-3 / _vl;

    for (int ei = 0; ei < emat.rows(); ++ei) {
        double x1 = pmat(emat(ei, 0), 0);
        double z1 = pmat(emat(ei, 0), 1);
        double x2 = pmat(emat(ei, 1), 0);
        double z2 = pmat(emat(ei, 1), 1);

        if (z1 > 1.0 - ztol && z2 > 1.0 - ztol)
            eimat(ei, 0) = MESH2D::NORTH_ID;
        else if (z1 < ztol && z2 < ztol)
            eimat(ei, 0) = MESH2D::SOUTH_ID;
        else if (x1 < xtol && x2 < xtol)
            eimat(ei, 0) = MESH2D::WEST_ID;
        else if (x1 > 1.0 - xtol && x2 > 1.0 - xtol)
            eimat(ei, 0) = MESH2D::EAST_ID;
        else
            eimat(ei, 0) = MESH2D::INTERIOR_ID;
    }

    std::set<std::tuple<int, int>> visited_edges;
    std::tuple<int, int> edge;

    int v1, v2;
    int ei = 0;
    for (int ci = 0; ci < cmat.rows(); ++ci) {
        Eigen::VectorXi cell = cmat.row(ci);

        for (int i = 0; i < _edges_per_cell; ++i) {
            v1 = cell(i * _degree);
            if (i != _edges_per_cell - 1)
                v2 = cell((i + 1) * _degree);
            else
                v2 = cell(0);

            if (v1 < v2)
                edge = std::tuple<int, int>(v1, v2);
            else
                edge = std::tuple<int, int>(v2, v1);

            if (visited_edges.find(edge) == visited_edges.end()) {
                eimat(ei, 1) = v1;
                eimat(ei, dpe) = v2;
                if (_degree == 2)
                    eimat(ei, 2) = cell(i * _degree + 1);
                visited_edges.insert(edge);
                ei++;
            }
        }
    }
}

const int inline StructuredMesh::corner_dof_SW()
{
    return 0;
}

const int inline StructuredMesh::corner_dof_SE()
{
    return _hl;
}

const int inline StructuredMesh::corner_dof_NE()
{
    return (_hl + 1)*(_vl + 1) - 1;
}

const int inline StructuredMesh::corner_dof_NW()
{
    return (_hl + 1)*_vl;
}

bool inline StructuredMesh::is_corner_dof(int vi)
{
    return (
        vi == corner_dof_SW() || vi == corner_dof_SE() ||
        vi == corner_dof_NE() || vi == corner_dof_NW()
    );
}

void StructuredMesh::compute_edge_normals_and_tangents()
{
    edge_tangents = Eigen::MatrixXd::Zero(_nof_edges, 2);
    edge_normals = Eigen::MatrixXd::Zero(_nof_edges, 2);

    std::vector<int> boundary_einds = extract_edge_inds(MESH2D::BOUNDARY_ID);

    for (int ei : boundary_einds) {
        int v1 = eimat(ei, 1);
        int v2 = eimat(ei, _dofs_per_edge);

        edge_tangents(ei, Eigen::placeholders::all) = pmat.row(v2) - pmat.row(v1);
        edge_tangents(ei, Eigen::placeholders::all) /= edge_tangents.row(ei).norm();

        edge_normals(ei, 0) = edge_tangents(ei, 1);
        edge_normals(ei, 1) = -edge_tangents(ei, 0);
    }
}

void StructuredMesh::compute_dof_normals_and_tangents()
{
    // Initialize the dof_tangents and dof_normals matrices with zeros
    dof_tangents = Eigen::MatrixXd::Zero(_nof_dofs, 2);
    dof_normals = Eigen::MatrixXd::Zero(_nof_dofs, 2);

    std::vector<int> boundary_einds = extract_edge_inds(MESH2D::BOUNDARY_ID);
    for (int i = 0; i < boundary_einds.size(); ++i) {
        int ei = boundary_einds[i];
        int v1 = eimat(ei, 1);
        int v2 = eimat(ei, _dofs_per_edge);

        dof_tangents(v1, Eigen::placeholders::all) += edge_tangents.row(ei);
        dof_normals(v1, Eigen::placeholders::all) += edge_normals.row(ei);
        dof_tangents(v2, Eigen::placeholders::all) += edge_tangents.row(ei);
        dof_normals(v2, Eigen::placeholders::all) += edge_normals.row(ei);
    }

    std::vector<int> boundary_vertex_dofs = extract_vertex_dof_inds(MESH2D::BOUNDARY_ID);
    for (int di: boundary_vertex_dofs) {
        dof_tangents(di, Eigen::placeholders::all) /= dof_tangents.row(di).norm();
        dof_normals(di, Eigen::placeholders::all) /= dof_normals.row(di).norm();
    }

    if (_degree == 2)
        for (int ei: boundary_einds) {
            int ve = eimat(ei, 2);
            dof_tangents(ve, Eigen::placeholders::all) = edge_tangents.row(ei);
            dof_normals(ve, Eigen::placeholders::all) = edge_normals.row(ei);
        }

    dof_tangents(
        corner_dof_SW(), Eigen::placeholders::all
    ) = dof_tangents(corner_dof_SW()+1, Eigen::placeholders::all);
    dof_normals(
        corner_dof_SW(), Eigen::placeholders::all
    ) = dof_normals(corner_dof_SW()+1, Eigen::placeholders::all);

    dof_tangents(
        corner_dof_SE(), Eigen::placeholders::all
    ) = dof_tangents(corner_dof_SE()-1, Eigen::placeholders::all);
    dof_normals(
        corner_dof_SE(), Eigen::placeholders::all
    ) = dof_normals(corner_dof_SE()-1, Eigen::placeholders::all);

    dof_tangents(
        corner_dof_NE(), Eigen::placeholders::all
    ) = dof_tangents(corner_dof_NE()-1, Eigen::placeholders::all);
    dof_normals(
        corner_dof_NE(), Eigen::placeholders::all
    ) = dof_normals(corner_dof_NE()-1, Eigen::placeholders::all);

    dof_tangents(
        corner_dof_NW(), Eigen::placeholders::all
    ) = dof_tangents(corner_dof_NW()+1, Eigen::placeholders::all);
    dof_normals(
        corner_dof_NW(), Eigen::placeholders::all
    ) = dof_normals(corner_dof_NW()+1, Eigen::placeholders::all);
}

void StructuredMesh::compute_vertex_normals_and_tangents()
{
    vertex_tangents = dof_tangents(vertex_dof_inds, Eigen::placeholders::all);
    vertex_normals = dof_normals(vertex_dof_inds, Eigen::placeholders::all);
}

void StructuredMesh::compute_normals_and_tangents()
{
    compute_edge_normals_and_tangents();
    compute_dof_normals_and_tangents();
    compute_vertex_normals_and_tangents();
}

std::vector<int> StructuredMesh::extract_edge_inds(int id)
{
    std::vector<int> einds;
    for (int ei = 0; ei < eimat.rows(); ++ei)
        if (eimat(ei, 0) & id)
            einds.push_back(ei);
    return einds;
}

std::vector<int> StructuredMesh::extract_cell_inds(int id)
{
    std::vector<int> cell_inds;
    for (int ci = 0; ci < cimat.rows(); ++ci)
        if (cimat(ci) & id)
            cell_inds.push_back(ci);
    return cell_inds;
}

std::vector<int> StructuredMesh::extract_vertex_dof_inds(int id)
{
    std::vector<int> dinds;
    for (int vi : vertex_dof_inds)
        if (dimat(vi) & id)
            dinds.push_back(vi);

    return dinds;
}

std::vector<int> StructuredMesh::extract_dof_inds(int id)
{
    std::vector<int> dinds;
    for (int di = 0; di < dimat.rows(); ++di)
        if (dimat(di) & id)
            dinds.push_back(di);
    return dinds;
}

void StructuredMesh::extrude_x(double x0, double x1)
{
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(pmat.rows());
    pmat(Eigen::placeholders::all, 0) = (ones - pmat(Eigen::placeholders::all, 0)) * x0 + pmat(Eigen::placeholders::all, 0) * x1;
    compute_normals_and_tangents();
}

void StructuredMesh::extrude_x(const Eigen::VectorXd &xvec_p1)
{
    Eigen::VectorXd x_vec;

    if (_degree == 1)
        x_vec = xvec_p1;
    else if (_degree == 2) {
        x_vec = Eigen::VectorXd::Zero(_hl + 1);
        x_vec(Eigen::seq(0, Eigen::placeholders::last, 2)) = xvec_p1;

        for (int i = 1; i < x_vec.size() - 1; i += 2)
            x_vec(i) = 0.5 * (x_vec(i - 1) + x_vec(i + 1));
    }

    int dof = 0;
    for (int i = 0; i < _vl + 1; ++i) {
        for (int j = 0; j < _hl + 1; ++j) {
            pmat(dof, 0) = x_vec(j);
            dof++;
        }
    }

    // Recompute boundary normals and tangents after extrusion
    compute_normals_and_tangents();
}

void StructuredMesh::extrude_z(
    std::function<double (double)> zb_expr,
    std::function<double (double)> zs_expr
)
{
    std::vector<int> surf_vert_inds = extract_vertex_dof_inds(MESH2D::SURFACE_ID);

    Eigen::VectorXd xvec_p1 = pmat(surf_vert_inds, 0);

    Eigen::VectorXd zb_vec_p1 = xvec_p1.unaryExpr(zb_expr);
    Eigen::VectorXd zs_vec_p1 = xvec_p1.unaryExpr(zs_expr);

    extrude_z(zb_vec_p1, zs_vec_p1);
}

void StructuredMesh::extrude_z(const Eigen::VectorXd &zs_vec_p1)
{
    std::vector<int> bed_vert_inds = extract_vertex_dof_inds(MESH2D::BED_ID);
    Eigen::VectorXd zb_vec_p1 = pmat(bed_vert_inds, 1);

    extrude_z(zb_vec_p1, zs_vec_p1);
}

void StructuredMesh::extrude_z(
    const Eigen::VectorXd &zb_vec_p1, const Eigen::VectorXd &zs_vec_p1
)
{
    Eigen::VectorXd zb_vec_p2 = Eigen::VectorXd::Zero(_hl+1);
    Eigen::VectorXd zs_vec_p2 = Eigen::VectorXd::Zero(_hl+1);

    // Precalculate displacement of surface and bedrock
    int k = 0;
    for (int i = 0; i < _hl+1; i+=_degree) {
        zb_vec_p2[i] = zb_vec_p1[k];
        zs_vec_p2[i] = zs_vec_p1[k];
        k++;
    }
    // Set the correct z position of all cell corner nodes
    for (int i = 0; i < _vl+1; i+=_degree) {
        for (int j = 0; j < _hl+1; j+=_degree) {
            int vi = i*(_hl+1) + j;
            double z = pmat_unit_box(vi, 1);
            pmat(vi, 1) = (1.0-z)*zb_vec_p2(v2b_map(vi)) + z*zs_vec_p2(v2b_map(vi));
        }
    }

    if (_degree == 2) {
        // set the correct z position of all cell interior nodes
        for (int i = 1; i < _vl; i+=_degree) {
            for (int j = 1; j < _hl; j+=_degree) {
                int vi_NW = (i+1)*(_hl+1) + j-1;
                int vi_NE = (i+1)*(_hl+1) + j+1;
                int vi_SW = (i-1)*(_hl+1) + j-1;
                int vi_SE = (i-1)*(_hl+1) + j+1;
                int vi = i*(_hl+1) + j;
                if (_cell_type == MESH2D::TRIANGLE_LEFT)
                    pmat(vi, 1) = 0.5*(pmat(vi_NE, 1) + pmat(vi_SW, 1));
                else if (_cell_type == MESH2D::TRIANGLE_RIGHT)
                    pmat(vi, 1) = 0.5*(pmat(vi_NW, 1) + pmat(vi_SE, 1));
                else if (_cell_type == MESH2D::QUADRILATERAL)
                    pmat(vi, 1) = 0.25*(pmat(vi_NW, 1) + pmat(vi_NE, 1) + pmat(vi_SW, 1) + pmat(vi_SE, 1));
            }
        }

        // set the correct z position of all horizontal edge nodes
        for (int i = 0; i < _vl+1; i+=_degree) {
            for (int j = 1; j < _hl; j+=_degree) {
                int vi_W = i*(_hl+1) + j-1;
                int vi_E = i*(_hl+1) + j+1;
                int vi = i*(_hl+1) + j;	
                pmat(vi, 1) = 0.5*(pmat(vi_W, 1) + pmat(vi_E, 1));
            }
        }

        // Set the correct z position of all vertical edge nodes
        for (int i = 1; i < _vl; i+=_degree) {
            for (int j = 0; j < _hl+1; j+=_degree) {
                int vi_N = (i+1)*(_hl+1) + j;
                int vi_S = (i-1)*(_hl+1) + j;
                int vi = i*(_hl+1) + j;
                pmat(vi, 1) = 0.5*(pmat(vi_N, 1) + pmat(vi_S, 1));
            }
        }
    }

    // Recompute boundary normals and tangents
    compute_normals_and_tangents();
}

void StructuredMesh::compute_vertex_dof_inds() {
    int n_verts = (_nx+1)*(_nz+1);
    vertex_dof_inds.reserve(n_verts);

    for (int i = 0; i < _vl+1; i+=_degree) {
        for (int j = 0; j < _hl+1; j+=_degree) {
            vertex_dof_inds.push_back(i*(_hl+1) + j);
        }
    }
}

int StructuredMesh::nx()
{
    return _nx;
}

int StructuredMesh::nz()
{
    return _nz;
}

int StructuredMesh::hl()
{
    return _hl;
}

int StructuredMesh::vl()
{
    return _vl;
}

int StructuredMesh::nof_cells()
{
    return _nof_cells;
}

int StructuredMesh::dofs_per_cell()
{
    return _dofs_per_cell;
}

int StructuredMesh::nof_edges()
{
    return _nof_edges;
}

int StructuredMesh::dofs_per_edge()
{
    return _dofs_per_edge;
}

int StructuredMesh::cell_type()
{
    return _cell_type;
}

int StructuredMesh::nof_verts()
{
    return _nof_verts;
}

int StructuredMesh::nof_dofs()
{
    return _nof_dofs;
}

int StructuredMesh::degree()
{
    return _degree;
}

double StructuredMesh::area()
{
    Eigen::MatrixXd node_coords, qpoints_rs, qpoints_xz, phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element;

    FEM2D::gauss_legendre_quadrature(
        1, cell_type(), qpoints_rs, qweights
    );

    FEM2D::lagrange_basis(
        degree(), cell_type(), qpoints_rs, phi_rs, dphi_rs
    );

    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2 * qpoints_rs.rows(), dofs_per_cell());

    double _area = 0.0;

    for (int k = 0; k < cmat.rows(); ++k) {
        element = cmat.row(k);
        node_coords = pmat(element, Eigen::placeholders::all);

        FEM2D::map_from_reference_cell(
            degree(), cell_type(), node_coords, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        for (int q = 0; q < qpoints_rs.rows(); ++q) {
            _area += fabs(detJ_rs(q)) * qweights(q);
        }
    }

    return _area;
}
