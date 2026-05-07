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
#include <interval_mesh.hpp>
#include <enums.hpp>

IntervalMesh::IntervalMesh(double x0, double x1, int n_cells, int degree) :
    _degree(degree), _nof_cells(n_cells)
{
    if (n_cells < 1)
        throw std::invalid_argument("n_cells must be greater than 0");
    if (degree < 1)
        throw std::invalid_argument("degree must be greater than 0");
    else if (degree > 2)
        throw std::invalid_argument("degree greater than 2 is currently not supported");

    _hl = _degree * n_cells;

    pmat = Eigen::MatrixXd::Zero(nof_dofs(), 2);
    cmat = Eigen::MatrixXi::Zero(nof_cells(), dofs_per_cell());
    dimat = Eigen::VectorXi::Zero(nof_dofs());

    for (int di = 0; di < nof_dofs(); di++) {
        pmat(di, 0) = x0 + (x1 - x0) * di / (nof_dofs() - 1);

        if (fabs(pmat(di, 0) - x0) < 1e-10)
            dimat(di) = MESH1D::WEST_ID;
        else if (fabs(pmat(di, 0) - x1) < 1e-10)
            dimat(di) = MESH1D::EAST_ID;
        else
            dimat(di) = MESH1D::INTERIOR_ID;
    }

    for (int ci = 0; ci < nof_cells(); ci++)
        for (int k = 0; k < dofs_per_cell(); k++)
            cmat(ci, k) = ci * degree + k;

    compute_vertex_dof_inds();
}

void IntervalMesh::compute_vertex_dof_inds() {
    vertex_dof_inds.reserve(nof_verts());
    for (int vi = 0; vi < _hl + 1; vi += _degree)
        vertex_dof_inds.push_back(vi);
}

IntervalMesh::IntervalMesh(const Eigen::MatrixXd &pmat, int degree)
    : _degree(degree), pmat(pmat)
{
    _nof_cells = (pmat.rows() - 1) / degree;

    cmat = Eigen::MatrixXi::Zero(nof_cells(), dofs_per_cell());
    dimat = Eigen::VectorXi::Zero(nof_dofs());

    double x0 = pmat(0, 0);
    double x1 = pmat(Eigen::last, 0);

    for (int di = 0; di < nof_dofs(); di++) {
        if (fabs(pmat(di, 0) - x0) < 1e-10)
            dimat(di) = MESH1D::WEST_ID;
        else if (fabs(pmat(di, 0) - x1) < 1e-10)
            dimat(di) = MESH1D::EAST_ID;
        else
            dimat(di) = MESH1D::INTERIOR_ID;
    }

    for (int ci = 0; ci < nof_cells(); ci++)
        for (int k = 0; k < dofs_per_cell(); k++)
            cmat(ci, k) = ci * degree + k;
}

std::vector<int> IntervalMesh::extract_vertex_dof_inds(int id)
{
    std::vector<int> vinds;
    for (int vi : vertex_dof_inds)
        if (dimat(vi) & id)
            vinds.push_back(vi);
    return vinds;
}

std::vector<int> IntervalMesh::extract_dof_inds(int id)
{
    std::vector<int> dinds;
    for (int di = 0; di < dimat.rows(); di++)
        if (dimat(di) & id)
            dinds.push_back(di);
    return dinds;
}

int IntervalMesh::nof_cells()
{
    return _nof_cells;
}

int IntervalMesh::nof_verts()
{
    return _nof_cells + 1;
}

int IntervalMesh::nof_dofs()
{
    return degree() * nof_cells() + 1;
}

int IntervalMesh::dofs_per_cell()
{
    return _degree + 1;
}

int IntervalMesh::degree()
{
    return _degree;
}
