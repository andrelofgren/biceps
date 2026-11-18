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

class IntervalMesh {
private:
    int _degree;
    int _nof_cells;
    int _nof_verts;

    int _nof_dofs;

    int _hl;

    std::vector<int> vertex_dof_inds;

    void compute_vertex_dof_inds();

public:
    Eigen::MatrixXd pmat;
    Eigen::MatrixXi cmat;
    Eigen::VectorXi dimat;

    IntervalMesh(double x0, double x1, int n_cells, int degree);
    IntervalMesh(const Eigen::MatrixXd &pmat, int degree);
    std::vector<int> extract_dof_inds(int id);
    std::vector<int> extract_vertex_dof_inds(int id);

    int nof_cells();
    int nof_verts();
    int nof_dofs();
    int dofs_per_cell();
    int degree();
};

