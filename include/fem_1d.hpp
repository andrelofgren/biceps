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
#define EIGEN_SPARSEMATRIX_PLUGIN <eigen_spmat_addons.hpp>

#include <Eigen/Sparse>

#include <interval_mesh.hpp>

namespace FEM1D {
    Eigen::MatrixXd reference_element_points_rs(const int degree);

    void gauss_legendre_quadrature(
        const int precision,
        Eigen::VectorXd &points_r_ret,
        Eigen::VectorXd &weights_r_ret
    );

    void lagrange_basis(
        const int degree,
        Eigen::VectorXd &qpoints_r,
        Eigen::MatrixXd &phi_r_ret,
        Eigen::MatrixXd &grad_phi_r_ret
    );

    void map_from_reference_cell(
        int degree, 
        Eigen::MatrixXd &node_coords_x,
        Eigen::VectorXd &qpoints_r,
        Eigen::MatrixXd &phi_r,
        Eigen::MatrixXd &grad_phi_r,
        Eigen::VectorXd &detJ_r_ret,
        Eigen::MatrixXd &qpoints_x_ret,
        Eigen::MatrixXd &grad_phi_x_ret
    );

    Eigen::SparseMatrix<double> assemble_mass_matrix(IntervalMesh &mesh, int gp);

    double inner(
        const Eigen::VectorXd &u_vec,
        const Eigen::SparseMatrix<double> &M,
        const Eigen::VectorXd &v_vec
    );
};
