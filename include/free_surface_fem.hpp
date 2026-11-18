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

#include <fem_function_1d.hpp>
#include <Eigen/Sparse>
#include <enums.hpp>

class FreeSurfaceProblem {

private:
    std::vector<Eigen::Triplet<double>> lhs_coeffs;
    Eigen::SparseMatrix<double> lhs_mat;
    Eigen::VectorXd rhs_vec;

public:
    IntervalMesh &h_mesh;
    IntervalMesh &u_mesh;

    Eigen::VectorXd zs_vec;

    int gp_lhs = 5;
    int gp_rhs = 5;

    int nof_pushed_elements = 0; ///< Number of elements pushed into the LHS matrix.

    FreeSurfaceProblem(IntervalMesh &h_mesh, IntervalMesh &u_mesh);

    void assemble_lhs_explicit();
    void assemble_rhs_explicit(
        FEMFunction1D &h0_fem_func,
        FEMFunction1D &ux_fem_func,
        FEMFunction1D &uz_fem_func,
        FEMFunction1D &ac_fem_func,
        double dt
    );

    void assemble_lhs_simplicit(
        FEMFunction1D &ux_fem_func,
        double dt
    );
    void assemble_rhs_simplicit(
        FEMFunction1D &h0_fem_func,
        FEMFunction1D &uz_fem_func,
        FEMFunction1D &ac_fem_func,
        double dt
    );

    void commit_lhs();

    void solve_linear_system();

    FEMFunction1D height();

    void reset_lhs();
    void reset_rhs();

    void reset_system();
};
