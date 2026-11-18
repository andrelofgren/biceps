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

#include <vector>
#include <functional>
#include <Eigen/Sparse>

#include <fem_function_2d.hpp>

class PoissonProblem {

private:
    int n_dofs;

public:
    std::vector<int> free_dofs;
    std::vector<int> fixed_dofs;

    explicit PoissonProblem(StructuredMesh &mesh);

    StructuredMesh &mesh;

    std::vector<Eigen::Triplet<double>> lhs_coeffs;

    Eigen::SparseMatrix<double> lhs_mat;
    Eigen::SparseMatrix<double> lhs_mat_free;
    Eigen::SparseMatrix<double> lhs_mat_fixed;

    Eigen::VectorXd bc_vec;
    Eigen::VectorXd rhs_vec;
    Eigen::VectorXd rhs_vec_free;
    Eigen::VectorXd sol_vec_free;
    Eigen::VectorXd sol_vec;

    void assemble_stiffness_block(
        std::function<double(double, double)> alpha,
        std::function<double(double, double)> beta,
        int gp
    );

    void assemble_mass_block(
        std::function<double(double, double)> gamma,
        int gauss_precision
    );

    void assemble_robin_block(
        std::function<double(double, double)> a_robin,
        std::function<double(double, double)> b_robin,
        std::function<double(double, double)> g_robin,
        int boundary_id,
        int gauss_precision
    );

    void commit_lhs_mat();

    void assemble_force_rhs(
        std::function<double(double, double)> f,
        int gauss_precision
    );
    void assemble_force_rhs(FEMFunction2D &f, int gauss_precision);

    void assemble_neumann_rhs(
        std::function<double(double, double)> g_neumann,
        int boundary_id,
        int gauss_precision
    );

    void assemble_robin_rhs(
        std::function<double(double, double)> b_robin,
        std::function<double(double, double)> g_robin,
        int boundary_id,
        int gauss_precision
    );

    void apply_dirichlet_bc(
        std::function<double(double, double)> g_dirichlet,
        int boundary_id
    );

    void solve_linear_system();

    FEMFunction2D solution();

    void reset_system();
};

