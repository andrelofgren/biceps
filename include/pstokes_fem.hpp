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
#include <logger.hpp>
#include <fem_function_2d.hpp>
#include <enums.hpp>

class pStokesProblem {

private:
    int nv_dofs;
    int np_dofs;
    int n_dofs;
    Logger logger;

public:
    double rate_factor;
    double glen_exponent;
    double eps_reg_2;
    int fssa_version = FSSA_NONE;
    double fssa_param = 0.0;
    int gp_stress = 5;
    int gp_incomp = 5;
    int gp_fssa_lhs = 5;
    int gp_fssa_rhs = 5;
    int gp_rhs = 5;

    std::function<double(double, double)> force_x;
    std::function<double(double, double)> force_z;
    std::function<double(double, double)> fssa_accum;

    int ux_dirichlet_bc_mask = 0;
    int uz_dirichlet_bc_mask = 0;

    std::vector<Eigen::Triplet<double>> lhs_coeffs;
    int nof_pushed_elements = 0;
    double picard_stol = 1e-8;
    double picard_max_iter = 100;

    StructuredMesh &u_mesh;
    StructuredMesh &p_mesh;

    Eigen::VectorXi ux_v2d;
    Eigen::VectorXi uz_v2d;
    Eigen::VectorXi u_v2d;
    Eigen::VectorXi p_v2d;

    Eigen::VectorXi ux_d2v;
    Eigen::VectorXi uz_d2v;
    Eigen::VectorXi p_d2v;
    Eigen::VectorXi w_d2v;

    Eigen::SparseMatrix<double> lhs_mat;
    Eigen::VectorXd rhs_vec;
    Eigen::VectorXd w_vec;

    pStokesProblem(
        double rate_factor,
        double glen_exponent,
        double eps_reg_2,
        std::function<double(double, double)> force_x,
        std::function<double(double, double)> force_z,
        StructuredMesh &u_mesh,
        StructuredMesh &p_mesh
    );

    void assemble_stress_block();
    void assemble_incomp_block();
    void assemble_fssa_vertical_block();
    void assemble_fssa_normal_block();

    void commit_lhs_mat();

    void assemble_rhs_vec();
    void assemble_fssa_vertical_rhs();

    void apply_zero_dirichlet_bc();
    void apply_dirichlet_bc(
        const int boundary_part,
        const int velocity_component,
        std::function<double(double, double)> ub_func
    );

    void prune_lhs(double threshold);

    void solve_linear_system();
    void solve_nonlinear_system();

    FEMFunction2D velocity_x();
    FEMFunction2D velocity_z();
    FEMFunction2D pressure();

    int log_level();
    void set_log_level(int value);

    void resize_lhs(int size);

    void reset_lhs();
    void reset_rhs();
    void reset_system();
};
