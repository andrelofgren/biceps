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

#include <structured_mesh.hpp>
#include <Eigen/Sparse>

class FEMFunction2D {
private:
    Eigen::SparseMatrix<double> M;
    bool mass_mat_is_assembled = false;

public:
    explicit FEMFunction2D(StructuredMesh &mesh);

    StructuredMesh &mesh;
    Eigen::VectorXd vals;

    void assign(std::function<double(double, double)> func);
    void assign(const Eigen::VectorXd &vec);
    void assign(const FEMFunction2D &f);
    double eval(double x, double z);

    Eigen::VectorXd eval_cell(int cell_index);
    Eigen::VectorXd eval_edge(int edge_index);
    Eigen::MatrixXd extract_vertex_subvec(int domain_id);
    Eigen::MatrixXd extract_dof_subvec(int domain_id);

    void assemble_mass_matrix();
    void set_mass_matrix(Eigen::SparseMatrix<double> &M);

    FEMFunction2D diff_x_interp();
    FEMFunction2D diff_z_interp();
    FEMFunction2D diff_x_proj();
    FEMFunction2D diff_z_proj();

    Eigen::SparseMatrix<double> mass_matrix();

    double L2_norm();

    double operator()(double x, double z);
    FEMFunction2D operator+(double val);
    FEMFunction2D operator-(double val);
    FEMFunction2D operator*(double val);
    FEMFunction2D operator+(const FEMFunction2D &f);
    FEMFunction2D operator-(const FEMFunction2D &f);
    FEMFunction2D operator*(const FEMFunction2D &f);
};
