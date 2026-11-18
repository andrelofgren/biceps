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

#include <interval_mesh.hpp>
#include <Eigen/Sparse>

class FEMFunction1D {
private:
    Eigen::SparseMatrix<double> M;
    bool mass_mat_is_assembled = false;

public:
    IntervalMesh &mesh;
    Eigen::VectorXd vals;

    explicit FEMFunction1D(IntervalMesh &mesh);

    void assign(std::function<double(double, double)> func);
    void assign(const Eigen::VectorXd &vec);
    void assemble_mass_matrix();
    void set_mass_matrix(Eigen::SparseMatrix<double> &M);

    double integrate(int gp);
    Eigen::VectorXd eval_cell(int cell_index);
    Eigen::MatrixXd extract_vertex_subvec(int domain_id);
    Eigen::MatrixXd extract_dof_subvec(int domain_id);
    Eigen::SparseMatrix<double> mass_matrix();

    double L2_norm();
    FEMFunction1D operator+(double val);
    FEMFunction1D operator-(double val);
    FEMFunction1D operator*(double val);
    FEMFunction1D operator+(const FEMFunction1D &f);
    FEMFunction1D operator-(const FEMFunction1D &f);
    FEMFunction1D operator*(const FEMFunction1D &f);
};
