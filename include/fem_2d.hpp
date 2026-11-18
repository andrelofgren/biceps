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

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <fem_function_2d.hpp>
#include <structured_mesh.hpp>

/**
 * @namespace FEM2D
 * @brief Namespace for 2D FEM utilities.
 */
namespace FEM2D {
    /**
     * @brief Returns the reference element node coordinates
     *
     * This function calculates the reference element node coordinates
     * for the specified cell type and degree.
     *
     * @param[in] cell_type The type of the cell (e.g., triangle, quadrilateral).
     * @param[in] degree The mesh degree.
     * @return Eigen::MatrixXd The reference element node coordinates.
     */
    Eigen::MatrixXd reference_element_points_rs(int cell_type, int degree);

    /**
     * @brief Computes the Gauss-Legendre quadrature in reference space (r, s)
     *
     * This function computes the Gauss-Legendre quadrature points and weights
     * in the reference space (r, s) for a given precision and cell type.
     *
     * @param[in] precision The quadrature precision.
     * @param[in] cell_type The type of the cell (e.g., triangle, quadrilateral).
     * @param[out] points_ret The computed quadrature points in reference space.
     * @param[out] weights_ret The computed quadrature weights in reference space.
     */
    void gauss_legendre_quadrature(
        const int precision,
        const int cell_type,
        Eigen::MatrixXd &points_ret,
        Eigen::VectorXd &weights_ret
    );


    /**
     * @brief Maps quadrature points from reference space (r,s) to physical space (x,z).
     *
     * This function transforms quadrature points from reference coordinates (r, s)
     * to physical coordinates (x, z).
     *
     * @param[in] node_coords The coordinates of the nodes in physical space.
     * @param[in] qpoints_rs The quadrature points in reference space.
     * @param[out] qpoints_xz_ret The quadrature points in physical space.
     */
    void map_rs_to_xz(
        Eigen::MatrixXd &node_coords_xz,
        Eigen::MatrixXd &qpoints_rs,
        Eigen::MatrixXd &qpoints_xz_ret
    );

    void map_xz_to_rs(
        Eigen::MatrixXd &node_coords_xz,
        Eigen::MatrixXd &qpoints_xz,
        Eigen::MatrixXd &qpoints_rs_ret
    );
    void lagrange_basis(
        const int degree,
        const int cell_type,
        Eigen::MatrixXd &qpoints_rs,
        Eigen::MatrixXd &phi_rs_ret,
        Eigen::MatrixXd &grad_phi_rs_ret
    );

    void map_from_reference_cell(
        const int degree,
        const int cell_type,
        const Eigen::MatrixXd &node_coords_xz,
        const Eigen::MatrixXd &qpoints_rs,
        const Eigen::MatrixXd &phi_rs,
        const Eigen::MatrixXd &grad_phi_rs,
        Eigen::VectorXd &detJ_rs_ret,
        Eigen::MatrixXd &qpoints_xz_ret,
        Eigen::MatrixXd &grad_phi_xz_ret
    );

    Eigen::SparseMatrix<double> assemble_mass_matrix(
        StructuredMesh &mesh, int gp
    );

    Eigen::SparseMatrix<double> assemble_stiffness_matrix(
        StructuredMesh &mesh, int gp
    );

    Eigen::SparseMatrix<double> assemble_stiffness_xx_matrix(
        StructuredMesh &mesh, int gp
    );

    Eigen::SparseMatrix<double> assemble_stiffness_zz_matrix(
        StructuredMesh &mesh, int gp
    );

    Eigen::SparseMatrix<double> assemble_stiffness_x_matrix(
        StructuredMesh &mesh, int gp
    );

    Eigen::SparseMatrix<double> assemble_stiffness_z_matrix(
        StructuredMesh &mesh, int gp
    );

    Eigen::MatrixXd assemble_extension_matrix(
        StructuredMesh &mesh,
        std::function<double(double, double)> force,
        int gp
    );

    double inner(
        const Eigen::VectorXd &u_vec,
        const Eigen::SparseMatrix<double> &M,
        const Eigen::VectorXd &v_vec
    );
}
