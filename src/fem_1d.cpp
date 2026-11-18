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
#include <fem_1d.hpp>

void FEM1D::map_from_reference_cell(
    int degree, 
    Eigen::MatrixXd &node_coords,
    Eigen::VectorXd &qpoints_r,
    Eigen::MatrixXd &phi_r,
    Eigen::MatrixXd &grad_phi_r,
    Eigen::VectorXd &detJ_r_ret,
    Eigen::MatrixXd &qpoints_x_ret,
    Eigen::MatrixXd &grad_phi_x_ret
) {
    Eigen::VectorXd dF;

    for (int i = 0; i < qpoints_r.rows(); i++) {
        dF = grad_phi_r(i, Eigen::placeholders::all) * node_coords;
        detJ_r_ret(i) = dF.norm();
        qpoints_x_ret(i, Eigen::placeholders::all) = phi_r(i, Eigen::placeholders::all) * node_coords;
        grad_phi_x_ret(i, Eigen::placeholders::all) = 1.0 / detJ_r_ret(i) * grad_phi_r(i, Eigen::placeholders::all);
    }
}

void FEM1D::lagrange_basis(
    int degree,
    Eigen::VectorXd &qpoints_r,
    Eigen::MatrixXd &phi_r_ret,
    Eigen::MatrixXd &grad_phi_r_ret
) {
    phi_r_ret = Eigen::MatrixXd::Zero(qpoints_r.rows(), degree+1);
    grad_phi_r_ret = Eigen::MatrixXd::Zero(qpoints_r.rows(), degree+1);

    for (int i = 0; i < qpoints_r.rows(); i++) {
        double r = qpoints_r(i);
        if (degree == 1) {
            phi_r_ret(i, 0) = (1-r);
            phi_r_ret(i, 1) = r;

            grad_phi_r_ret(i, 0) = -1;
            grad_phi_r_ret(i, 1) = 1;
        } else if (degree == 2) {
            phi_r_ret(i, 0) = 1 - 3*r + 2*r*r;
            phi_r_ret(i, 1) = 4*r - 4*r*r;
            phi_r_ret(i, 2) = -r + 2*r*r;

            grad_phi_r_ret(i, 0) = -3 + 4*r;
            grad_phi_r_ret(i, 1) = 4 - 8*r;
            grad_phi_r_ret(i, 2) = -1 + 4*r;
        }
    }
}

Eigen::MatrixXd FEM1D::reference_element_points_rs(
    const int degree
) {
    IntervalMesh mesh(0.0, 1.0, 1, degree);
    Eigen::MatrixXd ref_coords = mesh.pmat(mesh.cmat.row(0), Eigen::placeholders::all);
    return ref_coords;
}

void FEM1D::gauss_legendre_quadrature(
    const int precision,
    Eigen::VectorXd &points,
    Eigen::VectorXd &weights
) {
    points = Eigen::VectorXd::Zero(precision);
    weights = Eigen::VectorXd::Zero(precision);

    if (precision == 1) {
        points <<
            0.500000000000000000000000L;
        weights <<
            1.000000000000000000000000L;
    } else if (precision == 2) {
        points <<
            0.211324865405187117745426L,
            0.788675134594812882254574L;
        weights <<
            0.500000000000000000000000L,
            0.500000000000000000000000L;
    } else if (precision == 3) {
        points <<
            0.112701665379258311482073L,
            0.500000000000000000000000L,
            0.887298334620741688517927L;
        weights <<
            0.277777777777777777777778L,
            0.444444444444444444444444L,
            0.277777777777777777777778L;
    } else if (precision == 4) {
        points <<
            0.069431844202973712388027L,
            0.330009478207571867598667L,
            0.669990521792428132401333L,
            0.930568155797026287611973L;
        weights <<
            0.173927422568726928686532L,
            0.326072577431273071313468L,
            0.326072577431273071313468L,
            0.173927422568726928686532L;
    } else if (precision == 5) {
        points <<
            0.046910077030668003601187L,
            0.230765344947158454481843L,
            0.500000000000000000000000L,
            0.769234655052841545518157L,
            0.953089922969331996398813L;
        weights <<
            0.118463442528094543757132L,
            0.239314335249683234020646L,
            0.284444444444444444444444L,
            0.239314335249683234020646L,
            0.118463442528094543757132L;
    } else {
        throw std::invalid_argument("FEM1D::reference_element_points_rs: precision > 5 is too damn high");
    }
}

Eigen::SparseMatrix<double> FEM1D::assemble_mass_matrix(IntervalMesh &mesh, int gp) {
    std::vector<Eigen::Triplet<double>> mat_coeffs;

    Eigen::SparseMatrix<double> M_sp_ret = Eigen::SparseMatrix<double>(
        mesh.nof_dofs(), mesh.nof_dofs()
    );

    Eigen::MatrixXd node_coords, qpoints_x, phi_r, dphi_r, dphi_x;
    Eigen::VectorXd qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element;

    FEM1D::gauss_legendre_quadrature(gp, qpoints_r, qweights);
    FEM1D::lagrange_basis(mesh.degree(), qpoints_r, phi_r, dphi_r);

    qpoints_x = Eigen::MatrixXd::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorXd::Zero(qpoints_r.rows());
    dphi_x = Eigen::MatrixXd::Zero(
        qpoints_r.rows(), mesh.dofs_per_cell()
    );

    Eigen::MatrixXd M_block = Eigen::MatrixXd::Zero(
        mesh.dofs_per_cell(), mesh.dofs_per_cell()
    );

    for (int ci = 0; ci < mesh.nof_cells(); ci++) {
        element = mesh.cmat(ci, Eigen::placeholders::all);
        node_coords = mesh.pmat(element, Eigen::placeholders::all);

        FEM1D::map_from_reference_cell(
            mesh.degree(), node_coords, qpoints_r,
            phi_r, dphi_r, detJ_r, qpoints_x, dphi_x
        );

        for (int q = 0; q < qpoints_r.rows(); q++) {
            double detJxW = qweights(q) * detJ_r(q);
            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                    M_block(i, j) += phi_r(q, i) * phi_r(q, j) * detJxW;
                }
            }
        }

        for (int i = 0; i < mesh.dofs_per_cell(); i++) {
            for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                mat_coeffs.push_back(Eigen::Triplet<double>(
                    element(i),
                    element(j),
                    M_block(i, j)
                ));
            }
        }
        M_block.setZero();
    }

    M_sp_ret.setFromTriplets(mat_coeffs.begin(), mat_coeffs.end());

    return M_sp_ret;
}

double FEM1D::inner(
    const Eigen::VectorXd &u_vec,
    const Eigen::SparseMatrix<double> &M,
    const Eigen::VectorXd &v_vec
) {
    return u_vec.dot(M*v_vec);
}
