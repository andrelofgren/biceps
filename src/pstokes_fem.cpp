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
#include <pstokes_fem.hpp>
#include <fem_1d.hpp>
#include <fem_2d.hpp>
#include <enums.hpp>
#include <boost/format.hpp>
#include <cmath>

// Nonlinear viscosity function
static inline double glen_flow_viscosity(
    double A, double n_i, double eps_reg_2, double eff_strain_rate_2
) {
    return 0.5*pow(A, -1.0/n_i)*pow(
        eff_strain_rate_2 + eps_reg_2, (1.0-n_i)/(2.0*n_i)
    );
}

pStokesProblem::pStokesProblem(
    double rate_factor, double glen_exponent, double eps_reg_2,
    std::function<double(double, double)> force_x,
    std::function<double(double, double)> force_z,
    StructuredMesh &u_mesh, StructuredMesh &p_mesh
) :
    rate_factor(rate_factor), glen_exponent(glen_exponent), eps_reg_2(eps_reg_2),
    force_x(force_x), force_z(force_z), u_mesh(u_mesh), p_mesh(p_mesh),
    logger(INFO, std::cout)
{
    nv_dofs = u_mesh.nof_dofs();  // Number of velocity dofs
    np_dofs = p_mesh.nof_dofs();  // Number of pressure dofs
    n_dofs = 2*nv_dofs + np_dofs;  // Total number of dofs (velocity + pressure)

    ux_v2d = Eigen::VectorXi::Zero(nv_dofs);
    uz_v2d = Eigen::VectorXi::Zero(nv_dofs);
    u_v2d = Eigen::VectorXi::Zero(2*nv_dofs);
    p_v2d = Eigen::VectorXi::Zero(np_dofs); 

    ux_d2v = Eigen::VectorXi::Constant(n_dofs, -1);
    uz_d2v = Eigen::VectorXi::Constant(n_dofs, -1);

    int dof = 0;
    for (int vi = 0; vi < nv_dofs; ++vi) {
        ux_v2d(vi) = dof;
        ux_d2v(dof) = vi;
        dof += 2;
    }

    dof = 1;
    for (int vi = 0; vi < nv_dofs; ++vi) {
        uz_v2d(vi) = dof;
        uz_d2v(dof) = vi;
        dof += 2;
    }

    dof = 2*nv_dofs;
    for (int vi = 0; vi < np_dofs; ++vi) {
        p_v2d(vi) = dof;
        dof += 1;
    }

    u_v2d(Eigen::seqN(0, nv_dofs, 2)) = ux_v2d;
    u_v2d(Eigen::seq(1, 2*nv_dofs, 2)) = uz_v2d;

    lhs_mat = Eigen::SparseMatrix<double>(n_dofs, n_dofs);
    rhs_vec = Eigen::VectorXd::Zero(n_dofs);
    w_vec = Eigen::VectorXd::Zero(n_dofs);

    // Estimate the maximum number of non-zero entries
    int u_nnz_per_dof = 2*(2*u_mesh.degree() + 1)*(2*u_mesh.degree() + 1);
    int p_nnz_per_dof = (2*p_mesh.degree() + 1)*(2*p_mesh.degree() + 1);
    int nnz_per_dof = u_nnz_per_dof + p_nnz_per_dof;

    lhs_coeffs.reserve(nnz_per_dof * n_dofs);
}

void pStokesProblem::assemble_stress_block()
{
    Eigen::MatrixXd node_coords_u, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element_u;

    FEM2D::gauss_legendre_quadrature(
        gp_stress, u_mesh.cell_type(), qpoints_rs, qweights
    );

    FEM2D::lagrange_basis(
        u_mesh.degree(),
        u_mesh.cell_type(),
        qpoints_rs,
        phi_rs, dphi_rs
    );

    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2*qpoints_rs.rows(), u_mesh.dofs_per_cell());

    Eigen::MatrixXd A_xx = Eigen::MatrixXd::Zero(
        u_mesh.dofs_per_cell(), u_mesh.dofs_per_cell()
    );
    Eigen::MatrixXd A_xz = Eigen::MatrixXd::Zero(
        u_mesh.dofs_per_cell(), u_mesh.dofs_per_cell()
    );
    Eigen::MatrixXd A_zz = Eigen::MatrixXd::Zero(
        u_mesh.dofs_per_cell(), u_mesh.dofs_per_cell()
    );

    Eigen::MatrixXd A_block = Eigen::MatrixXd::Zero(
        2*u_mesh.dofs_per_cell(), 2*u_mesh.dofs_per_cell()
    );

    for (int k = 0; k < u_mesh.cmat.rows(); ++k) {
        element_u = u_mesh.cmat.row(k);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);

        FEM2D::map_from_reference_cell(
            u_mesh.degree(), u_mesh.cell_type(), node_coords_u, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        Eigen::VectorXd ux_vec = w_vec(ux_v2d(element_u));
        Eigen::VectorXd uz_vec = w_vec(uz_v2d(element_u));

        Eigen::MatrixXd Sx = dphi_xz(Eigen::seq(0, Eigen::last, 2), Eigen::all);
        Eigen::MatrixXd Sz = dphi_xz(Eigen::seq(1, Eigen::last, 2), Eigen::all);

        // Compute the gradients of the velocity components
        Eigen::VectorXd ddx_ux = Sx*ux_vec;
        Eigen::VectorXd ddz_ux = Sz*ux_vec;
        Eigen::VectorXd ddx_uz = Sx*uz_vec;
        Eigen::VectorXd ddz_uz = Sz*uz_vec;

        // Loop over quadrature points
        for (int q = 0; q < qpoints_rs.rows(); ++q) {
            int wx = 2*q;  // Index of x-derivative
            int wz = wx + 1;  // Index of z-derivative

            // Compute the effective strain rate based on the velocity gradients
            double eff_strain_rate_2 = 0.5*(
                pow(ddx_ux(q), 2)
                + 0.5*pow(ddz_ux(q) + ddx_uz(q), 2)
                + pow(ddz_uz(q), 2)
            );

            // Glen's flow law rheology to compute viscosity
            double eta = glen_flow_viscosity(
                rate_factor, glen_exponent, eps_reg_2, eff_strain_rate_2
            );

            double detJxWxEta = fabs(detJ_rs(q)) * qweights(q) * eta;

            for (int i = 0; i < u_mesh.dofs_per_cell(); i++) {
                for (int j = 0; j < u_mesh.dofs_per_cell(); j++) {
                    A_xx(i, j) += (2.0*dphi_xz(wx, i)*dphi_xz(wx, j) + dphi_xz(wz, i)*dphi_xz(wz, j))*detJxWxEta;
                    A_xz(i, j) += dphi_xz(wz, i)*dphi_xz(wx, j) * detJxWxEta;
                    A_zz(i, j) += (dphi_xz(wx, i)*dphi_xz(wx, j) + 2.0*dphi_xz(wz, i)*dphi_xz(wz, j))* detJxWxEta;
                }
            }
        }

        for (int i = 0; i < u_mesh.dofs_per_cell(); ++i) {
            for (int j = 0; j < u_mesh.dofs_per_cell(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    ux_v2d(element_u(i)),
                    ux_v2d(element_u(j)),
                    A_xx(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    ux_v2d(element_u(i)),
                    uz_v2d(element_u(j)),
                    A_xz(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    uz_v2d(element_u(i)),
                    ux_v2d(element_u(j)),
                    A_xz(j, i)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    uz_v2d(element_u(i)),
                    uz_v2d(element_u(j)),
                    A_zz(i, j)
                ));
            }
        }

        A_xx.setZero();
        A_xz.setZero();
        A_zz.setZero();
    }

    nof_pushed_elements = lhs_coeffs.size();
}

void pStokesProblem::assemble_incomp_block()
{
    Eigen::MatrixXd node_coords_u, node_coords_p, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz, psi_rs, dpsi_rs, dpsi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element_u, element_p;

    FEM2D::gauss_legendre_quadrature(
        gp_incomp, u_mesh.cell_type(), qpoints_rs, qweights
    );

    FEM2D::lagrange_basis(
        p_mesh.degree(), p_mesh.cell_type(), qpoints_rs, psi_rs, dpsi_rs
    );

    FEM2D::lagrange_basis(
        u_mesh.degree(), u_mesh.cell_type(), qpoints_rs, phi_rs, dphi_rs
    );

    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2*qpoints_rs.rows(), u_mesh.dofs_per_cell());
    dpsi_xz = Eigen::MatrixXd::Zero(2*qpoints_rs.rows(), p_mesh.dofs_per_cell());

    Eigen::MatrixXd B_px = Eigen::MatrixXd::Zero(
        p_mesh.dofs_per_cell(), u_mesh.dofs_per_cell()
    );
    Eigen::MatrixXd B_pz = Eigen::MatrixXd::Zero(
        p_mesh.dofs_per_cell(), u_mesh.dofs_per_cell()
    );
    Eigen::MatrixXd B_xp = Eigen::MatrixXd::Zero(
        u_mesh.dofs_per_cell(), p_mesh.dofs_per_cell()
    );
    Eigen::MatrixXd B_zp = Eigen::MatrixXd::Zero(
        u_mesh.dofs_per_cell(), p_mesh.dofs_per_cell()
    );

    for (int k = 0; k < u_mesh.cmat.rows(); ++k) {
        element_u = u_mesh.cmat.row(k);
        element_p = p_mesh.cmat.row(k);

        node_coords_u = u_mesh.pmat(element_u, Eigen::all);
        node_coords_p = p_mesh.pmat(element_p, Eigen::all);

        FEM2D::map_from_reference_cell(
            p_mesh.degree(), p_mesh.cell_type(), node_coords_p, qpoints_rs,
            psi_rs, dpsi_rs, detJ_rs, qpoints_xz, dpsi_xz
        );

        FEM2D::map_from_reference_cell(
            u_mesh.degree(), u_mesh.cell_type(), node_coords_u, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        for (int q = 0; q < qpoints_rs.rows(); ++q) {
            int w = 2*q;
            double detJxW = fabs(detJ_rs(q)) * qweights(q);

            for (int i = 0; i < p_mesh.dofs_per_cell(); ++i) {
                for (int j = 0; j < u_mesh.dofs_per_cell(); ++j) {
                    B_px(i, j) += -psi_rs(q, i) * dphi_xz(w, j) * detJxW;
                    B_pz(i, j) += -psi_rs(q, i) * dphi_xz(w + 1, j) * detJxW;
                }
            }
            for (int i = 0; i < u_mesh.dofs_per_cell(); ++i) {
                for (int j = 0; j < p_mesh.dofs_per_cell(); ++j) {
                    B_xp(i, j) += -dphi_xz(w, i) * psi_rs(q, j) * detJxW;
                    B_zp(i, j) += -dphi_xz(w + 1, i) * psi_rs(q, j) * detJxW;
                }
            }
        }

        for (int i = 0; i < p_mesh.dofs_per_cell(); ++i) {
            for (int j = 0; j < u_mesh.dofs_per_cell(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    p_v2d(element_p(i)), ux_v2d(element_u(j)), B_px(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    p_v2d(element_p(i)), uz_v2d(element_u(j)), B_pz(i, j)
                ));
            }
        }

        for (int i = 0; i < u_mesh.dofs_per_cell(); ++i) {
            for (int j = 0; j < p_mesh.dofs_per_cell(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    ux_v2d(element_u(i)), p_v2d(element_p(j)), B_xp(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    uz_v2d(element_u(i)), p_v2d(element_p(j)), B_zp(i, j)
                ));
            }
        }

        // Reset blocks
        B_px.setZero();
        B_pz.setZero();
        B_xp.setZero();
        B_zp.setZero();
    }

    nof_pushed_elements = lhs_coeffs.size();
}

void pStokesProblem::assemble_fssa_vertical_block() {
    Eigen::MatrixXd node_coords, qpoints_x, phi_r, dphi_r, dphi_x;
    Eigen::VectorXd qweights, qpoints_r, detJ_r;
    Eigen::VectorXi edge_vi;

    FEM1D::gauss_legendre_quadrature(gp_fssa_lhs, qpoints_r, qweights);

    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, phi_r, dphi_r
    );

    qpoints_x = Eigen::MatrixXd::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorXd::Zero(qpoints_r.rows());
    dphi_x = Eigen::MatrixXd::Zero(
        2*qpoints_r.rows(), u_mesh.dofs_per_edge()
    );

    Eigen::MatrixXd A_xz = Eigen::MatrixXd::Zero(
        u_mesh.dofs_per_edge(), u_mesh.dofs_per_edge()
    );
    Eigen::MatrixXd A_zz = Eigen::MatrixXd::Zero(
        u_mesh.dofs_per_edge(), u_mesh.dofs_per_edge()
    );

    std::vector<int> surf_edge_inds = u_mesh.extract_edge_inds(MESH2D::SURFACE_ID);

    for (int si : surf_edge_inds) {
        edge_vi = u_mesh.emat(si, Eigen::all);
        node_coords = u_mesh.pmat(edge_vi, Eigen::all);

        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords, qpoints_r,
            phi_r, dphi_r, detJ_r, qpoints_x, dphi_x
        );

        double nx = u_mesh.edge_normals(si, 0);
        double nz = u_mesh.edge_normals(si, 1);

        for (int q = 0; q < qpoints_r.rows(); ++q) {
            double fz = force_z(qpoints_x(q, 0), qpoints_x(q, 1));

            double dFxW = fabs(detJ_r(q)) * qweights(q);

            for (int i = 0; i < u_mesh.dofs_per_edge(); i++) {
                for (int j = 0; j < u_mesh.dofs_per_edge(); j++) {
                    A_xz(i, j) -= fssa_param * phi_r(q, i) * phi_r(q, j) * fz * dFxW * nx;
                    A_zz(i, j) -= fssa_param * phi_r(q, i) * phi_r(q, j) * fz * dFxW * nz;
                }
            }
        }

        for (int i = 0; i < u_mesh.dofs_per_edge(); ++i) {
            for (int j = 0; j < u_mesh.dofs_per_edge(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    uz_v2d(edge_vi(i)),
                    ux_v2d(edge_vi(j)),
                    A_xz(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    uz_v2d(edge_vi(i)),
                    uz_v2d(edge_vi(j)),
                    A_zz(i, j)
                ));
            }
        }

        A_xz.setZero();
        A_zz.setZero();
    }

    nof_pushed_elements = lhs_coeffs.size();
}

void pStokesProblem::assemble_fssa_normal_block() {
    Eigen::MatrixXd node_coords_u, qpoints_x, phi_r, dphi_r, dphi_x;
    Eigen::VectorXd qweights, qpoints_r, detJ_r;
    Eigen::VectorXi edge_vi;

    FEM1D::gauss_legendre_quadrature(gp_fssa_lhs, qpoints_r, qweights);

    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, phi_r, dphi_r
    );

    qpoints_x = Eigen::MatrixXd::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorXd::Zero(qpoints_r.rows());
    dphi_x = Eigen::MatrixXd::Zero(
        2*qpoints_r.rows(), u_mesh.dofs_per_edge()
    );

    Eigen::MatrixXd A_xx = Eigen::MatrixXd::Zero(
        u_mesh.dofs_per_edge(), u_mesh.dofs_per_edge()
    );
    Eigen::MatrixXd A_xz = Eigen::MatrixXd::Zero(
        u_mesh.dofs_per_edge(), u_mesh.dofs_per_edge()
    );
    Eigen::MatrixXd A_zx = Eigen::MatrixXd::Zero(
        u_mesh.dofs_per_edge(), u_mesh.dofs_per_edge()
    );
    Eigen::MatrixXd A_zz = Eigen::MatrixXd::Zero(
        u_mesh.dofs_per_edge(), u_mesh.dofs_per_edge()
    );

    std::vector<int> surf_edge_inds = u_mesh.extract_edge_inds(MESH2D::SURFACE_ID);

    for (int si : surf_edge_inds) {
        edge_vi = u_mesh.emat(si, Eigen::all);
        node_coords_u = u_mesh.pmat(edge_vi, Eigen::all);

        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            phi_r, dphi_r, detJ_r, qpoints_x, dphi_x
        );

        double nz = sqrt(u_mesh.edge_normals(si, 1));
        double nx = u_mesh.edge_normals(si, 0) / nz;

        for (int q = 0; q < qpoints_r.rows(); ++q) {
            double fz = force_z(qpoints_x(q, 0), qpoints_x(q, 1));

            double dFxW = fabs(detJ_r(q)) * qweights(q);

            for (int i = 0; i < u_mesh.dofs_per_edge(); ++i) {
                for (int j = 0; j < u_mesh.dofs_per_edge(); ++j) {
                    A_xx(i, j) -= fssa_param*phi_r(q, i)*phi_r(q, j)*fz*dFxW*nx*nx;
                    A_xz(i, j) -= fssa_param*phi_r(q, i)*phi_r(q, j)*fz*dFxW*nx*nz;
                    A_zx(i, j) -= fssa_param*phi_r(q, i)*phi_r(q, j)*fz*dFxW*nz*nx;
                    A_zz(i, j) -= fssa_param*phi_r(q, i)*phi_r(q, j)*fz*dFxW*nz*nz;
                }
            }
        }

        for (int i = 0; i < u_mesh.dofs_per_edge(); ++i) {
            for (int j = 0; j < u_mesh.dofs_per_edge(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    ux_v2d(edge_vi(i)),
                    ux_v2d(edge_vi(j)),
                    A_xx(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    ux_v2d(edge_vi(i)),
                    uz_v2d(edge_vi(j)),
                    A_xz(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    uz_v2d(edge_vi(i)),
                    ux_v2d(edge_vi(j)),
                    A_zx(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    uz_v2d(edge_vi(i)),
                    uz_v2d(edge_vi(j)),
                    A_zz(i, j)
                ));
            }
        }

        A_xx.setZero();
        A_xz.setZero();
        A_zx.setZero();
        A_zz.setZero();
    }

    nof_pushed_elements = lhs_coeffs.size();
}

void pStokesProblem::assemble_fssa_vertical_rhs() {
    Eigen::MatrixXd node_coords_u, qpoints_x, phi_r, dphi_r, dphi_x;
    Eigen::VectorXd qweights, qpoints_r, detJ_r;
    Eigen::VectorXi edge_vi;

    FEM1D::gauss_legendre_quadrature(gp_fssa_rhs, qpoints_r, qweights);

    FEM1D::lagrange_basis(u_mesh.degree(), qpoints_r, phi_r, dphi_r);

    qpoints_x = Eigen::MatrixXd::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorXd::Zero(qpoints_r.rows());
    dphi_x = Eigen::MatrixXd::Zero(
        2 * qpoints_r.rows(), u_mesh.dofs_per_edge()
    );

    std::vector<int> surf_edge_inds = u_mesh.extract_edge_inds(MESH2D::SURFACE_ID);

    for (int si : surf_edge_inds) {
        edge_vi = u_mesh.emat(si, Eigen::all);
        node_coords_u = u_mesh.pmat(edge_vi, Eigen::all);

        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            phi_r, dphi_r, detJ_r, qpoints_x, dphi_x
        );

        double nz = u_mesh.edge_normals(si, 1);

        for (int q = 0; q < qpoints_r.rows(); ++q) {
            double fx = force_x(qpoints_x(q, 0), qpoints_x(q, 1));
            double fz = force_z(qpoints_x(q, 0), qpoints_x(q, 1));
            double ac = fssa_accum(qpoints_x(q, 0), qpoints_x(q, 1));

            double dFxW = fabs(detJ_r(q)) * qweights(q);

            for (int i = 0; i < u_mesh.dofs_per_edge(); ++i) {
                rhs_vec(ux_v2d(edge_vi(i))) += fssa_param*phi_r(q, i)*ac*fx*dFxW*nz;
                rhs_vec(uz_v2d(edge_vi(i))) += fssa_param*phi_r(q, i)*ac*fz*dFxW*nz;
            }
        }
    }
}

void pStokesProblem::assemble_rhs_vec()
{
    Eigen::MatrixXd node_coords_u, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element_u;

    FEM2D::gauss_legendre_quadrature(
        gp_rhs, u_mesh.cell_type(), qpoints_rs, qweights
    );

    FEM2D::lagrange_basis(
        u_mesh.degree(), u_mesh.cell_type(), qpoints_rs, phi_rs, dphi_rs
    );

    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2 * qpoints_rs.rows(), u_mesh.dofs_per_cell());

    for (int k = 0; k < u_mesh.cmat.rows(); k++) {
        element_u = u_mesh.cmat.row(k);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);

        FEM2D::map_to_reference_cell(
            u_mesh.degree(), u_mesh.cell_type(), node_coords_u, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        for (int q = 0; q < qpoints_rs.rows(); ++q) {
            double detJxW = fabs(detJ_rs(q)) * qweights(q);

            for (int i = 0; i < u_mesh.dofs_per_cell(); ++i) {
                rhs_vec(ux_v2d(element_u(i))) += (
                    force_x(qpoints_xz(q, 0), qpoints_xz(q, 1)) * phi_rs(q, i)
                ) * detJxW;

                rhs_vec(uz_v2d(element_u(i))) += (
                    force_z(qpoints_xz(q, 0), qpoints_xz(q, 1))*phi_rs(q, i)
                )*detJxW;
            }
        }
    }
}

void pStokesProblem::commit_lhs_mat()
{
    lhs_mat.setFromTriplets(lhs_coeffs.begin(), lhs_coeffs.end());
}

void pStokesProblem::apply_zero_dirichlet_bc()
{
    // Loop over all nnz elements and set off-diagonals corresponding to boundary nodes to zero
    for (int col = 0; col < lhs_mat.outerSize(); ++col)
        // Check if the column corresponds to a boundary node for either ux or uz
        if (
            ux_d2v(col) != -1 && u_mesh.dimat(ux_d2v(col)) & ux_dirichlet_bc_mask ||
            uz_d2v(col) != -1 && u_mesh.dimat(uz_d2v(col)) & uz_dirichlet_bc_mask 
        ) {
            // Eliminate column
            for (Eigen::SparseMatrix<double>::InnerIterator it(lhs_mat, col); it; ++it) {
                it.valueRef() = 0.0;
            }
            lhs_mat.coeffRef(col, col) = 1.0;
            rhs_vec(col) = 0.0; // Zero Dirichlet bc
        } else {
            for (Eigen::SparseMatrix<double>::InnerIterator it(lhs_mat, col); it; ++it) {
                int row = it.row();
                // Check if the row corresponds to a boundary node for either ux or uz
                if (
                    ux_d2v(row) != -1 && u_mesh.dimat(ux_d2v(row)) & ux_dirichlet_bc_mask ||
                    uz_d2v(row) != -1 && u_mesh.dimat(uz_d2v(row)) & uz_dirichlet_bc_mask
                ) {
                    // Eliminate row
                    it.valueRef() = 0.0;
                }
            }
        }
    prune_lhs(1e-12);
}

void pStokesProblem::apply_dirichlet_bc(
    const int boundary_id,
    const int velocity_component,
    std::function<double(double, double)> ub_func
) {
    // Loop over all nnz elements and set off-diagonals corresponding to boundary nodes to zero
    int *outer_end = lhs_mat.outerIndexPtr() + lhs_mat.rows();
    int col = 0;
    for (
        int *outer_p = lhs_mat.outerIndexPtr();
        outer_p != outer_end;
        outer_p++
    ) {
        int nnz = *(outer_p + 1) - *outer_p;
        for (int i = 0; i < nnz; i++) {
            int row = *(lhs_mat.innerIndexPtr() + *outer_p + i);
            if (velocity_component == HORIZONTAL && ux_d2v(row) != -1) {
                if (u_mesh.dimat(ux_d2v(row)) & boundary_id) {
                    double *val_ptr = lhs_mat.valuePtr() + *outer_p + i;
                    *val_ptr = 0.0;
                }
            }
            else if (velocity_component == VERTICAL && uz_d2v(row) != -1) {
                if (u_mesh.dimat(uz_d2v(row)) & boundary_id) {
                    double *val_ptr = lhs_mat.valuePtr() + *outer_p + i;
                    *val_ptr = 0.0;
                }
            }
        }
        col++;
    }

    // Loop over all nnz elements and set diagonals corresponding to boundary nodes to one
    col = 0;
    for (
        int *outer_p = lhs_mat.outerIndexPtr();
        outer_p != outer_end;
        outer_p++
    ) {
        if (velocity_component == HORIZONTAL && ux_d2v(col) != -1) {
            if (u_mesh.dimat(ux_d2v(col)) & boundary_id) {
                double x = u_mesh.pmat(ux_d2v(col), 0);
                double z = u_mesh.pmat(ux_d2v(col), 1);
                lhs_mat.coeffRef(col, col) = 1.0;
                rhs_vec(col) = ub_func(x, z);  // Dirichlet bc
            }
        }
        else if (velocity_component == VERTICAL && uz_d2v(col) != -1) {
            if (u_mesh.dimat(uz_d2v[col]) & boundary_id) {
                double x = u_mesh.pmat(uz_d2v(col), 0);
                double z = u_mesh.pmat(uz_d2v(col), 1);
                lhs_mat.coeffRef(col, col) = 1.0;
                rhs_vec(col) = ub_func(x, z);  // Dirichlet bc
            }
        }
        col++;
    }
}

void pStokesProblem::prune_lhs(double threshold)
{
    // NOTE: Pruning zeros is a costly operation and should only be done once.
    lhs_mat.prune(threshold, 1);
}

void pStokesProblem::solve_linear_system()
{
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(lhs_mat);
    solver.factorize(lhs_mat);
    w_vec = solver.solve(rhs_vec);
}

void pStokesProblem::solve_nonlinear_system()
{
    // Assemble incompressibility block
    assemble_incomp_block();

    // Assemble FSSA block according to fssa version
    if (fssa_version == FSSA_VERTICAL)
        assemble_fssa_vertical_block();
    else if (fssa_version == FSSA_NORMAL)
        assemble_fssa_normal_block();

    // Assemble the right-hand side vector
    assemble_rhs_vec();
    int nof_pushed_elements_before_stress_assembly = nof_pushed_elements;

    double error_step = 0.0;
    FEMFunction2D ux_old_func = velocity_x();
    FEMFunction2D uz_old_func = velocity_z();
    FEMFunction2D ux_new_func(u_mesh);
    FEMFunction2D uz_new_func(u_mesh);
    FEMFunction2D ex_func(u_mesh);
    FEMFunction2D ez_func(u_mesh);

    // Assemble the mass matrix for calculating the L2 error
    Eigen::SparseMatrix<double> M = FEM2D::assemble_mass_matrix(u_mesh, u_mesh.degree() + 1);
    ux_new_func.set_mass_matrix(M);
    uz_new_func.set_mass_matrix(M);
    ex_func.set_mass_matrix(M);
    ez_func.set_mass_matrix(M);

    // Picard iteration loop
    int ip = 0;
    do {
        // Only need to reassemble stress block between iterations
        lhs_coeffs.resize(nof_pushed_elements_before_stress_assembly);
        nof_pushed_elements = nof_pushed_elements_before_stress_assembly;

        // Assemble stress block and commit
        assemble_stress_block();
        commit_lhs_mat();

        // Apply homogenous Dirichlet BC and solve linear system
        apply_zero_dirichlet_bc();
        solve_linear_system();

        // Compute the step error and check convergence
        ux_new_func.vals = w_vec(ux_v2d);
        uz_new_func.vals = w_vec(uz_v2d);
        ex_func.vals = ux_new_func.vals - ux_old_func.vals;
        ez_func.vals = uz_new_func.vals - uz_old_func.vals;
        double ux = ux_new_func.L2_norm();
        double uz = uz_new_func.L2_norm();
        double ex = ex_func.L2_norm();
        double ez = ez_func.L2_norm();
        error_step = sqrt((ex*ex + ez*ez)/(ux*ux + uz*uz));

        // Log iteration information
        logger.log_msg((boost::format{"<pStokes> Picard iteration %d: r(step) = %g"} %(ip+1) %error_step).str(), TRACE);

        // Update old functions for the next iteration
        ux_old_func.assign(ux_new_func);
        uz_old_func.assign(uz_new_func);
        ++ip;
    } while (error_step > picard_stol && ip < picard_max_iter);

    // Log summary after solving
    logger.log_msg("-------------------------------------", INFO);
    logger.log_msg("********** pStokes Summary **********", INFO);
    logger.log_msg("-------------------------------------", INFO);
    logger.log_msg((boost::format{"Picard solver finished in %d iterations"} %ip).str(), INFO);
    logger.log_msg((boost::format{"r (step): %.4g"} % (error_step)).str(), INFO);
    logger.log_msg((boost::format{"min ux, max ux: %.4g, %.4g m/yr"} % (1e3*w_vec(ux_v2d).minCoeff()) % (1e3*w_vec(ux_v2d).maxCoeff())).str(), INFO);
    logger.log_msg((boost::format{"min uz, max uz: %.4g, %.4g m/yr"} % (1e3*w_vec(uz_v2d).minCoeff()) % (1e3*w_vec(uz_v2d).maxCoeff())).str(), INFO);
    logger.log_msg((boost::format{"min p, max p: %.4g, %.4g MPa"} % (w_vec(p_v2d).minCoeff()) % (w_vec(p_v2d).maxCoeff())).str(), INFO);
    logger.log_msg("-------------------------------------", INFO);
}

FEMFunction2D pStokesProblem::velocity_x()
{
    FEMFunction2D ux_fem_func(u_mesh);
    ux_fem_func.vals = w_vec(ux_v2d);
    return ux_fem_func;
}

FEMFunction2D pStokesProblem::velocity_z()
{
    FEMFunction2D uz_fem_func(u_mesh);
    uz_fem_func.vals = w_vec(uz_v2d);
    return uz_fem_func;
}

FEMFunction2D pStokesProblem::pressure()
{
    FEMFunction2D p_fem_func(p_mesh);
    p_fem_func.vals = w_vec(p_v2d);
    return p_fem_func;
}

void pStokesProblem::resize_lhs(int size)
{
    if (size < 0)
        throw std::invalid_argument("size can not be negative");

    lhs_coeffs.resize(size);
}

void pStokesProblem::reset_lhs()
{
    lhs_coeffs.clear();
    nof_pushed_elements = 0;
}

void pStokesProblem::reset_rhs()
{
    rhs_vec.setZero();
}

void pStokesProblem::reset_system()
{
    reset_lhs();
    reset_rhs();
}

int pStokesProblem::log_level()
{
    return logger.log_level();
}

void pStokesProblem::set_log_level(int value)
{
    logger.set_log_level(value);
}
