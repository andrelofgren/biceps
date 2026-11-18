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
#include <poisson_fem.hpp>
#include <fem_1d.hpp>
#include <fem_2d.hpp>
#include <enums.hpp>

PoissonProblem::PoissonProblem(StructuredMesh &mesh) : mesh(mesh)
{
    int deg = mesh.degree();
    n_dofs = mesh.nof_dofs();
    lhs_mat = Eigen::SparseMatrix<double>(n_dofs, n_dofs);
    rhs_vec = Eigen::VectorXd::Zero(n_dofs);
    sol_vec = Eigen::VectorXd::Zero(n_dofs);

    int nnz_per_dof = (deg + 2)*(deg + 2);
    lhs_coeffs.reserve(n_dofs*nnz_per_dof);
}

void PoissonProblem::reset_system()
{
    lhs_coeffs.clear();
    rhs_vec.setZero();
}

void PoissonProblem::assemble_stiffness_block(
    std::function<double(double, double)> alpha,
    std::function<double(double, double)> beta,
    int gauss_precision
) {
    // Initialize variables for quadrature, basis functions, and geometry
    Eigen::MatrixXd node_coords_v, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element;

    // Retrieve quadrature points and weights
    FEM2D::gauss_legendre_quadrature(
        gauss_precision, mesh.cell_type(), qpoints_rs, qweights
    );
    // Retrieve lagrange basis evaluated at quadrature points
    FEM2D::lagrange_basis(
        mesh.degree(), mesh.cell_type(), qpoints_rs, phi_rs, dphi_rs
    );

    // Initialize matrices and vectors for stiffness assembly
    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2 * qpoints_rs.rows(), mesh.dofs_per_cell());
    Eigen::MatrixXd A_block = Eigen::MatrixXd::Zero(
        mesh.dofs_per_cell(), mesh.dofs_per_cell()
    );

    // Loop over elements to assemble the stiffness matrix
    for (int k = 0; k < mesh.cmat.rows(); k++) {
        element = mesh.cmat.row(k);
        node_coords_v = mesh.pmat(element, Eigen::placeholders::all);

        // Map to reference cell and compute the stiffness block for the element
        FEM2D::map_from_reference_cell(
            mesh.degree(), mesh.cell_type(), node_coords_v, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        // Assemble stiffness matrix block for the element
        for (int q = 0; q < qpoints_rs.rows(); q++) {
            int wx = 2 * q;
            int wz = wx + 1;
            double alpha_xz = alpha(qpoints_xz(q, 0), qpoints_xz(q, 1));
            double beta_xz = beta(qpoints_xz(q, 0), qpoints_xz(q, 1));
            double detJxW = fabs(detJ_rs(q)) * qweights(q);
            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                    A_block(i, j) += (
                        alpha_xz * dphi_xz(wx, i) * dphi_xz(wx, j) + 
                        beta_xz * dphi_xz(wz, i) * dphi_xz(wz, j)
                    ) * detJxW;
                }
            }
        }

        // Store computed block into the global matrix
        for (int i = 0; i < mesh.dofs_per_cell(); i++) {
            for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    element(i), element(j), A_block(i, j)
                ));
            }
        }

        A_block.setZero(); // Reset block for the next element
    }
}

void PoissonProblem::assemble_mass_block(
    std::function<double(double, double)> gamma,
    int gauss_precision
) {
    // Initialize variables for quadrature, basis functions, and geometry
    Eigen::MatrixXd node_coords_v, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element;

    // Retrieve quadrature points and weights
    FEM2D::gauss_legendre_quadrature(
        gauss_precision, mesh.cell_type(), qpoints_rs, qweights
    );
    // Retrieve lagrange basis evaluated at quadrature points
    FEM2D::lagrange_basis(
        mesh.degree(), mesh.cell_type(), qpoints_rs, phi_rs, dphi_rs
    );

    // Initialize matrices and vectors for mass matrix assembly
    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(
        2*qpoints_rs.rows(), mesh.dofs_per_cell()
    );
    Eigen::MatrixXd M_block = Eigen::MatrixXd::Zero(
        mesh.dofs_per_cell(), mesh.dofs_per_cell()
    );

    // Loop over elements to assemble the mass matrix
    for (int k = 0; k < mesh.cmat.rows(); k++) {
        element = mesh.cmat.row(k);
        node_coords_v = mesh.pmat(element, Eigen::placeholders::all);

        // Map to reference cell and compute the mass block for the element
        FEM2D::map_from_reference_cell(
            mesh.degree(), mesh.cell_type(), node_coords_v, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        // Assemble mass matrix block for the element
        for (int q = 0; q < qpoints_rs.rows(); q++) {
            double detJxWxGamma = gamma(
                qpoints_xz(q, 0), qpoints_xz(q, 1)
            ) * fabs(detJ_rs(q)) * qweights(q);
            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                    M_block(i, j) += phi_rs(q, i) * phi_rs(q, j) * detJxWxGamma;
                }
            }
        }

        // Store computed block into the global matrix
        for (int i = 0; i < mesh.dofs_per_cell(); i++) {
            for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    element(i), element(j), M_block(i, j)
                ));
            }
        }

        M_block.setZero(); // Reset block for the next element
    }
}

void PoissonProblem::assemble_robin_block(
    std::function<double(double, double)> g_robin,
    std::function<double(double, double)> a_robin,
    std::function<double(double, double)> b_robin,
    int boundary_id,
    int gauss_precision
) {
    // Declare variables for geometry, quadrature, and basis functions
    Eigen::MatrixXd node_coords, qpoints_x, phi_r, dphi_r, dphi_x;
    Eigen::VectorXd qweights, qpoints_r, detJ_r;
    Eigen::VectorXi edge_vi;

    // Retrieve quadrature points and weights for 1D integration
    FEM1D::gauss_legendre_quadrature(gauss_precision, qpoints_r, qweights);

    // Retrieve Lagrange basis functions and their derivatives at quadrature points
    FEM1D::lagrange_basis(mesh.degree(), qpoints_r, phi_r, dphi_r);

    // Initialize matrices and vectors for mapping and transformations
    qpoints_x = Eigen::MatrixXd::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorXd::Zero(qpoints_r.rows());
    dphi_x = Eigen::MatrixXd::Zero(
        2*qpoints_r.rows(), mesh.dofs_per_cell()
    );

    // Initialize a block matrix for mass computation
    Eigen::MatrixXd M_block = Eigen::MatrixXd::Zero(
        mesh.dofs_per_edge(), mesh.dofs_per_edge()
    );

    // Extract the edges of the mesh corresponding to the given boundary ID
    std::vector<int> surf_edge_inds = mesh.extract_edge_inds(boundary_id);
    for (int si: surf_edge_inds) {
        // Get the indices of nodes forming the edge
        edge_vi = mesh.emat(si, Eigen::placeholders::all);
        node_coords = mesh.pmat(edge_vi, Eigen::placeholders::all);

        // Map the node coordinates to the reference cell
        FEM1D::map_from_reference_cell(
            mesh.degree(), node_coords, qpoints_r,
            phi_r, dphi_r, detJ_r, qpoints_x, dphi_x
        );

        // Loop over the quadrature points
        for (int q = 0; q < qpoints_r.rows(); q++) {
            // Retrieve the Robin boundary function values at the quadrature point
            double ar = a_robin(qpoints_x(q, 0), qpoints_x(q, 1));
            double br = b_robin(qpoints_x(q, 0), qpoints_x(q, 1));
            double gr = g_robin(qpoints_x(q, 0), qpoints_x(q, 1));

            // Calculate a coefficient for the block matrix
            double cr = ar/br;
            double dFxWxCr = cr*fabs(detJ_r(q))*qweights(q);

            // Update the block matrix for the mass term
            for (int i = 0; i < mesh.dofs_per_edge(); i++) {
                for (int j = 0; j < mesh.dofs_per_edge(); j++) {
                    M_block(i, j) += phi_r(q, i)*phi_r(q, j)*dFxWxCr;
                }
            }
        }

        // Fill the sparse matrix with the values from the block matrix
        for (int i = 0; i < mesh.dofs_per_edge(); i++) {
            for (int j = 0; j < mesh.dofs_per_edge(); j++) {
                lhs_coeffs.push_back(Eigen::Triplet<double>(
                    edge_vi(i),
                    edge_vi(j),
                    M_block(i, j)
                ));
            }
        }
        M_block.setZero();  // Reset the block matrix for the next iteration
    }
}

void PoissonProblem::assemble_neumann_rhs(
    std::function<double(double, double)> g_neumann,
    int boundary_id,
    int gauss_precision
) {
    // Declare variables for geometry, quadrature, and basis functions
    Eigen::MatrixXd node_coords, qpoints_x, phi_r, dphi_r, dphi_x;
    Eigen::VectorXd qweights, qpoints_r, detJ_r;
    Eigen::VectorXi edge_vi;

    // Retrieve quadrature points and weights for 1D integration
    FEM1D::gauss_legendre_quadrature(gauss_precision, qpoints_r, qweights);

    // Retrieve Lagrange basis functions and their derivatives at quadrature points
    FEM1D::lagrange_basis(mesh.degree(), qpoints_r, phi_r, dphi_r);

    // Initialize matrices and vectors for mapping and transformations
    qpoints_x = Eigen::MatrixXd::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorXd::Zero(qpoints_r.rows());
    dphi_x = Eigen::MatrixXd::Zero(
        2*qpoints_r.rows(), mesh.dofs_per_cell()
    );

    // Extract the edges of the mesh corresponding to the given boundary ID
    std::vector<int> surf_edge_inds = mesh.extract_edge_inds(boundary_id);
    for (int si: surf_edge_inds) {
        // Get the indices of nodes forming the edge
        edge_vi = mesh.emat(si, Eigen::placeholders::all);
        node_coords = mesh.pmat(edge_vi, Eigen::placeholders::all);

        // Map the node coordinates to the reference cell
        FEM1D::map_from_reference_cell(
            mesh.degree(), node_coords, qpoints_r,
            phi_r, dphi_r, detJ_r, qpoints_x, dphi_x
        );

        // Loop over the quadrature points
        for (int q = 0; q < qpoints_r.rows(); q++) {
            // Retrieve the Neumann boundary condition value at the quadrature point
            double gn = g_neumann(qpoints_x(q, 0), qpoints_x(q, 1));

            // Compute the weighted value at this quadrature point
            double detJxWxGn = gn*fabs(detJ_r(q))*qweights(q);

            // Add contribution to the right-hand side vector using the basis functions
            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                rhs_vec(edge_vi(i)) += phi_r(q, i)*detJxWxGn;
            }
        }
    }
}

void PoissonProblem::assemble_force_rhs(
    std::function<double(double, double)> f, int gauss_precision
) {
    // Declare variables for geometry, quadrature, and basis functions
    Eigen::MatrixXd node_coords_v, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element;

    // Retrieve quadrature points and weights for 2D integration
    FEM2D::gauss_legendre_quadrature(
        gauss_precision, mesh.cell_type(), qpoints_rs, qweights
    );

    // Retrieve Lagrange basis functions and their derivatives at quadrature points
    FEM2D::lagrange_basis(
        mesh.degree(), mesh.cell_type(), qpoints_rs, phi_rs, dphi_rs
    );

    // Initialize matrices and vectors for mapping and transformations
    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2*qpoints_rs.rows(), mesh.dofs_per_cell());

    // Loop over each element in the mesh
    for (int k = 0; k < mesh.cmat.rows(); k++) {
        // Retrieve element indices for the current element
        element = mesh.cmat.row(k);
        node_coords_v = mesh.pmat(element, Eigen::placeholders::all);

        // Map the node coordinates to the reference cell
        FEM2D::map_from_reference_cell(
            mesh.degree(), mesh.cell_type(), node_coords_v, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        // Loop over the quadrature points
        for (int q = 0; q < qpoints_rs.rows(); q++) {
            // Compute the weighted value of the function f at the quadrature point
            double detJxWxf = f(
                qpoints_xz(q, 0), qpoints_xz(q, 1)
            ) * fabs(detJ_rs(q)) * qweights(q);

            // Add contribution to the right-hand side vector using the basis functions
            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                rhs_vec(element(i)) += phi_rs(q, i) * detJxWxf;
            }
        }
    }
}

void PoissonProblem::assemble_force_rhs(
    FEMFunction2D &f, int gauss_precision
) {
    // Declare variables for geometry, quadrature, basis functions, and force evaluation
    Eigen::MatrixXd node_coords_v, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs, f_cell;
    Eigen::VectorXi element;

    // Retrieve quadrature points and weights for integration in 2D
    FEM2D::gauss_legendre_quadrature(
        gauss_precision, mesh.cell_type(), qpoints_rs, qweights
    );

    // Retrieve Lagrange basis functions and their derivatives at quadrature points
    FEM2D::lagrange_basis(
        mesh.degree(), mesh.cell_type(), qpoints_rs, phi_rs, dphi_rs
    );

    // Initialize matrices and vectors for mappings and transformations
    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2*qpoints_rs.rows(), mesh.dofs_per_cell());

    // Loop over each element in the mesh
    for (int k = 0; k < mesh.cmat.rows(); k++) {
        // Retrieve element indices for the current element
        element = mesh.cmat.row(k);
        node_coords_v = mesh.pmat(element, Eigen::placeholders::all);

        // Map the node coordinates to the reference cell
        FEM2D::map_from_reference_cell(
            mesh.degree(), mesh.cell_type(), node_coords_v, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        // Evaluate the force function for the current element
        f_cell = f.eval_cell(k);

        // Loop over the quadrature points
        for (int q = 0; q < qpoints_rs.rows(); q++) {
            // Compute the weighted value of the force function f at the quadrature point
            double detJxWxf = (
                f_cell.dot(phi_rs(q, Eigen::placeholders::all)) * fabs(detJ_rs(q)) * qweights(q)
            );

            // Add contribution to the right-hand side vector (rhs_vec) using the basis functions
            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                rhs_vec(element(i)) += phi_rs(q, i) * detJxWxf;
            }
        }
    }
}

void PoissonProblem::assemble_robin_rhs(
    std::function<double(double, double)> b_robin,
    std::function<double(double, double)> g_robin,
    int boundary_id,
    int gauss_precision
) {
    // Declare variables for geometry, quadrature, and basis functions
    Eigen::MatrixXd node_coords, qpoints_x, phi_r, dphi_r, dphi_x;
    Eigen::VectorXd qweights, qpoints_r, detJ_r;
    Eigen::VectorXi edge_vi;

    // Retrieve quadrature points and weights using Gauss-Legendre quadrature
    FEM1D::gauss_legendre_quadrature(gauss_precision, qpoints_r, qweights);

    // Retrieve Lagrange basis functions and their derivatives at quadrature points
    FEM1D::lagrange_basis(
        mesh.degree(), qpoints_r, phi_r, dphi_r
    );

    // Initialize matrices and vectors for mapping and transformations
    qpoints_x = Eigen::MatrixXd::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorXd::Zero(qpoints_r.rows());
    dphi_x = Eigen::MatrixXd::Zero(2*qpoints_r.rows(), mesh.dofs_per_cell());

    // Extract the edges of the mesh corresponding to the boundary ID
    std::vector<int> surf_edge_inds = mesh.extract_edge_inds(boundary_id);

    // Loop over the edges of the specified boundary
    for (int si: surf_edge_inds) {
        // Retrieve the vertex indices for the current edge
        edge_vi = mesh.emat(si, Eigen::placeholders::all);
        node_coords = mesh.pmat(edge_vi, Eigen::placeholders::all);

        // Map the node coordinates of the edge to the reference cell
        FEM1D::map_from_reference_cell(
            mesh.degree(), node_coords, qpoints_r,
            phi_r, dphi_r, detJ_r, qpoints_x, dphi_x
        );

        // Loop over the quadrature points for numerical integration
        for (int q = 0; q < qpoints_r.rows(); q++) {
            // Evaluate the Robin boundary conditions at the current quadrature point
            double br = b_robin(qpoints_x(q, 0), qpoints_x(q, 1));
            double gr = g_robin(qpoints_x(q, 0), qpoints_x(q, 1));

            // Compute the coefficient c based on Robin condition (g/b)
            double cr = gr/br;

            // Calculate the weighted contribution for the right-hand side (rhs) vector
            double detJxWxCr = cr * fabs(detJ_r(q)) * qweights(q);

            // Accumulate the contribution for each basis function at the quadrature points
            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                rhs_vec(edge_vi(i)) += phi_r(q, i) * detJxWxCr;
            }
        }
    }
}

void PoissonProblem::commit_lhs_mat()
{
    lhs_mat.setFromTriplets(lhs_coeffs.begin(), lhs_coeffs.end());
}

void PoissonProblem::apply_dirichlet_bc(
    std::function<double(double, double)> bc_func, int boundary_part
) {
    // Extract the DOFs for free (interior) and fixed (boundary) parts
    free_dofs = mesh.extract_dof_inds(MESH2D::DOMAIN_ID & ~boundary_part);
    fixed_dofs = mesh.extract_dof_inds(boundary_part);

    // Extract the corresponding blocks of the LHS matrix for free and fixed DOFs
    lhs_mat_free = lhs_mat.extract_block(free_dofs, free_dofs);
    lhs_mat_fixed = lhs_mat.extract_block(free_dofs, fixed_dofs);

    // Initialize the boundary condition vector for the fixed DOFs
    bc_vec = Eigen::VectorXd::Zero(fixed_dofs.size());

    // Loop through fixed DOFs to apply the Dirichlet boundary conditions
    int i = 0;
    for (int dof: fixed_dofs) {
        double x = mesh.pmat(dof, 0);
        double z = mesh.pmat(dof, 1);
        bc_vec(i++) = bc_func(x, z); // Apply the boundary condition
    }

    // Update the RHS vector for free DOFs considering the boundary conditions
    rhs_vec_free = rhs_vec(free_dofs) - lhs_mat_fixed * bc_vec;
}

void PoissonProblem::solve_linear_system()
{
    // Initialize sparse LU solver for the linear system
    Eigen::SparseLU<Eigen::SparseMatrix<double>> sp_solver;
    sp_solver.analyzePattern(lhs_mat_free);
    sp_solver.factorize(lhs_mat_free);

    // Solve the linear system for free dofs
    sol_vec_free = sp_solver.solve(rhs_vec_free);

    // Combine the solution for free and fixed dofs into the full solution vector
    sol_vec(free_dofs) = sol_vec_free;
    sol_vec(fixed_dofs) = bc_vec;
}
FEMFunction2D PoissonProblem::solution()
{
    FEMFunction2D sol_function(mesh);
    sol_function.assign(sol_vec);
    return sol_function;
}
