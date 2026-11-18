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
#include <boost/format.hpp>
#include <matplot/matplot.h>

#include <enums.hpp>
#include <pstokes_fem.hpp>
#include <free_surface_fem.hpp>

#define GRAVITY 9.8
#define ICE_DENSITY 910

// Define domain and grid parameters
double x0 = 0.0;  // Left end
double x1 = 100.0;  // Right end
double L = x1 - x0;  // Length of the domain
double H = 1.0;  // Mean height of the domain
double z0 = 0.1;  // Amplitude of surface undulation

double A = 100.0;  // Ice softness parameter
double n_i = 3.0;  // Glen exponent
double eps_reg_2 = 1e-10;  // Regularization parameter
int fssa_version = FSSA_NONE;  // No FSSA stabilization
double fssa_param = 0;  // Stabilization parameter in FSSA 

int nx = 50;  // Number of elements in x-direction
int nz = 5;  // Number of elements in z-direction
int nt = 100;  // Number of time steps
double dt = 35.0;  // Time step size
int deg_u = 2;  // Polynomial degree for velocity field
int deg_p = 1;  // Polynomial degree for pressure field
int deg_h = 1;  // Polynomial degree for height field
int gauss_precision = 5;  // Number of Gauss points in each direction per element
int max_iter = 100;  // Maximum number of iterations for solver
double stol = 1e-6;  // Convergence tolerance for solver
int cell_type = MESH2D::TRIANGLE_LEFT;  // 2D mesh cell type

double zb_expr(double x)
{
    return 0.0;
}

double zs_expr(double x)
{
    return H + z0*cos(M_PI*x/L);
}

double force_x(double x, double z)
{
    return 0.0;
}

double force_z(double x, double z)
{
    return -1e-3*ICE_DENSITY*GRAVITY;
}

int main(int argc, char *argv[])
{
    // Create structured meshes for velocity, pressure, and height fields
    StructuredMesh u_mesh_2d(nx, nz, deg_u, cell_type);
    StructuredMesh p_mesh_2d(nx, nz, deg_p, cell_type);

    // Extrude the meshes in the x and z directions
    u_mesh_2d.extrude_x(x0, x1);
    u_mesh_2d.extrude_z(zb_expr, zs_expr);
    p_mesh_2d.extrude_x(x0, x1);
    p_mesh_2d.extrude_z(zb_expr, zs_expr);

    // Extract degrees of freedom for surface nodes
    std::vector<int> sdofs_u = u_mesh_2d.extract_dof_inds(MESH2D::SURFACE_ID);
    std::vector<int> sdofs_h = u_mesh_2d.extract_vertex_dof_inds(MESH2D::SURFACE_ID);

    // Extract surface coordinates
    Eigen::MatrixXd spmat_u = u_mesh_2d.pmat(sdofs_u, Eigen::placeholders::all);
    Eigen::MatrixXd spmat_h = u_mesh_2d.pmat(sdofs_h, Eigen::placeholders::all);
    Eigen::VectorXd xs_vec = spmat_h(Eigen::placeholders::all, 0);
    Eigen::VectorXd zs_vec = spmat_h(Eigen::placeholders::all, 1);

    // Project mesh to z=0
    spmat_u(Eigen::placeholders::all, 1).array() = 0.0;
    spmat_h(Eigen::placeholders::all, 1).array() = 0.0;

    // Create 1D meshes for velocity and height on the surface
    IntervalMesh u_mesh_1d = IntervalMesh(spmat_u, deg_u);
    IntervalMesh h_mesh_1d = IntervalMesh(spmat_h, deg_h);

    // Define ids for Dirichlet boundary condition
    int ux_boundary_id = (
        MESH2D::NORTH_WEST_ID |
        MESH2D::WEST_ID |
        MESH2D::BED_ID |
        MESH2D::EAST_ID |
        MESH2D::NORTH_EAST_ID
    );
    int uz_boundary_id = MESH2D::BED_ID;

    // Initialize FEM functions for height, velocity, and accumulation
    FEMFunction1D ux_func = FEMFunction1D(u_mesh_1d);
    FEMFunction1D uz_func = FEMFunction1D(u_mesh_1d);
    FEMFunction1D h0_func = FEMFunction1D(h_mesh_1d);
    FEMFunction1D ac_func = FEMFunction1D(h_mesh_1d);

    // For L2 norm calculation
    h0_func.assemble_mass_matrix();

    // Initialize the pStokes
    pStokesProblem psp(A, n_i, eps_reg_2, force_x, force_z, u_mesh_2d, p_mesh_2d);

    // Configure BC mask (impenetrability on sides and noslip on bedrock)
    psp.ux_dirichlet_bc_mask = MESH2D::NORTH_WEST_ID | MESH2D::WEST_ID | MESH2D::BED_ID | MESH2D::EAST_ID | MESH2D::NORTH_EAST_ID;
    psp.uz_dirichlet_bc_mask = MESH2D::BED_ID;

    // Initialize the Free Surface Problem
    FreeSurfaceProblem fsp(h_mesh_1d, u_mesh_1d);

    // Plot initial surface profile
    matplot::plot(xs_vec, zs_vec)->line_width(2.0);
    matplot::hold(true);
    for (int k = 0; k < nt; k++) {
        psp.solve_nonlinear_system();

        // Clear lhs matrix and rhs vector.
        psp.reset_system();

        // Extract velocity field solutions
        Eigen::VectorXd ux_vec = psp.velocity_x().vals;
        Eigen::VectorXd uz_vec = psp.velocity_z().vals;
        // Set free surface velocity
        ux_func.vals = ux_vec(sdofs_u);
        uz_func.vals = uz_vec(sdofs_u);
        // Set initial height
        h0_func.vals = zs_vec;

        // Print surface energy and domain area
        std::cout << boost::format{"||E|| = %.16f"} %h0_func.L2_norm() << std::endl;
        std::cout << boost::format{"A = %.16f"} %u_mesh_2d.area() << std::endl;

        // Solve the free surface problem using explicit time stepping
        fsp.assemble_lhs_explicit();
        fsp.commit_lhs();
        fsp.assemble_rhs_explicit(
            h0_func, ux_func, uz_func, ac_func, dt
        );
        fsp.solve_linear_system();
        // Clear lhs matrix and rhs vector
        fsp.reset_system();

        // Update surface elevation
        zs_vec = fsp.zs_vec;

        // Update mesh with new surface elevation
        u_mesh_2d.extrude_z(zs_vec);
        p_mesh_2d.extrude_z(zs_vec);

    }

    // Plot final surface
    matplot::plot(xs_vec, zs_vec)->line_width(2.0);
    matplot::show();

    return 0;
}
