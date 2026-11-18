#include <time_integrator.hpp>

TimeIntegrator::TimeIntegrator(
    pStokesProblem &psp, FreeSurfaceProblem &fsp
) : psp(psp), fsp(fsp)
{
    // Empty constructor
}

Eigen::VectorXd TimeIntegrator::step_explicit(double dt)
{
    // Solve the pStokes problem
    psp.solve_nonlinear_system();
    // Reset the linear system
    psp.reset_system();

    // Allocate FEM functions
    FEMFunction1D
        h0_fem_func(fsp.h_mesh),
        ux_fem_func(fsp.u_mesh),
        uz_fem_func(fsp.u_mesh),
        ac_fem_func(fsp.h_mesh);

    // Extract velocities
    FEMFunction2D ux_fem_func_p2 = psp.velocity_x();
    FEMFunction2D uz_fem_func_p2 = psp.velocity_z();
    h0_fem_func.vals =
        ux_fem_func_p2.extract_vertex_subvec(MESH2D::SURFACE_ID)(Eigen::placeholders::all, 1);
    ux_fem_func.vals =
        ux_fem_func_p2.extract_dof_subvec(MESH2D::SURFACE_ID)(Eigen::placeholders::all, 2);
    uz_fem_func.vals =
        uz_fem_func_p2.extract_dof_subvec(MESH2D::SURFACE_ID)(Eigen::placeholders::all, 2);

    // Assemble the free-surface problem
    fsp.assemble_lhs_explicit();
    fsp.commit_lhs();
    fsp.assemble_rhs_explicit(
        h0_fem_func, ux_fem_func, uz_fem_func, ac_fem_func, dt
    );
    // Solve the free-surface problem
    fsp.solve_linear_system();
    // Reset the linear system
    fsp.reset_system();

    // Calculate change in surface heights
    Eigen::VectorXd dh_vec = fsp.zs_vec - h0_fem_func.vals;
    return dh_vec;
}

Eigen::VectorXd TimeIntegrator::step_simplicit(double dt)
{
    // Solve the pStokes problem
    psp.solve_nonlinear_system();
    // Reset the linear system
    psp.reset_system();

    // Allocate FEM functions
    FEMFunction1D
        h0_fem_func(fsp.h_mesh),
        ux_fem_func(fsp.u_mesh),
        uz_fem_func(fsp.u_mesh),
        ac_fem_func(fsp.h_mesh);

    // Extract velocities
    FEMFunction2D ux_fem_func_p2 = psp.velocity_x();
    FEMFunction2D uz_fem_func_p2 = psp.velocity_z();
    h0_fem_func.vals =
        ux_fem_func_p2.extract_vertex_subvec(MESH2D::SURFACE_ID)(Eigen::placeholders::all, 1);
    ux_fem_func.vals =
        ux_fem_func_p2.extract_dof_subvec(MESH2D::SURFACE_ID)(Eigen::placeholders::all, 2);
    uz_fem_func.vals =
        uz_fem_func_p2.extract_dof_subvec(MESH2D::SURFACE_ID)(Eigen::placeholders::all, 2);

    // Assemble the free-surface problem
    fsp.assemble_lhs_simplicit(ux_fem_func, dt);
    fsp.commit_lhs();
    fsp.assemble_rhs_simplicit(
        h0_fem_func, uz_fem_func, ac_fem_func, dt
    );
    // Solve the free-surface problem
    fsp.solve_linear_system();
    // Reset the linear system
    fsp.reset_system();

    // Calculate change in surface heights
    Eigen::VectorXd dh_vec = fsp.zs_vec - h0_fem_func.vals;
    return dh_vec;
}

void TimeIntegrator::extrude_mesh_z(const Eigen::VectorXd &zs_vec)
{
    psp.u_mesh.extrude_z(zs_vec);
    psp.p_mesh.extrude_z(zs_vec);
}
