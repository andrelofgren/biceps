#pragma once
#include <pstokes_fem.hpp>
#include <free_surface_fem.hpp>
#include <Eigen/Dense>

class TimeIntegrator {
private:

public:
    pStokesProblem &psp;
    FreeSurfaceProblem &fsp;

    TimeIntegrator(pStokesProblem &psp, FreeSurfaceProblem &fsp);

    Eigen::VectorXd step_explicit(double dt);
    Eigen::VectorXd step_simplicit(double dt);

    void extrude_mesh_z(const Eigen::VectorXd &zs_vec);
};
