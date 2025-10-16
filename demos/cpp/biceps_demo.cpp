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
    #include <enums.hpp>
    #include <pstokes_fem.hpp> 
    #include <free_surface_fem.hpp>
    #include <boost/format.hpp>
    #include <matplot/matplot.h>
    #include <fstream>
    #include <iomanip>


    #define GRAVITY 9.8
    #define ICE_DENSITY 910

    bool higherorder = false;
    bool higherorderFSonly = true;
    bool cheat = true;


    // Define domain and grid parameters
    FloatType x0 = 0.0;  // Left end
    FloatType x1 = 100.0;  // Right end
    FloatType L = x1 - x0;  // Length of the domain
    FloatType H = 1.0;  // Mean height of the domain
    FloatType z0 = 0.1;  // Amplitude of surface undulation

    FloatType eta = 1e12 * PA_TO_MPA / SEC_PER_YEAR;
    FloatType A = 0.5/eta;  // Ice softness parameter
    FloatType n_i = 1.0;  // Glen exponent
    FloatType eps_reg_2 = 1e-10;  // Regularization parameter
    int fssa_version = FSSA_NONE; //FSSA_VERTICAL;//FSSA_VERTICAL; // FSSA_NONE; FSSA_NORMAL;  // No FSSA stabilization
    FloatType fssa_theta = 0.0;  // Stabilization parameter in FSSA 
    FloatType OriginalTheta = fssa_theta;
    int implmax = 100;

    int nx = 50;  // Number of elements in x-direction
    int nz = 5;  // Number of elements in z-direction
    FloatType FinalTime=20.0;
    FloatType dt = 20;  // Time step size
    int nt = (FinalTime/(1000*dt))*1000;  // Number of time steps

    int deg_u = 2;  // Polynomial degree for velocity field
    int deg_p = 1;  // Polynomial degree for pressure field
    int deg_h = 1;  // Polynomial degree for height field
    int gauss_precision = 5;  // Number of Gauss points in each direction per element
    FloatType stol = 1e-6;  // Convergence tolerance for solver
    int cell_type = MESH2D::TRIANGLE_LEFT;  // 2D mesh cell type

    FloatType zsdiffold = 0.0;

    FloatType zb_expr(FloatType x)
    {
        return 0.0;
    }

    FloatType zs_expr(FloatType x)
    {
        return H + z0*COS_FUNC(PI_CONST*x/L);
    }

    FloatType force_x(FloatType x, FloatType z)
    {
        return 0.0;
    }

    FloatType force_z(FloatType x, FloatType z)
    {
        return -1e-3*ICE_DENSITY*GRAVITY;
    }

    int main(int argc, char *argv[])
    {

        std::cout << "Number of steps: ";
        std::cout << nt;
        std::cout << "\n";


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
        Eigen::MatrixX<FloatType> spmat_u = u_mesh_2d.pmat(sdofs_u, Eigen::all);
        Eigen::MatrixX<FloatType> spmat_h = u_mesh_2d.pmat(sdofs_h, Eigen::all);
        Eigen::VectorX<FloatType> xs_vec = spmat_h(Eigen::all, 0);
        Eigen::VectorX<FloatType> zs_vec = spmat_h(Eigen::all, 1);

        // Project mesh to z=0
        spmat_u(Eigen::all, 1).array() = 0.0;
        spmat_h(Eigen::all, 1).array() = 0.0;

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

        FEMFunction1D ux_funcOld = FEMFunction1D(u_mesh_1d);
        FEMFunction1D uz_funcOld = FEMFunction1D(u_mesh_1d);
        FEMFunction1D h0_funcOld = FEMFunction1D(h_mesh_1d);
        FEMFunction1D ac_funcOld = FEMFunction1D(h_mesh_1d);

        // For L2 norm calculation
        h0_func.assemble_mass_matrix();

        // Initialize the pStokes
        pStokesProblem psp(A, n_i, eps_reg_2, force_x, force_z, u_mesh_2d, p_mesh_2d);
        psp.fssa_version = fssa_version; //nytt
        // Configure BC mask (impenetrability on sides and noslip on bedrock)
        psp.ux_dirichlet_bc_mask = MESH2D::NORTH_WEST_ID | MESH2D::WEST_ID | MESH2D::BED_ID | MESH2D::EAST_ID | MESH2D::NORTH_EAST_ID;
        psp.uz_dirichlet_bc_mask = MESH2D::BED_ID;

        // Initialize the Free Surface Problem
        FreeSurfaceProblem fsp(h_mesh_1d, u_mesh_1d);

        Eigen::VectorXd OldFSSA=Eigen::VectorXd::Zero(psp.w_vec.size());
        Eigen::VectorXd OldFSSAcheat=Eigen::VectorXd::Zero(psp.w_vec.size());
        Eigen::VectorXd oldw_vec=Eigen::VectorXd::Zero(psp.w_vec.size());   

        // Plot initial surface profile
      //  matplot::plot(xs_vec, zs_vec)->line_width(4.0);
      //  matplot::hold(true);
      //  matplot::show();

        std::cout << "nt = " << nt << "\n";
        int itercount = 0;

        for (int k = 0; k < nt; k++) {
           std::cout << "k = " << k << "\n";
           std::cout << "---------------\n";

          FloatType zsdiff = 100;

          int impl = 0;
          while (zsdiff > 1e-9 and impl<implmax) {
          //for (int impl = 0; impl < implmax; impl++) {
            std::cout << "impl = " << impl << "\n";

          if (cheat) {
                if ( implmax>1 and fssa_version != FSSA_NONE) {
                    psp.reset_lhs();
                    psp.fssa_param=-dt*fssa_theta; 
                    psp.assemble_fssa_vertical_block(); //pushing stuff into vector
                    psp.commit_lhs_mat(); //here elements are inserted in matrix based on matrix
                    Eigen::SparseMatrix<FloatType> C(psp.lhs_mat);
                    OldFSSAcheat = C*oldw_vec; //
                    //std::cout << "old fssa  \n: ";
                    //std::cout << OldFSSA;
                    psp.reset_lhs(); 
                   // std::cout << "updating old FSSA";

                }
            }


            // p-stokes
            // Assemble the system
            psp.assemble_stress_block();
            psp.assemble_incomp_block();


            // old HO reynolds transport theorem stuff 
            //if (higherorder and k>0) {
             //       psp.fssa_param=dt*fssa_theta*(1.0/2.0); }
             //   else {psp.fssa_param=dt*fssa_theta;}

           //std::cout << "adding new FSSA term with fssa_theta=" << fssa_theta;

            //if (impl==1) {fssa_theta=0;}
            //if (impl>3) {fssa_theta=0.0;}
            //if (impl>0) {fssa_theta=+1.0;}
            //if (impl==0) {fssa_theta=10;}
            //fssa_theta=fssa_theta*0.7;

            // New FSSA term
            if (fssa_version != FSSA_NONE){
                psp.fssa_param=dt*fssa_theta;
                psp.assemble_fssa_vertical_block(); //pushing stuff into vector
            }
            // Call commit to insert elements to system matrix
            psp.commit_lhs_mat(); //here elements are inserted in matrix based on matrix
            psp.assemble_rhs_vec();


  

            if (impl>0) {
            //std::cout << "subtracting old FSSA from FSSA";
                if (cheat) {
                    psp.rhs_vec = psp.rhs_vec- fssa_theta*OldFSSAcheat; 
                }
                else {
                    psp.rhs_vec = psp.rhs_vec- fssa_theta*OldFSSA; 
                }//OriginalTheta*(zsdiff/origdiff);
            }
       

            //std::cout << psp.rhs_vec;

            // Old higher order reynolds transport theorem stuff
            //if (higherorder and fssa_version != FSSA_NONE and k>0) {
            //psp.rhs_vec = psp.rhs_vec + OldFSSA; 
            //}

            // Apply bcs by modifying the system matrix
            psp.apply_zero_dirichlet_bc();

            // Solve for velocity and pressure
            psp.solve_linear_system();
            // Clear lhs matrix and rhs vector.
            psp.reset_system();

            if (cheat) {
                oldw_vec=psp.w_vec;
            }

            // Fixar FSSA k-1 termen for next timestep
            if ( implmax>1 and fssa_version != FSSA_NONE) {
                psp.reset_lhs();
                psp.fssa_param=-dt*1.0; 
                psp.assemble_fssa_vertical_block(); //pushing stuff into vector
                psp.commit_lhs_mat(); //here elements are inserted in matrix based on matrix
                Eigen::SparseMatrix<FloatType> C(psp.lhs_mat);
                OldFSSA = C*psp.w_vec; //
                //std::cout << "old fssa  \n: ";
                //std::cout << OldFSSA;
                psp.reset_lhs(); 
               // std::cout << "updating old FSSA";

            }
            

         
            // save values from old time-step
            if (impl==0 ){
                ux_funcOld.vals = ux_func.vals;
                uz_funcOld.vals = uz_func.vals;
                h0_funcOld.vals = h0_func.vals;
                h0_func.vals = zs_vec;
               // std::cout << "updating u^k,h^k,h^k-1 \n";
            }

            // Update U

            Eigen::VectorX<FloatType> ux_vec = psp.velocity_x().vals;
            Eigen::VectorX<FloatType> uz_vec = psp.velocity_z().vals;
            // Set free surface velocity
            ux_func.vals = ux_vec(sdofs_u);
            uz_func.vals = uz_vec(sdofs_u);
            // Set old height



            //std::cout << "ux new  \n: ";
            //std::cout << ux_func.vals;
            //std::cout << "ux old \n: ";
            //std::cout << ux_funcOld.vals;
            //std::cout << "ux new - ux old \n: ";
            //std::cout << ux_func.vals - ux_funcOld.vals;


            // Print surface energy and domain area
            //std::cout << boost::format{"||E|| = %.16f"} %h0_func.L2_norm() << std::endl;
            //std::cout << boost::format{"A = %.16f"} %u_mesh_2d.area() << std::endl;

            // Solve the free surface problem using explicit time stepping
            if (higherorderFSonly and k>0) {
                //fsp.assemble_lhs_BDF2(ux_func,dt);
                fsp.assemble_lhs_crankish(ux_func,dt);
                //fsp.assemble_lhs_explicit();

            }
            else {
                //fsp.assemble_lhs_explicit();
                fsp.assemble_lhs_simplicit(ux_func,dt);
                }


            fsp.commit_lhs();
            


            
            if (higherorderFSonly and k>0) {
                //fsp.assemble_rhs_Multistep(
                 //   h0_func, ux_func, uz_func, ac_func, h0_funcOld, ux_funcOld, uz_funcOld, ac_funcOld, 3.0/2.0, -1.0/2.0,dt
               // );
                //fsp.assemble_rhs_BDF2(
                 //   h0_func, ux_func, uz_func, ac_func, h0_funcOld,dt);
                //fsp.assemble_rhs_crankish(h0_func, ux_func, uz_func, ac_func, dt);
                fsp.assemble_rhs_CN(h0_func, ux_func, uz_func, ac_func,ux_funcOld,uz_funcOld, ac_funcOld, dt);

            }
            else {
               // fsp.assemble_rhs_explicit(
               // h0_func, ux_func, uz_func, ac_func, dt);
                fsp.assemble_rhs_simplicit(
                    h0_func, uz_func, ac_func,dt);
               
            }

            fsp.solve_linear_system();
            // Clear lhs matrix and rhs vector
            fsp.reset_system();

            // Update surface elevation
            zsdiffold = zsdiff;
            //zsdiff=(zs_vec - fsp.zs_vec).norm();
            zsdiff = std::sqrt(
            ((zs_vec - fsp.zs_vec).array().square() 
            / (zs_vec.array() - 1.0).square()).mean()
        );
           // zsdiff = np.sqrt(np.mean((zs_vec - fsp.zs_vec)**2/(zs_vec-np.ones_like(zs_vec))**2))

            std::cout << "h diff = " << zsdiff << "\n";

         
            if (zsdiff > zsdiffold and impl > 4) {
            std::cout << "breaking out  "  << "\n";
                break;}

            zs_vec = fsp.zs_vec;


            //Plot final surface

           // matplot::plot(xs_vec, h0_func.vals)->line_width(2.0).line_style("--");
           // matplot::show();

           // matplot::plot(xs_vec, zs_vec)->line_width(2.0);
           // matplot::show();

            // Update mesh with new surface elevation
            u_mesh_2d.extrude_z(zs_vec);
            p_mesh_2d.extrude_z(zs_vec);

            impl++;
            itercount++;
        }

        //
        //Print out surface
        std::ofstream out("zs_vec.txt");
        out << std::setprecision(17); 
        for (auto z : zs_vec) {
            out << z << "\n";
        }
        out.close();

        std::cout << "Tot NOF iterations: " << itercount << "\n";

        //Print out result
        std::ofstream out2("itercount.txt");
        out2 << std::setprecision(17); 
        out2 << itercount/nt << "\n";
        out2.close();

        //Plot final surface
        // matplot::plot(xs_vec, zs_vec)->line_width(2.0);
        // matplot::show();
        } //end time
        return 0;
        
    }
