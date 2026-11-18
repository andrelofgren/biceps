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
#include <fem_2d.hpp>

FEMFunction2D::FEMFunction2D(StructuredMesh &mesh) : mesh(mesh)
{
    mass_mat_is_assembled = false;
    vals = Eigen::VectorXd::Zero(mesh.nof_dofs());
}

void FEMFunction2D::assemble_mass_matrix()
{
    M = FEM2D::assemble_mass_matrix(mesh, 2*mesh.degree()+1);
    mass_mat_is_assembled = true;
}

void FEMFunction2D::set_mass_matrix(Eigen::SparseMatrix<double> &M_sp)
{
    M = M_sp;
    mass_mat_is_assembled = true;
}

Eigen::SparseMatrix<double> FEMFunction2D::mass_matrix() {
    return M;
}

void FEMFunction2D::assign(std::function<double(double, double)> func)
{
    for (int i = 0; i < mesh.nof_dofs(); i++) {
        double x = mesh.pmat(i, 0);
        double z = mesh.pmat(i, 1);
        vals(i) = func(x, z);
    }
}

void FEMFunction2D::assign(const Eigen::VectorXd &vec)
{
    if (vec.size() != mesh.nof_dofs()) {
        throw std::invalid_argument("FEMFunction2D::assign: Size mismatch vec.size() != f.mesh.nof_dofs()");
    }
    vals.noalias() = vec;
}

void FEMFunction2D::assign(const FEMFunction2D &f)
{
    if (f.vals.size() != f.mesh.nof_dofs()) {
        throw std::invalid_argument("FEMFunction2D::assign: Size mismatch f.vals.size() != f.mesh.nof_dofs()");
    }
    vals.noalias() = f.vals;
}

FEMFunction2D FEMFunction2D::diff_x_interp()
{
    Eigen::MatrixXd node_coords, qpoints_rs, qpoints_xz, phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd detJ_rs;
    Eigen::VectorXi element;
    FEMFunction2D u_diff(mesh);

    qpoints_rs = FEM2D::reference_element_points_rs(mesh.cell_type(), mesh.degree());
    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2 * qpoints_rs.rows(), mesh.dofs_per_cell());

    FEM2D::lagrange_basis(
        mesh.degree(),
        mesh.cell_type(),
        qpoints_rs,
        phi_rs, dphi_rs
    );

    Eigen::VectorXd diff_vals = Eigen::VectorXd::Zero(mesh.nof_dofs());
    Eigen::VectorXi duplicate_dof_count = Eigen::VectorXi::Zero(mesh.nof_dofs());

    for (int ci = 0; ci < mesh.cmat.rows(); ci++) {
        element = mesh.cmat.row(ci);
        node_coords = mesh.pmat(element, Eigen::all);

        FEM2D::map_from_reference_cell(
            mesh.degree(), mesh.cell_type(), node_coords, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        for (int q = 0; q < qpoints_rs.rows(); q++) {
            int w = 2*q;
            duplicate_dof_count(element(q)) += 1;
            diff_vals(element(q)) += vals(element).dot(dphi_xz(w, Eigen::all));
        }
    }

    for (int dof = 0; dof < mesh.nof_dofs(); dof++) {
        u_diff.vals(dof) = diff_vals(dof) / duplicate_dof_count(dof);
    }

    return u_diff;
}

FEMFunction2D FEMFunction2D::diff_z_interp()
{
    Eigen::MatrixXd node_coords, qpoints_rs, qpoints_xz, phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd detJ_rs;
    Eigen::VectorXi element;
    FEMFunction2D u_diff(mesh);

    qpoints_rs = FEM2D::reference_element_points_rs(mesh.cell_type(), mesh.degree());
    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2 * qpoints_rs.rows(), mesh.dofs_per_cell());

    FEM2D::lagrange_basis(
        mesh.degree(),
        mesh.cell_type(),
        qpoints_rs,
        phi_rs, dphi_rs
    );

    Eigen::VectorXd diff_vals = Eigen::VectorXd::Zero(mesh.nof_dofs());
    Eigen::VectorXi duplicate_dof_count = Eigen::VectorXi::Zero(mesh.nof_dofs());

    for (int ci = 0; ci < mesh.cmat.rows(); ci++) {
        element = mesh.cmat.row(ci);
        node_coords = mesh.pmat(element, Eigen::all);

        FEM2D::map_from_reference_cell(
            mesh.degree(), mesh.cell_type(), node_coords, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        for (int q = 0; q < qpoints_rs.rows(); q++) {
            int w = 2*q + 1;
            duplicate_dof_count(element(q)) += 1;
            diff_vals(element(q)) += vals(element).dot(dphi_xz(w, Eigen::all));
        }
    }

    for (int dof = 0; dof < mesh.nof_dofs(); dof++) {
        u_diff.vals(dof) = diff_vals(dof) / duplicate_dof_count(dof);
    }

    return u_diff;
}

FEMFunction2D FEMFunction2D::diff_x_proj()
{
    FEMFunction2D u_diff(mesh);

    int deg = mesh.degree();

    Eigen::SparseMatrix<double> M = FEM2D::assemble_mass_matrix(mesh, 2*deg + 1);
    Eigen::SparseMatrix<double> Ax = FEM2D::assemble_stiffness_x_matrix(mesh, 2*deg + 1);

    Eigen::VectorXd rhs_vec = Ax * vals;

    Eigen::SparseLU<Eigen::SparseMatrix<double>> sp_solver;
    sp_solver.analyzePattern(M);
    sp_solver.factorize(M);
    u_diff.vals = sp_solver.solve(rhs_vec);

    return u_diff;
}

FEMFunction2D FEMFunction2D::diff_z_proj()
{
    FEMFunction2D u_diff(mesh);

    int deg = mesh.degree();

    Eigen::SparseMatrix<double> M = FEM2D::assemble_mass_matrix(mesh, 2*deg + 1);
    Eigen::SparseMatrix<double> Az = FEM2D::assemble_stiffness_z_matrix(mesh, 2*deg + 1);

    Eigen::VectorXd rhs_vec = Az * vals;

    Eigen::SparseLU<Eigen::SparseMatrix<double>> sp_solver;
    sp_solver.analyzePattern(M);
    sp_solver.factorize(M);
    u_diff.vals = sp_solver.solve(rhs_vec);

    return u_diff;
}

Eigen::VectorXd FEMFunction2D::eval_cell(int ci)
{
    return vals(mesh.cmat.row(ci), Eigen::all);
}

Eigen::VectorXd FEMFunction2D::eval_edge(int ei)
{
    return vals(mesh.emat.row(ei), Eigen::all);
}

Eigen::MatrixXd FEMFunction2D::extract_vertex_subvec(int domain_id)
{
    std::vector<int> subdomain_inds = mesh.extract_vertex_dof_inds(domain_id);
    Eigen::MatrixXd vector_xzf(subdomain_inds.size(), 3);
    vector_xzf(Eigen::all, Eigen::seq(0, 1)) = mesh.pmat(subdomain_inds, Eigen::all);
    vector_xzf(Eigen::all, 2) = vals(subdomain_inds);
    return vector_xzf;
}

Eigen::MatrixXd FEMFunction2D::extract_dof_subvec(int domain_id)
{
    std::vector<int> subdomain_inds = mesh.extract_dof_inds(domain_id);
    Eigen::MatrixXd vector_xzf(subdomain_inds.size(), 3);
    vector_xzf(Eigen::all, Eigen::seq(0, 1)) = mesh.pmat(subdomain_inds, Eigen::all);
    vector_xzf(Eigen::all, 2) = vals(subdomain_inds);
    return vector_xzf;
}

// TODO: implement this!
double FEMFunction2D::operator()(double x, double z)
{
    Eigen::MatrixXd qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element_v;
    return 0.0;
}

FEMFunction2D FEMFunction2D::operator+(const FEMFunction2D &f)
{
    if (vals.size() != f.vals.size()) {
        throw std::invalid_argument("FEMFunction2D::operator+: Size mismatch vals.size() != f.vals.size()");
    }
    FEMFunction2D func(mesh);
    func.vals = this->vals + f.vals;
    return func;
}

FEMFunction2D FEMFunction2D::operator-(const FEMFunction2D &f)
{
    if (vals.size() != f.vals.size()) {
        throw std::invalid_argument("FEMFunction2D::operator-: Size mismatch vals.size() != f.vals.size()");
    }
    FEMFunction2D func(mesh);
    func.vals = this->vals - f.vals;
    return func;
}

FEMFunction2D FEMFunction2D::operator*(const FEMFunction2D &f)
{
    if (vals.size() != f.vals.size()) {
        throw std::invalid_argument("FEMFunction2D::operator*: Size mismatch vals.size() != f.vals.size()");
    }
    FEMFunction2D func(mesh);
    func.vals = this->vals * f.vals;
    return func;
}

FEMFunction2D FEMFunction2D::operator+(double val)
{
    FEMFunction2D func(mesh);
    func.vals = this->vals.array() + val;
    return func;
}

FEMFunction2D FEMFunction2D::operator-(double val)
{
    FEMFunction2D func(mesh);
    func.vals = this->vals.array() - val;
    return func;
}

FEMFunction2D FEMFunction2D::operator*(double val)
{
    FEMFunction2D func(mesh);
    func.vals = this->vals * val;
    return func;
}

double FEMFunction2D::L2_norm()
{
    return sqrt(FEM2D::inner(vals, M, vals));
}
