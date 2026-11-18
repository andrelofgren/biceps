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
#include <fem_1d.hpp>
#include <fem_2d.hpp>
#include <enums.hpp>

void FEM2D::map_to_reference_cell(
    int degree, 
    int cell_type,
    const Eigen::MatrixXd &node_coords,
    const Eigen::MatrixXd &qpoints_rs,
    const Eigen::MatrixXd &phi_rs,
    const Eigen::MatrixXd &grad_phi_rs,
    Eigen::VectorXd &detJ_rs_ret,
    Eigen::MatrixXd &qpoints_xz_ret,
    Eigen::MatrixXd &grad_phi_xz_ret
) {
    Eigen::Matrix2<double> dF;
    for (int i = 0; i < qpoints_rs.rows(); i++) {
        int j = 2*i;

        dF = grad_phi_rs(Eigen::seq(j, j+1), Eigen::all)*node_coords;
        detJ_rs_ret(i) = dF(0, 0)*dF(1, 1) - dF(0, 1)*dF(1, 0);
        qpoints_xz_ret(i, Eigen::all) = phi_rs(i, Eigen::all)*node_coords;
        grad_phi_xz_ret(j, Eigen::all) = (
            dF(1, 1)*grad_phi_rs(j, Eigen::all)
            -dF(0, 1)*grad_phi_rs(j+1, Eigen::all)
        );
        grad_phi_xz_ret(j+1, Eigen::all) = (
            dF(0, 0)*grad_phi_rs(j+1, Eigen::all) 
            -dF(1, 0)*grad_phi_rs(j, Eigen::all)
        );
        grad_phi_xz_ret(Eigen::seq(j, j+1), Eigen::all) *= 1.0/detJ_rs_ret(i);
    }
}

void FEM2D::lagrange_basis(
    int degree,
    int cell_type,
    Eigen::MatrixXd &qpoints_rs,
    Eigen::MatrixXd &phi_rs_ret,
    Eigen::MatrixXd &grad_phi_rs_ret
) {
    int dofs_per_cell;
    if (cell_type == MESH2D::TRIANGLE_LEFT || cell_type == MESH2D::TRIANGLE_RIGHT) {
        dofs_per_cell = (degree+2)*(degree+1) >> 1;
    } else if (cell_type == MESH2D::QUADRILATERAL) {
        dofs_per_cell = (degree+1)*(degree+1);
    } else {
        throw std::invalid_argument(
            (boost::format{"FEM2D::lagrange_basis: Unrecgonized cell_type with id '%d'"} %cell_type).str()
        );
    }

    phi_rs_ret = Eigen::MatrixXd::Zero(qpoints_rs.rows(), dofs_per_cell);
    grad_phi_rs_ret = Eigen::MatrixXd::Zero(qpoints_rs.rows() << 1, dofs_per_cell);

    for (int i = 0; i < qpoints_rs.rows(); i++) {
        int j = i << 1;
        double r = qpoints_rs(i, 0);	
        double s = qpoints_rs(i, 1);

        if (cell_type == MESH2D::QUADRILATERAL && degree == 1) {
            phi_rs_ret(i, 0) = (1-r)*(1-s);
            phi_rs_ret(i, 1) = r*(1-s);
            phi_rs_ret(i, 2) = r*s;
            phi_rs_ret(i, 3) = (1-r)*s;

            grad_phi_rs_ret(j, 0) = s-1;
            grad_phi_rs_ret(j+1, 0) = r-1;

            grad_phi_rs_ret(j, 1) = 1-s;
            grad_phi_rs_ret(j+1, 1) = -r;

            grad_phi_rs_ret(j, 2) = s;
            grad_phi_rs_ret(j+1, 2) = r;

            grad_phi_rs_ret(j, 3) = -s;
            grad_phi_rs_ret(j+1, 3) = 1-r;
        } else if (
            (cell_type == MESH2D::TRIANGLE_LEFT || cell_type == MESH2D::TRIANGLE_RIGHT)
            && degree == 1
        ) {
            phi_rs_ret(i, 0) = 1-r-s;
            phi_rs_ret(i, 1) = r;
            phi_rs_ret(i, 2) = s;

            grad_phi_rs_ret(j, 0) = -1;
            grad_phi_rs_ret(j+1, 0) = -1;

            grad_phi_rs_ret(j, 1) = 1;
            grad_phi_rs_ret(j+1, 1) = 0;

            grad_phi_rs_ret(j, 2) = 0;
            grad_phi_rs_ret(j+1, 2) = 1;
        } else if (cell_type == MESH2D::QUADRILATERAL && degree == 2) {
            phi_rs_ret(i, 0) = 4.0*(0.5-r)*(1-r)*(1-s)*(0.5-s);
            phi_rs_ret(i, 1) = 8.0*r*(1.0-r)*(1-s)*(0.5-s);
            phi_rs_ret(i, 2) = 4.0*r*(r-0.5)*(1-s)*(0.5-s);
            phi_rs_ret(i, 3) = 8.0*r*(r-0.5)*s*(1-s);
            phi_rs_ret(i, 4) = 4.0*r*(r-0.5)*s*(s-0.5);
            phi_rs_ret(i, 5) = 8.0*r*(1.0-r)*s*(s-0.5);
            phi_rs_ret(i, 6) = 4.0*(0.5-r)*(1.0-r)*s*(s-0.5);
            phi_rs_ret(i, 7) = 8.0*(0.5-r)*(1.0-r)*s*(1.0-s);
            phi_rs_ret(i, 8) = 16.0*r*(1.0-r)*s*(1.0-s);

            grad_phi_rs_ret(j, 0) = 8.0*(r-0.75)*(0.5-s)*(1.0-s);
            grad_phi_rs_ret(j+1, 0) = 8.0*(0.5-r)*(1.0-r)*(s-0.75);

            grad_phi_rs_ret(j, 1) = 16.0*(0.5-r)*(0.5-s)*(1.0-s);
            grad_phi_rs_ret(j+1, 1) = 16.0*r*(1.0-r)*(s-0.75);

            grad_phi_rs_ret(j, 2) = 8.0*(r-0.25)*(0.5-s)*(1.0-s);
            grad_phi_rs_ret(j+1, 2) = 8.0*r*(r-0.5)*(s-0.75);

            grad_phi_rs_ret(j, 3) = 16.0*(r-0.25)*s*(1.0-s);
            grad_phi_rs_ret(j+1, 3) = 16.0*r*(r-0.5)*(0.5-s);

            grad_phi_rs_ret(j, 4) = 8.0*(r-0.25)*s*(s-0.5);
            grad_phi_rs_ret(j+1, 4) = 8.0*r*(r-0.5)*(s-0.25);

            grad_phi_rs_ret(j, 5) = 16.0*(0.5-r)*s*(s-0.5);
            grad_phi_rs_ret(j+1, 5) = 16.0*r*(1.0-r)*(s-0.25);

            grad_phi_rs_ret(j, 6) = 8.0*(r-0.75)*s*(s-0.5);
            grad_phi_rs_ret(j+1, 6) = 8.0*(0.5-r)*(1.0-r)*(s-0.25);

            grad_phi_rs_ret(j, 7) = 16.0*(r-0.75)*s*(1.0-s);
            grad_phi_rs_ret(j+1, 7) = 16.0*(0.5-r)*(1.0-r)*(0.5-s);

            grad_phi_rs_ret(j, 8) = 32.0*(0.5-r)*s*(1.0-s);
            grad_phi_rs_ret(j+1, 8) = 32.0*r*(1.0-r)*(0.5-s);
        } else if (
            (cell_type == MESH2D::TRIANGLE_LEFT || cell_type == MESH2D::TRIANGLE_RIGHT)
            && degree == 2
        ) {
            phi_rs_ret(i, 0) = 2.0*(0.5-r-s)*(1.0-r-s);
            phi_rs_ret(i, 1) = 4.0*r*(1.0-r-s);
            phi_rs_ret(i, 2) = 2.0*r*(r-0.5);
            phi_rs_ret(i, 3) = 4.0*r*s;
            phi_rs_ret(i, 4) = 2.0*s*(s-0.5);
            phi_rs_ret(i, 5) = 4.0*s*(1.0-r-s);

            grad_phi_rs_ret(j, 0) = -3.0 + 4.0*(r+s);
            grad_phi_rs_ret(j+1, 0) = -3.0 + 4.0*(r+s);

            grad_phi_rs_ret(j, 1) = 4.0*(1.0-2.0*r-s);
            grad_phi_rs_ret(j+1, 1) = -4.0*r;

            grad_phi_rs_ret(j, 2) = 4.0*r-1.0;
            grad_phi_rs_ret(j+1, 2) = 0.0;

            grad_phi_rs_ret(j, 3) = 4.0*s;
            grad_phi_rs_ret(j+1, 3) = 4.0*r;

            grad_phi_rs_ret(j, 4) = 0.0;
            grad_phi_rs_ret(j+1, 4) = 4.0*s-1.0;

            grad_phi_rs_ret(j, 5) = -4.0*s;
            grad_phi_rs_ret(j+1, 5) = 4.0*(1.0-2.0*s-r); 
        } else {
            throw std::invalid_argument(
                (boost::format{"FEM2D::lagrange_basis: Unsupported combination cell_type_id=%d and degree=%d"} %cell_type %degree).str()
            );
        }
    }
}

Eigen::MatrixXd FEM2D::reference_element_points_rs(
    int cell_type,
    const int degree
) {
    // Triangle reference element is defined to have a right cut
    if (cell_type == MESH2D::TRIANGLE_LEFT)
        cell_type = MESH2D::TRIANGLE_RIGHT;

    StructuredMesh mesh(1, 1, degree, cell_type);
    Eigen::MatrixXd ref_coords = mesh.pmat(mesh.cmat.row(0), Eigen::all);
    return ref_coords;
}

void FEM2D::gauss_legendre_quadrature(
    const int precision,
    const int cell_type,
    Eigen::MatrixXd &points,
    Eigen::VectorXd &weights
) {
    points = Eigen::MatrixXd::Zero(precision*precision, 2);
    weights = Eigen::VectorXd::Zero(precision*precision);

    if (cell_type == MESH2D::QUADRILATERAL) {
        if (precision == 1) {
        points <<
            0.500000000000000000000000L, 0.500000000000000000000000L;
        weights <<
            1.000000000000000000000000L;
        } else if (precision == 2) {
        points <<
            0.211324865405187117745426L, 0.211324865405187117745426L,
            0.211324865405187117745426L, 0.788675134594812882254574L,
            0.788675134594812882254574L, 0.211324865405187117745426L,
            0.788675134594812882254574L, 0.788675134594812882254574L;
        weights <<
            0.250000000000000000000000L,
            0.250000000000000000000000L,
            0.250000000000000000000000L,
            0.250000000000000000000000L;
        } else if (precision == 3) {
        points <<
            0.112701665379258311482073L, 0.112701665379258311482073L,
            0.112701665379258311482073L, 0.500000000000000000000000L,
            0.112701665379258311482073L, 0.887298334620741688517927L,
            0.500000000000000000000000L, 0.112701665379258311482073L,
            0.500000000000000000000000L, 0.500000000000000000000000L,
            0.500000000000000000000000L, 0.887298334620741688517927L,
            0.887298334620741688517927L, 0.112701665379258311482073L,
            0.887298334620741688517927L, 0.500000000000000000000000L,
            0.887298334620741688517927L, 0.887298334620741688517927L;
        weights <<
            0.077160493827160493827160L,
            0.123456790123456790123457L,
            0.077160493827160493827160L,
            0.123456790123456790123457L,
            0.197530864197530864197531L,
            0.123456790123456790123457L,
            0.077160493827160493827160L,
            0.123456790123456790123457L,
            0.077160493827160493827160L;
        } else if (precision == 4) {
        points <<
            0.069431844202973712388027L, 0.069431844202973712388027L,
            0.069431844202973712388027L, 0.330009478207571867598667L,
            0.069431844202973712388027L, 0.669990521792428132401333L,
            0.069431844202973712388027L, 0.930568155797026287611973L,
            0.330009478207571867598667L, 0.069431844202973712388027L,
            0.330009478207571867598667L, 0.330009478207571867598667L,
            0.330009478207571867598667L, 0.669990521792428132401333L,
            0.330009478207571867598667L, 0.930568155797026287611973L,
            0.669990521792428132401333L, 0.069431844202973712388027L,
            0.669990521792428132401333L, 0.330009478207571867598667L,
            0.669990521792428132401333L, 0.669990521792428132401333L,
            0.669990521792428132401333L, 0.930568155797026287611973L,
            0.930568155797026287611973L, 0.069431844202973712388027L,
            0.930568155797026287611973L, 0.330009478207571867598667L,
            0.930568155797026287611973L, 0.669990521792428132401333L,
            0.930568155797026287611973L, 0.930568155797026287611973L;
        weights <<
            0.030250748321400501380303L,
            0.056712962962962962962963L,
            0.056712962962962962962963L,
            0.030250748321400501380303L,
            0.056712962962962962962963L,
            0.106323325752673572693771L,
            0.106323325752673572693771L,
            0.056712962962962962962963L,
            0.056712962962962962962963L,
            0.106323325752673572693771L,
            0.106323325752673572693771L,
            0.056712962962962962962963L,
            0.030250748321400501380303L,
            0.056712962962962962962963L,
            0.056712962962962962962963L,
            0.030250748321400501380303L;
        } else if (precision == 5) {
        points <<
            0.046910077030668003601187L, 0.046910077030668003601187L,
            0.046910077030668003601187L, 0.230765344947158454481843L,
            0.046910077030668003601187L, 0.500000000000000000000000L,
            0.046910077030668003601187L, 0.769234655052841545518157L,
            0.046910077030668003601187L, 0.953089922969331996398813L,
            0.230765344947158454481843L, 0.046910077030668003601187L,
            0.230765344947158454481843L, 0.230765344947158454481843L,
            0.230765344947158454481843L, 0.500000000000000000000000L,
            0.230765344947158454481843L, 0.769234655052841545518157L,
            0.230765344947158454481843L, 0.953089922969331996398813L,
            0.500000000000000000000000L, 0.046910077030668003601187L,
            0.500000000000000000000000L, 0.230765344947158454481843L,
            0.500000000000000000000000L, 0.500000000000000000000000L,
            0.500000000000000000000000L, 0.769234655052841545518157L,
            0.500000000000000000000000L, 0.953089922969331996398813L,
            0.769234655052841545518157L, 0.046910077030668003601187L,
            0.769234655052841545518157L, 0.230765344947158454481843L,
            0.769234655052841545518157L, 0.500000000000000000000000L,
            0.769234655052841545518157L, 0.769234655052841545518157L,
            0.769234655052841545518157L, 0.953089922969331996398813L,
            0.953089922969331996398813L, 0.046910077030668003601187L,
            0.953089922969331996398813L, 0.230765344947158454481843L,
            0.953089922969331996398813L, 0.500000000000000000000000L,
            0.953089922969331996398813L, 0.769234655052841545518157L,
            0.953089922969331996398813L, 0.953089922969331996398813L;
        weights <<
            0.014033587215607158988663L,
            0.028350000000000000000000L,
            0.033696268096880225779806L,
            0.028350000000000000000000L,
            0.014033587215607158988663L,
            0.028350000000000000000000L,
            0.057271351055997779282942L,
            0.068071633137687675454761L,
            0.057271351055997779282942L,
            0.028350000000000000000000L,
            0.033696268096880225779806L,
            0.068071633137687675454761L,
            0.080908641975308641975309L,
            0.068071633137687675454761L,
            0.033696268096880225779806L,
            0.028350000000000000000000L,
            0.057271351055997779282942L,
            0.068071633137687675454761L,
            0.057271351055997779282942L,
            0.028350000000000000000000L,
            0.014033587215607158988663L,
            0.028350000000000000000000L,
            0.033696268096880225779806L,
            0.028350000000000000000000L,
            0.014033587215607158988663L;
        } else {
            throw std::invalid_argument("FEM2D::gauss_legendre_quadrature: precision > 5 is not implemented for triangle elements");
        }
    } else if (cell_type == MESH2D::TRIANGLE_LEFT || cell_type == MESH2D::TRIANGLE_RIGHT) {
        if (precision == 1) {
        points <<
            0.2500000000000000L, 0.5000000000000000L;
        weights <<
            0.5000000000000000L;
        } else if (precision == 2) {
        points << 
            0.1666666666666667L, 0.2113248654051871L,
            0.6220084679281462L, 0.2113248654051871L,
            0.0446581987385205L, 0.7886751345948129L,
            0.1666666666666667L, 0.7886751345948129L;
        weights << 
            0.1971687836487032L,
            0.1971687836487032L,
            0.0528312163512968L,
            0.0528312163512968L;
        } else if (precision == 3) {
        points << 
            0.1000000000000000L, 0.1127016653792583L,
            0.4436491673103709L, 0.1127016653792583L,
            0.7872983346207417L, 0.1127016653792583L,
            0.0563508326896291L, 0.5000000000000000L,
            0.2500000000000000L, 0.5000000000000000L,
            0.4436491673103709L, 0.5000000000000000L,
            0.0127016653792583L, 0.8872983346207417L,
            0.0563508326896291L, 0.8872983346207417L,
            0.1000000000000000L, 0.8872983346207417L;

        weights << 
            0.0684643776713536L,
            0.1095430042741657L,
            0.0684643776713536L,
            0.0617283950617284L,
            0.0987654320987654L,
            0.0617283950617284L,
            0.0086961161558070L,
            0.0139137858492912L,
            0.0086961161558070L;
        } else if (precision == 4) {
        points << 
            0.0646110632135477L, 0.0694318442029737L,
            0.3070963115311591L, 0.0694318442029737L,
            0.6234718442658671L, 0.0694318442029737L,
            0.8659570925834785L, 0.0694318442029737L,
            0.0465186775265609L, 0.3300094782075719L,
            0.2211032225007380L, 0.3300094782075719L,
            0.4488872992916901L, 0.3300094782075719L,
            0.6234718442658671L, 0.3300094782075719L,
            0.0229131666764128L, 0.6699905217924281L,
            0.1089062557068338L, 0.6699905217924281L,
            0.2211032225007380L, 0.6699905217924281L,
            0.3070963115311591L, 0.6699905217924281L,
            0.0048207809894260L, 0.9305681557970262L,
            0.0229131666764128L, 0.9305681557970262L,
            0.0465186775265609L, 0.9305681557970262L,
            0.0646110632135477L, 0.9305681557970262L;
        weights << 
            0.0281503830769256L,
            0.0527752773542295L,
            0.0527752773542295L,
            0.0281503830769256L,
            0.0379971476479502L,
            0.0712356204997401L,
            0.0712356204997401L,
            0.0379971476479502L,
            0.0187158153150127L,
            0.0350877052529335L,
            0.0350877052529335L,
            0.0187158153150127L,
            0.0021003652444748L,
            0.0039376856087335L,
            0.0039376856087335L,
            0.0021003652444748L;
        } else if (precision == 5) {
        points << 
            0.0447095217036448L, 0.0469100770306680L,
            0.2199401248396786L, 0.0469100770306680L,
            0.4765449614846660L, 0.0469100770306680L,
            0.7331497981296533L, 0.0469100770306680L,
            0.9083804012656871L, 0.0469100770306680L,
            0.0360848569231881L, 0.2307653449471584L,
            0.1775127005185774L, 0.2307653449471584L,
            0.3846173275264207L, 0.2307653449471584L,
            0.5917219545342640L, 0.2307653449471584L,
            0.7331497981296533L, 0.2307653449471584L,
            0.0234550385153340L, 0.5000000000000000L,
            0.1153826724735792L, 0.5000000000000000L,
            0.2500000000000000L, 0.5000000000000000L,
            0.3846173275264207L, 0.5000000000000000L,
            0.4765449614846660L, 0.5000000000000000L,
            0.0108252201074799L, 0.7692346550528415L,
            0.0532526444285810L, 0.7692346550528415L,
            0.1153826724735792L, 0.7692346550528415L,
            0.1775127005185774L, 0.7692346550528415L,
            0.2199401248396786L, 0.7692346550528415L,
            0.0022005553270232L, 0.9530899229693319L,
            0.0108252201074799L, 0.9530899229693319L,
            0.0234550385153340L, 0.9530899229693319L,
            0.0360848569231881L, 0.9530899229693319L,
            0.0447095217036448L, 0.9530899229693319L;
        weights << 
            0.0133752705583065L,
            0.0270200993161806L,
            0.0321155735648096L,
            0.0270200993161806L,
            0.0133752705583065L,
            0.0218078024707481L,
            0.0440551079739706L,
            0.0523630592355527L,
            0.0440551079739706L,
            0.0218078024707481L,
            0.0168481340484401L,
            0.0340358165688438L,
            0.0404543209876543L,
            0.0340358165688438L,
            0.0168481340484401L,
            0.0065421975292519L,
            0.0132162430820271L,
            0.0157085739021349L,
            0.0132162430820271L,
            0.0065421975292519L,
            0.0006583166573007L,
            0.0013299006838194L,
            0.0015806945320707L,
            0.0013299006838194L,
            0.0006583166573007L;
        } else {
            throw std::invalid_argument("FEM2D::gauss_legendre_quadrature: precision > 5 is not implemented for quadrilateral elements");
        }
    } else {
        throw std::invalid_argument(
            (boost::format{"FEM2D::gauss_legendre_quadrature: Unrecgonized cell_type with id '%d'"} %cell_type).str()
        );
    }
}

Eigen::SparseMatrix<double> FEM2D::assemble_mass_matrix(
    StructuredMesh &mesh, int gp
) {
    std::vector<Eigen::Triplet<double>> mat_coeffs;

    Eigen::SparseMatrix<double> M_sp_ret = Eigen::SparseMatrix<double>(mesh.nof_dofs(), mesh.nof_dofs());

    Eigen::MatrixXd node_coords, qpoints_rs, qpoints_xz, phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element;

    FEM2D::gauss_legendre_quadrature(
        gp, mesh.cell_type(), qpoints_rs, qweights
    );

    FEM2D::lagrange_basis(
        mesh.degree(), mesh.cell_type(), qpoints_rs, phi_rs, dphi_rs
    );

    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2 * qpoints_rs.rows(), mesh.dofs_per_cell());

    Eigen::MatrixXd M_block = Eigen::MatrixXd::Zero(
        mesh.dofs_per_cell(), mesh.dofs_per_cell()
    );

    for (int k = 0; k < mesh.cmat.rows(); k++) {
        element = mesh.cmat.row(k);
        node_coords = mesh.pmat(element, Eigen::all);

        FEM2D::map_from_reference_cell(
            mesh.degree(), mesh.cell_type(), node_coords, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        for (int q = 0; q < qpoints_rs.rows(); q++) {
            double detJxW = fabs(detJ_rs(q))*qweights(q);

            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                    M_block(i, j) += phi_rs(q, i)*phi_rs(q, j)*detJxW;
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

Eigen::SparseMatrix<double> FEM2D::assemble_stiffness_matrix(
    StructuredMesh &mesh, int gp
) {
    std::vector<Eigen::Triplet<double>> mat_coeffs; 

    Eigen::SparseMatrix<double> A_sp_ret = Eigen::SparseMatrix<double>(
        mesh.nof_dofs(), mesh.nof_dofs()
    );

    Eigen::MatrixXd node_coords_v, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element;

    FEM2D::gauss_legendre_quadrature(
        gp, mesh.cell_type(), qpoints_rs, qweights
    );

    FEM2D::lagrange_basis(
        mesh.degree(),
        mesh.cell_type(),
        qpoints_rs,
        phi_rs, dphi_rs
    );

    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2 * qpoints_rs.rows(), mesh.dofs_per_cell());

    Eigen::MatrixXd A_block = Eigen::MatrixXd::Zero(
        mesh.dofs_per_cell(), mesh.dofs_per_cell()
    );

    for (int k = 0; k < mesh.cmat.rows(); k++) {
        element = mesh.cmat.row(k);
        node_coords_v = mesh.pmat(element, Eigen::all);

        FEM2D::map_from_reference_cell(
            mesh.degree(), mesh.cell_type(), node_coords_v, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        for (int q = 0; q < qpoints_rs.rows(); q++) {
            int wx = 2*q;
            int wz = wx + 1;
            double detJxW = fabs(detJ_rs(q)) * qweights(q);

            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                    A_block(i, j) += (
                        dphi_xz(wx, i)*dphi_xz(wx, j) + dphi_xz(wz, i)*dphi_xz(wz, j)
                    )*detJxW;
                }
            }
        }

        for (int i = 0; i < mesh.dofs_per_cell(); i++) {
            for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                mat_coeffs.push_back(Eigen::Triplet<double>(
                    element(i),
                    element(j),
                    A_block(i, j)
                ));
            }
        }

        A_block.setZero();
    }

    A_sp_ret.setFromTriplets(mat_coeffs.begin(), mat_coeffs.end());
    return A_sp_ret;
}

Eigen::SparseMatrix<double> FEM2D::assemble_stiffness_xx_matrix(
    StructuredMesh &mesh, int gp
) {
    std::vector<Eigen::Triplet<double>> mat_coeffs; 

    Eigen::SparseMatrix<double> A_sp_ret = Eigen::SparseMatrix<double>(
        mesh.nof_dofs(), mesh.nof_dofs()
    );

    Eigen::MatrixXd node_coords_v, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element;

    FEM2D::gauss_legendre_quadrature(
        gp, mesh.cell_type(), qpoints_rs, qweights
    );

    FEM2D::lagrange_basis(
        mesh.degree(),
        mesh.cell_type(),
        qpoints_rs,
        phi_rs, dphi_rs
    );

    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2 * qpoints_rs.rows(), mesh.dofs_per_cell());

    Eigen::MatrixXd A_block = Eigen::MatrixXd::Zero(
        mesh.dofs_per_cell(), mesh.dofs_per_cell()
    );

    for (int k = 0; k < mesh.cmat.rows(); k++) {
        element = mesh.cmat.row(k);
        node_coords_v = mesh.pmat(element, Eigen::all);

        FEM2D::map_from_reference_cell(
            mesh.degree(), mesh.cell_type(), node_coords_v, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        for (int q = 0; q < qpoints_rs.rows(); q++) {
            int wx = 2 * q;
            double detJxW = fabs(detJ_rs(q))*qweights(q);

            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                    A_block(i, j) += dphi_xz(wx, i)*dphi_xz(wx, j)*detJxW;
                }
            }
        }

        for (int i = 0; i < mesh.dofs_per_cell(); i++) {
            for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                mat_coeffs.push_back(Eigen::Triplet<double>(
                    element(i),
                    element(j),
                    A_block(i, j)
                ));
            }
        }

        A_block.setZero();
    }

    A_sp_ret.setFromTriplets(mat_coeffs.begin(), mat_coeffs.end());
    return A_sp_ret;
}

Eigen::SparseMatrix<double> FEM2D::assemble_stiffness_zz_matrix(
    StructuredMesh &mesh, int gp
) {
    std::vector<Eigen::Triplet<double>> mat_coeffs;

    Eigen::SparseMatrix<double> A_sp_ret = Eigen::SparseMatrix<double>(
        mesh.nof_dofs(), mesh.nof_dofs()
    );

    Eigen::MatrixXd node_coords_v, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element;

    FEM2D::gauss_legendre_quadrature(
        gp, mesh.cell_type(), qpoints_rs, qweights
    );

    FEM2D::lagrange_basis(
        mesh.degree(),
        mesh.cell_type(),
        qpoints_rs,
        phi_rs, dphi_rs
    );

    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2 * qpoints_rs.rows(), mesh.dofs_per_cell());

    Eigen::MatrixXd A_block = Eigen::MatrixXd::Zero(
        mesh.dofs_per_cell(), mesh.dofs_per_cell()
    );

    for (int k = 0; k < mesh.cmat.rows(); k++) {
        element = mesh.cmat.row(k);
        node_coords_v = mesh.pmat(element, Eigen::all);

        FEM2D::map_from_reference_cell(
            mesh.degree(), mesh.cell_type(), node_coords_v, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        for (int q = 0; q < qpoints_rs.rows(); q++) {
            int wz = 2*q + 1;
            double detJxW = fabs(detJ_rs(q))*qweights(q);

            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                    A_block(i, j) += dphi_xz(wz, i)*dphi_xz(wz, j)*detJxW;
                }
            }
        }

        for (int i = 0; i < mesh.dofs_per_cell(); i++) {
            for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                mat_coeffs.push_back(Eigen::Triplet<double>(
                    element(i),
                    element(j),
                    A_block(i, j)
                ));
            }
        }

        A_block.setZero();
    }

    A_sp_ret.setFromTriplets(mat_coeffs.begin(), mat_coeffs.end());

    return A_sp_ret;
}

Eigen::SparseMatrix<double> FEM2D::assemble_stiffness_x_matrix(
    StructuredMesh &mesh, int gp
) {
    std::vector<Eigen::Triplet<double>> mat_coeffs;

    Eigen::SparseMatrix<double> Ax_sp_ret = Eigen::SparseMatrix<double>(
        mesh.nof_dofs(), mesh.nof_dofs()
    );

    Eigen::MatrixXd node_coords_v, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element;

    FEM2D::gauss_legendre_quadrature(
        gp, mesh.cell_type(), qpoints_rs, qweights
    );

    FEM2D::lagrange_basis(
        mesh.degree(),
        mesh.cell_type(),
        qpoints_rs,
        phi_rs, dphi_rs
    );

    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2 * qpoints_rs.rows(), mesh.dofs_per_cell());

    Eigen::MatrixXd Ax_block = Eigen::MatrixXd::Zero(
        mesh.dofs_per_cell(), mesh.dofs_per_cell()
    );

    for (int k = 0; k < mesh.cmat.rows(); k++) {
        element = mesh.cmat.row(k);
        node_coords_v = mesh.pmat(element, Eigen::all);

        FEM2D::map_from_reference_cell(
            mesh.degree(), mesh.cell_type(), node_coords_v, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        for (int q = 0; q < qpoints_rs.rows(); q++) {
            int wx = 2*q;
            double detJxW = fabs(detJ_rs(q))*qweights(q);

            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                    Ax_block(i, j) += phi_rs(q, i)*dphi_xz(wx, j)*detJxW;
                }
            }
        }

        for (int i = 0; i < mesh.dofs_per_cell(); i++) {
            for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                mat_coeffs.push_back(Eigen::Triplet<double>(
                    element(i),
                    element(j),
                    Ax_block(i, j)
                ));
            }
        }

        Ax_block.setZero();
    }

    Ax_sp_ret.setFromTriplets(mat_coeffs.begin(), mat_coeffs.end());

    return Ax_sp_ret;
}

Eigen::SparseMatrix<double> FEM2D::assemble_stiffness_z_matrix(
    StructuredMesh &mesh, int gp
) {
    std::vector<Eigen::Triplet<double>> mat_coeffs;

    Eigen::SparseMatrix<double> Az_sp_ret = Eigen::SparseMatrix<double>(
        mesh.nof_dofs(), mesh.nof_dofs()
    );

    Eigen::MatrixXd node_coords_v, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element;

    FEM2D::gauss_legendre_quadrature(
        gp, mesh.cell_type(), qpoints_rs, qweights
    );

    FEM2D::lagrange_basis(
        mesh.degree(),
        mesh.cell_type(),
        qpoints_rs,
        phi_rs, dphi_rs
    );

    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2 * qpoints_rs.rows(), mesh.dofs_per_cell());

    Eigen::MatrixXd Az_block = Eigen::MatrixXd::Zero(
        mesh.dofs_per_cell(), mesh.dofs_per_cell()
    );

    for (int k = 0; k < mesh.cmat.rows(); k++) {
        element = mesh.cmat.row(k);
        node_coords_v = mesh.pmat(element, Eigen::all);

        FEM2D::map_from_reference_cell(
            mesh.degree(), mesh.cell_type(), node_coords_v, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        for (int q = 0; q < qpoints_rs.rows(); q++) {
            int wz = 2*q + 1;
            double detJxW = fabs(detJ_rs(q)) * qweights(q);

            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                    Az_block(i, j) += phi_rs(q, i)*dphi_xz(wz, j)*detJxW;
                }
            }
        }

        for (int i = 0; i < mesh.dofs_per_cell(); i++) {
            for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                mat_coeffs.push_back(Eigen::Triplet<double>(
                    element(i),
                    element(j),
                    Az_block(i, j)
                ));
            }
        }
        Az_block.setZero(); // Reset local matrix for the next element
    }

    Az_sp_ret.setFromTriplets(mat_coeffs.begin(), mat_coeffs.end());

    return Az_sp_ret;
}

Eigen::MatrixXd FEM2D::assemble_extension_matrix(
    StructuredMesh &mesh,
    std::function<double(double, double)> force,
    int gp
) {
    if (mesh.cell_type() != MESH2D::QUADRILATERAL)
        throw std::invalid_argument("FEM2D::assemble_extension_matrix: Unsupported cell type: Only QUADRILATERAL is supported.");

    double x0 = mesh.pmat.col(0).minCoeff();
    double x1 = mesh.pmat.col(0).maxCoeff();
    double Lx = x1 - x0;

    int nx = mesh.nx();
    int nz = mesh.nz();

    Eigen::MatrixXd extension_mat = Eigen::MatrixXd::Zero(mesh.nof_dofs(), nx+1);

    Eigen::MatrixXd node_coords_v, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorXd qweights, detJ_rs;
    Eigen::VectorXi element_v;

    FEM2D::gauss_legendre_quadrature(
        gp, mesh.cell_type(), qpoints_rs, qweights
    );

    FEM2D::lagrange_basis(
        mesh.degree(), mesh.cell_type(), qpoints_rs, phi_rs, dphi_rs
    );

    detJ_rs = Eigen::VectorXd::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixXd::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixXd::Zero(2*qpoints_rs.rows(), mesh.dofs_per_cell());

    int k = 0;
    for (int kr = 0; kr < nz; kr++) {
        for (int kc = 0; kc < nx; kc++) {
            element_v = mesh.cmat.row(k);
            node_coords_v = mesh.pmat(element_v, Eigen::all);

            FEM2D::map_from_reference_cell(
                mesh.degree(), mesh.cell_type(), node_coords_v, qpoints_rs,
                phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
            );

            for (int q = 0; q < qpoints_rs.rows(); q++) {
                double Wxf = force(
                    qpoints_xz(q, 0), qpoints_xz(q, 1)
                )*qweights(q)*Lx/(nx*nz);
                double r = qpoints_rs(q, 0);

                for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                    extension_mat(element_v(i), kc) += phi_rs(q, i)*(1 - r)*Wxf;
                    extension_mat(element_v(i), kc + 1) += phi_rs(q, i)*r * Wxf;
                }
            }
            k++;
        }
    }

    return extension_mat;
}

double FEM2D::inner(
    const Eigen::VectorXd &u_vec,
    const Eigen::SparseMatrix<double> &M,
    const Eigen::VectorXd &v_vec
) {
    return u_vec.dot(M*v_vec);
}
