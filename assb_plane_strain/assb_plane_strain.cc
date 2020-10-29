/**
 * Copyright (C) 2020 by Sebastian Stark
 *
 * Plane strain example according to Sect. 5.1 of manuscript "A unified approach to standard dissipative continua with application to electrochemomechanically coupled problems"
 * For details regarding the physical situation simulated with this program, see this manuscript.
 */

#include <iostream>
#include <fstream>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>

#include <galerkin_tools/assembly_helper.h>

#include <incremental_fe/fe_model.h>
#include <incremental_fe/scalar_functionals/omega_lib.h>
#include <incremental_fe/scalar_functionals/psi_lib.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

// Define a ramp function (used to ramp up time-integrated current during first loading step)
template<unsigned int spacedim>
class RampFunction : public Function<spacedim>
{
private:

	/**
	 * ramp rate
	 */
	const double
	r;

public:
	RampFunction(const double r)
	:
	Function<spacedim>(1),
	r(r)
	{}

	double
	value(	const Point<spacedim>&	/*location*/,
			const unsigned int		/*component*/)
	const
	{
		return r * this->get_time();
	}
};

// the main program
int main()
{

/********************
 * parameters, etc. *
 ********************/

	const unsigned int spacedim = 2;	// spatial dimensionality; this does only work for spacedim == 2


	// quantities used for normalization (in SI units)
	const double c_ast = 10000.0;
	const double D_ast = 1e-16;
	const double R_ast = 8.3144;
	const double T_ast = 293.15;
	const double F_ast = 96485.33;
	const double L_ast = 1e-6;

	// parameters (note: geometrical parameters must be consistent with mesh file, which is read in later)
	const double L = 6e-6 / L_ast;							// width of domain
	const double B = 1.5e-6 / L_ast;						// height of computational domain

	const double c_Li_ref = 47500.0 / c_ast;
	const double c_Lip_ref = 750.0 / c_ast;
	const double c_LiX_ref = 750.0 / c_ast;

	const double lambda_se = 5e6 / (R_ast * T_ast * c_ast);
	const double mu_se = 5e6 / (R_ast * T_ast * c_ast);
	const double lambda_ap = 50.6e9 / (R_ast * T_ast * c_ast);
	const double mu_ap = 80e9 / (R_ast * T_ast * c_ast);
	const double deps_Li = -0.04;
	const double deps_LiX = 0.2;
	const double c_V_max = 50000.0 / c_ast;
	const double dmu_ap = 70000.0 / (R_ast * T_ast);
	const double D_Li = 5e-16 / D_ast;
	const double D_Lip = 2.5e-13 / D_ast;
	const double D_X = 3.0e-13 / D_ast;
	const double i_0_se_ap = 10.0 * L_ast / (F_ast * c_ast * D_ast);
	const double i_0_se_Li = 10.0 * L_ast / (F_ast * c_ast * D_ast);
	const double beta_se_ap = 0.5;
	const double beta_se_Li = 0.5;
	const double eta_bar_Li = 350000.0 / (R_ast * T_ast);

	const double R = 8.3144 / R_ast;
	const double F = 96485.33 / F_ast;
	const double T = 293.15 / T_ast;

	// loading
	const double dt = 600.0 * D_ast / (L_ast * L_ast);							// duration of loading steps
	const double j_ap_bar = -1e-17 / (F_ast * c_ast * L_ast * L_ast * D_ast);	// constant current charging current
	const double R_el = 4.0 * F_ast / (R_ast * T_ast) / fabs(j_ap_bar);			// electrical resistance (determined such that current equals constant current charging current in magnitude if voltage is 4 V)

	// numerical parameters
	const double eps_chemical = 1e-4;				// numerical parameter for regularization of chemical potential
	const unsigned int n_refinements_sing_edge = 3;	// number of refinements at edge with stress singularity
	const double alpha = 0.5;						// time integration parameter alpha
	const unsigned int method = 2;					// time integration method (0: Miehe's method, 1: alpha family, 2: modified alpha family)
	const unsigned N_1 = 20;						// nominal number of time steps for first loading step (automatic time stepping adjusts if necessary)
	const unsigned int degree = 1;					// degree of approximation of finite elements
	const unsigned int cell_divisions = degree;		// cell divisions for output
	const unsigned int solver_sym = 0;				// solver for method != 1: 0 - PARDISO, 1 - MA57, else - UMFPACK
	const unsigned int solver_unsym = 0;			// solver for method == 1: 0 - PARDISO, else - UMFPACK

	// mappings
	MappingQGeneric<spacedim, spacedim> mapping_domain(degree);			// FE mapping on domain
	MappingQGeneric<spacedim-1, spacedim> mapping_interface(degree);	// FE mapping on interfaces

	// global data object, used to transfer global data (like time step information) between different potential contributions and to define parameters for the Newton-Raphson algorithm, etc.
	GlobalDataIncrementalFE<spacedim> global_data;

	// define some parameters for the problem solution
	global_data.set_compute_sparsity_pattern(1);
	global_data.set_max_iter(20);
	global_data.set_max_cutbacks(1);
	global_data.set_threshold_residual(1e-12);
	global_data.set_perform_line_search(false);
	global_data.set_scale_residual(false);

/*****************************************************
 * grid, assignment of domain and interface portions *
 *****************************************************/

	// read in the mesh file
	// the mesh already contains the assignment of the domain portions by material id's:
	// 0 - solid electrolyte
	// 1 - active particles
	// attention: the region occupied by the mesh is -0.5 * L <= X <= 0.5 * L, 0 <= Y <= B (i.e., a different coordinate system is used then in the manuscript)
	Triangulation<spacedim> tria_domain;
	GridIn<spacedim> grid_in;
	ifstream input_file("tria_domain_2d.vtk");
	grid_in.attach_triangulation(tria_domain);
	grid_in.read_vtk(input_file);
	input_file.close();

	// triangulation system and interface definition
	// 0 - Sigma_se_Li
	// 2 - Sigma_se_Z=L
	// 3 - Sigma_ap_Z=L
	// 4 - Sigma_se_Y=0
	// 5 - Sigma_ap_Y=0
	// 6 - Sigma_se_Y=B
	// 7 - Sigma_ap_Y=B
	// 8 - Sigma_se_ap
	dealii::GalerkinTools::TriangulationSystem<spacedim> tria_system(tria_domain);

	for(const auto& cell : tria_domain.cell_iterators_on_level(0))
	{
		for(unsigned int face = 0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
		{
			if(cell->face(face)->at_boundary())
			{
				if(cell->face(face)->center()[0] < -L * 0.5 + 1e-12)
				{
					tria_system.add_interface_cell(cell, face, 0);
				}
				else if(cell->face(face)->center()[0] > L * 0.5 - 1e-12)
				{
					if(cell->material_id() == 0)
						tria_system.add_interface_cell(cell, face, 2);
					else
						tria_system.add_interface_cell(cell, face, 3);
				}
				else if(cell->face(face)->center()[1] < 1e-12)
				{
					if(cell->material_id() == 0)
						tria_system.add_interface_cell(cell, face, 4);
					else
						tria_system.add_interface_cell(cell, face, 5);
				}
				else if(cell->face(face)->center()[1] > B - 1e-12)
				{
					if(cell->material_id() == 0)
						tria_system.add_interface_cell(cell, face, 6);
					else
						tria_system.add_interface_cell(cell, face, 7);
				}
			}
			else
			{
				if( (cell->material_id() == 0) && (cell->neighbor(face)->material_id() == 1) )
					tria_system.add_interface_cell(cell, face, 8);
			}
		}
	}

	// attach manifolds, so that curved interface of active particles is correctly represented upon mesh refinement
	SphericalManifold<spacedim> spherical_manifold_domain = SphericalManifold<spacedim>(Point<spacedim>(B, 0.0));
	SphericalManifold<spacedim-1, spacedim> spherical_manifold_interface = SphericalManifold<spacedim-1, spacedim>(Point<spacedim>(B, 0.0));
	FlatManifold<spacedim> flat_manifold_domain;
	FlatManifold<spacedim-1, spacedim> flat_manifold_interface;
	tria_domain.set_manifold(1, spherical_manifold_domain);
	tria_domain.set_manifold(0, flat_manifold_domain);
	tria_system.set_interface_manifold(1, spherical_manifold_interface);
	tria_system.set_interface_manifold(0, flat_manifold_interface);
	tria_system.close();

	// mesh refinement at singular edge
	const Point<spacedim> p1(2.0 * B / 3.0, B);
	const Point<spacedim> p2(4.0 * B / 3.0, B);
	for(unsigned int refinement_step = 0; refinement_step < n_refinements_sing_edge; ++refinement_step)
	{
		for(const auto& cell : tria_domain.active_cell_iterators())
			for(unsigned int v = 0; v < GeometryInfo<spacedim>::vertices_per_cell; ++v)
				if( ( fabs(cell->vertex(v).distance(p1)) < 1e-12 ) || ( fabs(cell->vertex(v).distance(p2)) < 1e-12 ) )
					cell->set_refine_flag();
		tria_domain.execute_coarsening_and_refinement();
	}

	// fix hanging node positions where manifold is not flat (mathematically probably not strictly necessary - but looks better and apparently works better in practice)
	for(const auto& cell : tria_domain.active_cell_iterators())
	{
		for(unsigned int face = 0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
		{
			if(!cell->face(face)->at_boundary())
			{
				if(cell->face(face)->has_children())
				{
					const Point<spacedim> vertex_pos = 0.5 * (cell->face(face)->vertex(0) + cell->face(face)->vertex(1));
					const auto& child_0 = cell->face(face)->child(0);
					const auto& child_1 = cell->face(face)->child(1);
					if(child_0->vertex(0).distance(child_1->vertex(1)) < 1e-16)
						child_0->vertex(0) = child_1->vertex(1) = vertex_pos;
					else if(child_0->vertex(1).distance(child_1->vertex(0)) < 1e-16)
						child_0->vertex(1) = child_1->vertex(0) = vertex_pos;
					else if(child_0->vertex(1).distance(child_1->vertex(1)) < 1e-16)
						child_0->vertex(1) = child_1->vertex(1) = vertex_pos;
					else if(child_0->vertex(0).distance(child_1->vertex(0)) < 1e-16)
						child_0->vertex(0) = child_1->vertex(0) = vertex_pos;
				}
			}
		}
	}

/************
 * unknowns *
 ************/

	ConstantFunction<spacedim> c_Li_initial(c_Li_ref);														// initial condition Lithium concentration in active particles
	ConstantFunction<spacedim> c_LiX_initial(c_LiX_ref);													// initial condition salt concentration in solid electrolyte
	ConstantFunction<spacedim> c_Lip_initial(c_Lip_ref);													// initial condition Lithium ion concentration in solid electrolyte

	IndependentField<spacedim, spacedim> u("u", FE_Q<spacedim>(degree), spacedim, {0,1});					// displacement field (region 0 is solid electrolyte, region 1 is active particle region)
	IndependentField<spacedim, spacedim> c_Li("c_Li", FE_DGQ<spacedim>(degree), 1, {1}, &c_Li_initial);		// Lithium concentration in active particles
	IndependentField<spacedim, spacedim> c_LiX("c_LiX", FE_DGQ<spacedim>(degree), 1, {0}, &c_LiX_initial);	// salt concentration in solid electrolyte
	IndependentField<spacedim, spacedim> c_Lip("c_Lip", FE_DGQ<spacedim>(degree), 1, {0}, &c_Lip_initial);	// Lithium ion concentration in solid electrolyte

	IndependentField<spacedim, spacedim> eta_Li("eta_Li", FE_Q<spacedim>(degree), 1, {1});					// chemomechanical potential of Li in active particles
	IndependentField<spacedim, spacedim> eta_Lip("eta_Lip", FE_Q<spacedim>(degree), 1, {0});				// electrochemomechanical potential of Li+ ions in solid electrolyte
	IndependentField<spacedim, spacedim> eta_X("eta_X", FE_Q<spacedim>(degree), 1, {0});					// electrochemomechanical potential of X- ions in solid electrolyte

	IndependentField<0, spacedim> phi("phi");																// voltage
	IndependentField<0, spacedim> J("J");																	// total electrical current
	IndependentField<0, spacedim> u_N("u_N");																// constant displacement for periodic b.c.

/********************
 * dependent fields *
 ********************/

// the following fields are explicitly introduced in the manuscript

	// deformation gradient
	DependentField<spacedim, spacedim> F_xx("F_xx");
	DependentField<spacedim, spacedim> F_yy("F_yy");
	DependentField<spacedim, spacedim> F_zz("F_zz");
	DependentField<spacedim, spacedim> F_xy("F_xy");
	DependentField<spacedim, spacedim> F_yz("F_yz");
	DependentField<spacedim, spacedim> F_zx("F_zx");
	DependentField<spacedim, spacedim> F_yx("F_yx");
	DependentField<spacedim, spacedim> F_zy("F_zy");
	DependentField<spacedim, spacedim> F_xz("F_xz");
	F_xx.add_term(1.0, u, 0, 0);
	F_xx.add_term(1.0);
	F_yy.add_term(1.0, u, 1, 1);
	F_yy.add_term(1.0);
	F_xy.add_term(1.0, u, 0, 1);
	F_yx.add_term(1.0, u, 1, 0);
	F_zz.add_term(1.0);

	// X- concentration in solid electrolyte (determined by local electroneutrality)
	DependentField<spacedim, spacedim> c_X_("c_X_");
	c_X_.add_term(1.0, c_Lip);

	// Lithium ion concentration in solid electrolyte
	DependentField<spacedim, spacedim> c_Lip_("c_Lip_");
	c_Lip_.add_term(1.0, c_Lip);

	// vacancy concentration in active particles
	DependentField<spacedim, spacedim> c_V_("c_V");
	c_V_.add_term(-1.0, c_Li);
	c_V_.add_term(c_V_max);

	// potential difference on solid electrolyte - Lithium interface
	DependentField<spacedim-1, spacedim> deta_se_Li("deta_se_Li");
	deta_se_Li.add_term(eta_bar_Li);
	deta_se_Li.add_term(-1.0, eta_Lip, 0, InterfaceSide::minus);

	// potential difference on solid electrolyte - active material interface
	DependentField<spacedim-1, spacedim> deta_se_ap("deta_se_ap");
	deta_se_ap.add_term(F, phi);
	deta_se_ap.add_term(-1.0, eta_Lip, 0, InterfaceSide::minus);
	deta_se_ap.add_term(1.0, eta_Li, 0, InterfaceSide::plus);

// these dependent fields are not explicitly mentioned in the manuscript, but are needed for the actual implementation

	// sum of Lithium ion concentration and salt concentration in solid electrolyte
	// (introduced for simpler implementation of isotropic expansion strain in solid electrolyte)
	DependentField<spacedim, spacedim> c_Lip_LipX_("c_Lip_LipX_");
	c_Lip_LipX_.add_term(1.0, c_LiX);
	c_Lip_LipX_.add_term(1.0, c_Lip);

	// salt concentration in solid electrolyte
	DependentField<spacedim, spacedim> c_LiX_("c_LiX_");
	c_LiX_.add_term(1.0, c_LiX);

	// Lithium concentration in active particles
	DependentField<spacedim, spacedim> c_Li_("c_Li_");
	c_Li_.add_term(1.0, c_Li);

	// chemomechanical potential of Li in active particles and gradient thereof
	DependentField<spacedim, spacedim> eta_Li_("eta_Li_");
	DependentField<spacedim, spacedim> eta_Li_x("eta_Li_x");
	DependentField<spacedim, spacedim> eta_Li_y("eta_Li_y");
	DependentField<spacedim, spacedim> eta_Li_z("eta_Li_z");
	eta_Li_.add_term(1.0, eta_Li);
	eta_Li_x.add_term(1.0, eta_Li, 0, 0);
	eta_Li_y.add_term(1.0, eta_Li, 0, 1);

	// electrochemomechanical potential of Li+ ions in solid electrolyte and gradient thereof
	DependentField<spacedim, spacedim> eta_Lip_("eta_Lip_");
	DependentField<spacedim, spacedim> eta_Lip_x("eta_Lip_x");
	DependentField<spacedim, spacedim> eta_Lip_y("eta_Lip_y");
	DependentField<spacedim, spacedim> eta_Lip_z("eta_Lip_z");
	eta_Lip_.add_term(1.0, eta_Lip);
	eta_Lip_x.add_term(1.0, eta_Lip, 0, 0);
	eta_Lip_y.add_term(1.0, eta_Lip, 0, 1);

	// electrochemomechanical potential of X- ions in solid electrolyte and gradient thereof
	DependentField<spacedim, spacedim> eta_X_("eta_X_");
	DependentField<spacedim, spacedim> eta_X_x("eta_X_x");
	DependentField<spacedim, spacedim> eta_X_y("eta_X_y");
	DependentField<spacedim, spacedim> eta_X_z("eta_X_z");
	eta_X_.add_term(1.0, eta_X);
	eta_X_x.add_term(1.0, eta_X, 0, 0);
	eta_X_y.add_term(1.0, eta_X, 0, 1);

	// chemomechanical potential of LiX salt in solid electrolyte (computed from dissociation equilibrium)
	DependentField<spacedim, spacedim> eta_LiX_("eta_LiX_");
	eta_LiX_.add_term(1.0, eta_Lip);
	eta_LiX_.add_term(1.0, eta_X);

/*************************
 * incremental potential *
 *************************/

	// Note: The actual implementations of the forms of the contributions to the total potential (in particular, the integrands of the integrals) are contained in
	// the following global libraries:
	// 		incremental_fe/scalar_functionals/psi_lib.h (for contributions to the free energy Psi)
	//		incremental_fe/scalar_functionals/omega_lib.h (for contributions to Omega),
	// so that these can be re-used for the modeling of other problems.
	//
	// Note: "Omega" corresponds to "Gamma" in the manuscript (the symbol "Gamma" has been used in the manuscript in order to avoid a duplicate with the symbol used for the domain;
	//                                                         however, this name duplication has not yet been eliminated from the libraries)

	// mechanical part of Helmholtz free energy density in solid electrolyte - psi^se
	KirchhoffMaterial00<spacedim> psi_se_m(	{F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz, c_Lip_LipX_},
											{0},
											QGauss<spacedim>(degree + 1),
											global_data,
											lambda_se,
											mu_se,
											deps_LiX,
											c_Lip_ref + c_LiX_ref,
											alpha);
	TotalPotentialContribution<spacedim> psi_se_m_tpc(psi_se_m);

	// chemical part of Helmholtz free energy density in solid electrolyte (part 1) - psi^se
	PsiChemical00<spacedim> psi_se_c_1(	{c_LiX_},
										{0},
										QGauss<spacedim>(degree + 1),
										global_data,
										R*T, c_LiX_ref, 0.0,
										alpha,
										eps_chemical);
	TotalPotentialContribution<spacedim> psi_se_c_1_tpc(psi_se_c_1);

	// chemical part of Helmholtz free energy density in solid electrolyte (part 2) - psi^se
	PsiChemical00<spacedim> psi_se_c_2(	{c_Lip_},
										{0},
										QGauss<spacedim>(degree + 1),
										global_data,
										2.0 * R*T, c_Lip_ref, 0.0,
										alpha,
										eps_chemical);
	TotalPotentialContribution<spacedim> psi_se_c_2_tpc(psi_se_c_2);

	// mechanical part of Helmholtz free energy density in active material - psi^ap
	KirchhoffMaterial00<spacedim> psi_ap_m(	{F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz, c_Li_},
											{1},
											QGauss<spacedim>(degree + 1),
											global_data,
											lambda_ap,
											mu_ap,
											deps_Li,
											c_Li_ref,
											alpha);
	TotalPotentialContribution<spacedim> psi_ap_m_tpc(psi_ap_m);

	// chemical part of Helmholtz free energy density in active material (part 1) - psi^ap
	PsiChemical00<spacedim> psi_ap_c_1(	{c_Li_},
										{1},
										QGauss<spacedim>(degree + 1),
										global_data,
										R*T, c_Li_ref, 0.0,
										alpha,
										eps_chemical);
	TotalPotentialContribution<spacedim> psi_ap_c_1_tpc(psi_ap_c_1);

	// chemical part of Helmholtz free energy density in active material (part 2) - psi^ap
	PsiChemical00<spacedim> psi_ap_c_2(	{c_V_},
										{1},
										QGauss<spacedim>(degree + 1),
										global_data,
										R*T, c_V_max - c_Li_ref, 0.0,
										alpha,
										eps_chemical);
	TotalPotentialContribution<spacedim> psi_ap_c_2_tpc(psi_ap_c_2);

	// chemical part of Helmholtz free energy density in active material (part 3) - psi^ap
	PsiChemical01<spacedim> psi_ap_c_3(	{c_Li_},
										{1},
										QGauss<spacedim>(degree + 1),
										global_data,
										dmu_ap/c_V_max, c_Li_ref,
										alpha);
	TotalPotentialContribution<spacedim> psi_ap_c_3_tpc(psi_ap_c_3);

	// part 1 of Omega in solid electrolyte - phi^se
	OmegaDualFluxDissipation00<spacedim> omega_se_1({eta_Lip_x, eta_Lip_y, eta_Lip_z, c_Lip_},
													{0},
													QGauss<spacedim>(degree + 1),
													global_data,
													D_Lip/(R*T),
													method,
													alpha);
	TotalPotentialContribution<spacedim> omega_se_1_tpc(omega_se_1);

	// part 2 of Omega in solid electrolyte - phi^se
	OmegaDualFluxDissipation00<spacedim> omega_se_2({eta_X_x, eta_X_y, eta_X_z, c_X_},
													{0},
													QGauss<spacedim>(degree + 1),
													global_data,
													D_X/(R*T),
													method,
													alpha);
	TotalPotentialContribution<spacedim> omega_se_2_tpc(omega_se_2);

	// part 3 of Omega in solid electrolyte - c_dot^Li+ * eta^Li+
	OmegaMixedTerm00<spacedim> omega_se_3(	{c_Lip_, eta_Lip_},
											{0},
											QGauss<spacedim>(degree + 1),
											global_data,
											method,
											alpha);
	TotalPotentialContribution<spacedim> omega_se_3_tpc(omega_se_3);

	// part 4 of Omega in solid electrolyte - c_dot^X- * eta^X-
	OmegaMixedTerm00<spacedim> omega_se_4(	{c_X_, eta_X_},
											{0},
											QGauss<spacedim>(degree + 1),
											global_data,
											method,
											alpha);
	TotalPotentialContribution<spacedim> omega_se_4_tpc(omega_se_4);

	// part 5 of Omega in solid electrolyte - c_dot^LiX * (eta^Li+ + eta^X-)
	OmegaMixedTerm00<spacedim> omega_se_5(	{c_LiX_, eta_LiX_},
											{0},
											QGauss<spacedim>(degree + 1),
											global_data,
											method,
											alpha);
	TotalPotentialContribution<spacedim> omega_se_5_tpc(omega_se_5);

	// part 1 of Omega in active particles - phi^ap
	OmegaDualFluxDissipation00<spacedim> omega_ap_1({eta_Li_x, eta_Li_y, eta_Li_z, c_Li_},
													{1},
													QGauss<spacedim>(degree + 1),
													global_data,
													D_Li/(R*T),
													method,
													alpha);
	TotalPotentialContribution<spacedim> omega_ap_1_tpc(omega_ap_1);

	// part 2 of Omega in active particles - c_dot^Li * eta^Li
	OmegaMixedTerm00<spacedim> omega_ap_2(	{c_Li_, eta_Li_},
											{1},
											QGauss<spacedim>(degree + 1),
											global_data,
											method,
											alpha);
	TotalPotentialContribution<spacedim> omega_ap_2_tpc(omega_ap_2);

	// Omega on interface Sigma^se,Li - phi^se,Li
	OmegaDualButlerVolmer00<spacedim> omega_se_Li(	{deta_se_Li},
													{0},
													QGauss<spacedim-1>(degree + 1),
													global_data,
													i_0_se_Li * R * T / F, beta_se_Li, R * T, 20.0,
													method,
													alpha);
	TotalPotentialContribution<spacedim> omega_se_Li_tpc(omega_se_Li);

	// Omega on interface Sigma^se,ap - phi^se,ap
	OmegaDualButlerVolmer00<spacedim> omega_se_ap(	{deta_se_ap},
													{8},
													QGauss<spacedim-1>(degree + 1),
													global_data,
													i_0_se_ap * R * T / F, beta_se_ap, R * T, 20.0,
													method,
													alpha);
	TotalPotentialContribution<spacedim> omega_se_ap_tpc(omega_se_ap);

	// electrical loading related part of omega (part without relation to spatial locations)
	OmegaElectricalLoading<spacedim> electrical_loading_tpc({&J, &phi}, global_data, method, alpha);

	// finally assemble incremental potential as sum of individual contributions defined earlier
	TotalPotential<spacedim> total_potential;
 	total_potential.add_total_potential_contribution(psi_se_m_tpc);
 	total_potential.add_total_potential_contribution(psi_se_c_1_tpc);
 	total_potential.add_total_potential_contribution(psi_se_c_2_tpc);
 	total_potential.add_total_potential_contribution(psi_ap_m_tpc);
 	total_potential.add_total_potential_contribution(psi_ap_c_1_tpc);
 	total_potential.add_total_potential_contribution(psi_ap_c_2_tpc);
 	total_potential.add_total_potential_contribution(psi_ap_c_3_tpc);
 	total_potential.add_total_potential_contribution(omega_se_1_tpc);
 	total_potential.add_total_potential_contribution(omega_se_2_tpc);
 	total_potential.add_total_potential_contribution(omega_se_3_tpc);
 	total_potential.add_total_potential_contribution(omega_se_4_tpc);
 	total_potential.add_total_potential_contribution(omega_se_5_tpc);
 	total_potential.add_total_potential_contribution(omega_ap_1_tpc);
 	total_potential.add_total_potential_contribution(omega_ap_2_tpc);
	total_potential.add_total_potential_contribution(omega_se_Li_tpc);
	total_potential.add_total_potential_contribution(omega_se_ap_tpc);
	total_potential.add_total_potential_contribution(electrical_loading_tpc);

/*************************
 * Dirichlet constraints *
 *************************/

	RampFunction<spacedim> current_ramp(j_ap_bar);													// define ramp function for current ramp for first loading step

	// define the individual constraints
	DirichletConstraint<spacedim> dc_u_y_bottom(u, 1, InterfaceSide::minus, {4, 5});				// normal displacement constraint at bottom of domain
	DirichletConstraint<spacedim> dc_u_y_top(u, 1, InterfaceSide::minus, {6, 7}, nullptr, &u_N);	// normal displacement constraint at top of domain
	PointConstraint<spacedim, spacedim> dc_u_x(u, 0, Point<spacedim>(-0.5 * L, 0.0));				// lateral displacement constraint at single point
	PointConstraint<0, spacedim> dc_J(J, 0, Point<spacedim>(), &current_ramp); 						// current ramp constraint for first loading step

	// finally assemble the constraints into the constraints object
	Constraints<spacedim> constraints;
	constraints.add_dirichlet_constraint(dc_u_y_bottom);
	constraints.add_dirichlet_constraint(dc_u_y_top);
	constraints.add_point_constraint(dc_u_x);
	constraints.add_point_constraint(dc_J);

/***************************
 * Solution of the problem *
 ***************************/

	// define different solvers
	BlockSolverWrapperPARDISO solver_wrapper_pardiso;
	if((method != 1) || (alpha == 0.0))
		solver_wrapper_pardiso.matrix_type = 2;
	else
		solver_wrapper_pardiso.matrix_type = 0;
	BlockSolverWrapperUMFPACK2 solver_wrapper_umfpack;
	BlockSolverWrapperMA57 solver_wrapper_ma57;

	// select the solver
	SolverWrapper<Vector<double>, BlockVector<double>, TwoBlockMatrix<SparseMatrix<double>>, TwoBlockSparsityPattern>* solver_wrapper;
	if((method != 1) || (alpha == 0.0))
	{
		switch(solver_sym)
		{
			case 0:
				solver_wrapper = &solver_wrapper_pardiso;
				cout << "Selected PARDISO as solver" << endl;
				if(alpha == 0.0)
					solver_wrapper_pardiso.res_max = 1e16;
				break;
			case 1:
				solver_wrapper = &solver_wrapper_ma57;
				cout << "Selected MA57 as solver" << endl;
				break;
			default:
				solver_wrapper = &solver_wrapper_umfpack;
				cout << "Selected UMFPACK as solver" << endl;
		}
	}
	else
	{
		switch(solver_unsym)
		{
			case 0:
				solver_wrapper = &solver_wrapper_pardiso;
				cout << "Selected PARDISO as solver" << endl;
				break;
			default:
				solver_wrapper = &solver_wrapper_umfpack;
				cout << "Selected UMFPACK as solver" << endl;
		}
	}

	// adjust some solver settings
	solver_wrapper_pardiso.use_defaults = false;
	solver_wrapper_pardiso.n_iterative_refinements = 10;
	solver_wrapper_pardiso.apply_scaling = 1;
	solver_wrapper_pardiso.pivot_perturbation = 10;
	solver_wrapper_pardiso.analyze = 1;
	solver_wrapper_umfpack.analyze = 1;
	solver_wrapper_ma57.analyze = 1;

	// set up the finite element model
	FEModel<spacedim, Vector<double>, BlockVector<double>, GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>> fe_model(total_potential, tria_system, mapping_domain, mapping_interface, global_data, constraints, *solver_wrapper);

	// get the dof indices of phi and J (in order to be later able to read their values)
	const unsigned int dof_index_phi_ap = fe_model.get_assembly_helper().get_global_dof_index_C(&phi);
	const unsigned int dof_index_j_ap = fe_model.get_assembly_helper().get_global_dof_index_C(&J);

	// vector to store values (t, phi, J_dot)
	vector<tuple<double, double, double>> phi_j;

// first loading step (constant current loading)

	electrical_loading_tpc.loading_type = 0;
	electrical_loading_tpc.j_bar = j_ap_bar;

	double start_time = global_data.get_t();
	double end_time = start_time + dt;
	double inc = (end_time - start_time)/(double)N_1;
	for(;;)
	{
		double new_time = global_data.get_t() + inc;
		if(new_time > end_time)
			new_time = end_time;
		cout << "Old time=" << global_data.get_t() << endl;
		cout << "New time=" << new_time << endl;

		const int iter = fe_model.do_time_step(new_time);
		if(iter >= 0)
		{
			fe_model.write_output_independent_fields("results/domain", "results/interface", cell_divisions);
			phi_j.push_back(make_tuple(new_time, fe_model.get_solution_vector()(dof_index_phi_ap), (fe_model.get_solution_vector()(dof_index_j_ap) - fe_model.get_solution_ref_vector()(dof_index_j_ap))/inc));
			if(new_time == end_time)
				break;
		}
		else
		{
			cout << "ERROR, Computation failed!" << endl;
			global_data.print_error_messages();
			break;
		}
		cout << endl;
	}

// second loading step (constant voltage loading)

	electrical_loading_tpc.loading_type = 1;
	electrical_loading_tpc.phi = fe_model.get_solution_vector()(dof_index_phi_ap);
	dc_J.set_constraint_is_active(false);	// deactivate constraint for time-integrated current
	start_time = global_data.get_t();
	end_time = start_time + dt;

	// re-analyze sparsity pattern and matrix once as constraints have changed
	solver_wrapper_pardiso.analyze = 1;
	solver_wrapper_umfpack.analyze = 1;
	solver_wrapper_ma57.analyze = 1;
	global_data.set_compute_sparsity_pattern(1);

	for(;;)
	{
		double new_time = global_data.get_t() + inc;
		if(new_time > end_time)
			new_time = end_time;
		cout << "Old time=" << global_data.get_t() << endl;
		cout << "New time=" << new_time << endl;

		const int iter = fe_model.do_time_step(new_time);
		if(iter >= 0)
		{
			fe_model.write_output_independent_fields("results/domain", "results/interface", cell_divisions);
			phi_j.push_back(make_tuple(new_time, fe_model.get_solution_vector()(dof_index_phi_ap), (fe_model.get_solution_vector()(dof_index_j_ap) - fe_model.get_solution_ref_vector()(dof_index_j_ap))/inc) );
			if(new_time == end_time)
				break;
		}
		else
		{
			cout << "ERROR, Computation failed!" << endl;
			global_data.print_error_messages();
			break;
		}
		cout << endl;
	}

// third loading step (discharge through electrical resistance)

	electrical_loading_tpc.loading_type = 2;
	electrical_loading_tpc.R_el = R_el;
	dc_J.set_constraint_is_active(true);	// formally activate the constraint for time-integrated current again - the value to which it is constrained here is immaterial (the constraint is only needed to make the linear systems definite).
	start_time = global_data.get_t();
	end_time = start_time + dt;

	// re-analyze sparsity pattern and matrix once as constraints have changed
	solver_wrapper_pardiso.analyze = 1;
	solver_wrapper_umfpack.analyze = 1;
	solver_wrapper_ma57.analyze = 1;
	global_data.set_compute_sparsity_pattern(1);

	for(;;)
	{
		double new_time = global_data.get_t() + inc;
		if(new_time > end_time)
			new_time = end_time;
		cout << "Old time=" << global_data.get_t() << endl;
		cout << "New time=" << new_time << endl;

		const int iter = fe_model.do_time_step(new_time);
		if(iter >= 0)
		{
			fe_model.write_output_independent_fields("results/domain", "results/interface", cell_divisions);
			phi_j.push_back(make_tuple(new_time, fe_model.get_solution_vector()(dof_index_phi_ap), -fe_model.get_solution_vector()(dof_index_phi_ap)/electrical_loading_tpc.R_el));
			if(new_time == end_time)
				break;
		}
		else
		{
			cout << "ERROR, Computation failed!" << endl;
			global_data.print_error_messages();
			break;
		}
		cout << endl;
	}
	global_data.print_error_messages();

// write global results to file and print to screen (current, voltage)

	FILE* printout = fopen ("results/independent_scalars.dat", "w");
	for(const auto& phi_j_el : phi_j)
	{
		fprintf(printout, "%- 1.16e %- 1.16e %- 1.16e\n", get<0>(phi_j_el), get<1>(phi_j_el), get<2>(phi_j_el));
		printf("%- 10.2f %- 10.2f %- 10.2f\n", get<0>(phi_j_el), get<1>(phi_j_el), get<2>(phi_j_el));
	}
	fclose(printout);

}
