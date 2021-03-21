/**
 * Copyright (C) 2020 by Sebastian Stark
 *
 * 3d all solid state battery example according to Sect. 5.1 of manuscript "A unified approach to standard dissipative continua with application to electrochemomechanically coupled problems"
 * For details regarding the physical situation simulated with this program, see this manuscript.
 */

#include <iostream>
#include <fstream>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>

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

// post-processor computing determinant of deformation gradient
template<unsigned int spacedim>
class PostprocessorJ : public DataPostprocessorScalar<spacedim>
{
	// global component index of first component of displacement field
	const unsigned int
	global_component_index_u;

public:
	PostprocessorJ(	const string&		name,
					const unsigned int	global_component_index_u)
	:
	DataPostprocessorScalar<spacedim>(name, update_gradients),
	global_component_index_u(global_component_index_u)
	{
	}

	void
	evaluate_vector_field(	const DataPostprocessorInputs::Vector<spacedim>&	input_data,
							vector<Vector<double>>&								computed_quantities)
	const
	{
		//double avg = 0.0;
		for(unsigned int dataset = 0; dataset < input_data.solution_gradients.size(); ++dataset)
		{
			Tensor<2, spacedim> F;
			for(unsigned int m = 0; m < spacedim; ++m)
			{
				F[m][m] += 1.0;
				for(unsigned int n = 0; n < spacedim; ++n)
					F[m][n] += input_data.solution_gradients[dataset][global_component_index_u + m][n];
			}
			const double J = determinant(F);
			computed_quantities[dataset][0] = J;
			//avg += J;
		}
		//avg *= 1.0/input_data.solution_gradients.size();

		//for(unsigned int dataset = 0; dataset < input_data.solution_gradients.size(); ++dataset)
		//	computed_quantities[dataset][0] = avg;
	}
};

// postprocessor computing the pressure from the displacement field
template <int spacedim>
class PressurePostprocessor : public DataPostprocessorScalar<spacedim>
{
private:

	/**
	 * Index of first component of displacement field
	 */
	const unsigned int
	global_component_index_u;

	/**
	 * Indices of concentration fields for solid electrolyte and active particles
	 */
	const vector<vector<unsigned int>>
	global_component_indices_c;

	/**
	 * Lame's constant \f$\lambda^\mathrm{se}\f$, \f$\lambda^\mathrm{ap}\f$
	 */
	const vector<double>
	lambda;

	/**
	 * Lame's constant \f$\mu^\mathrm{se}\f$, \f$\mu^\mathrm{ap}\f$
	 */
	const vector<double>
	mu;

	/**
	 * volume strain change \f$\Delta \varepsilon^\mathrm{LiX}\f$, \f$\Delta \varepsilon^\mathrm{Li}\f$
	 */
	const vector<double>
	deps;

	/**
	 * reference concentrations \f$c^\mathrm{LiX,ref} + c^\mathrm{Li+,ref}\f$, \f$c^\mathrm{Li,ref}\f$
	 */
	const vector<double>
	c_ref;

public:
	PressurePostprocessor(const string&						name,
						const unsigned int					global_component_index_u,
						const vector<vector<unsigned int>>	global_component_indices_c,
						const vector<double>				lambda,
						const vector<double>				mu,
						const vector<double>				deps,
						const vector<double>				c_ref)
	:
	DataPostprocessorScalar<spacedim>(name, update_values | update_gradients),
	global_component_index_u(global_component_index_u),
	global_component_indices_c(global_component_indices_c),
	lambda(lambda),
	mu(mu),
	deps(deps),
	c_ref(c_ref)
	{
	}

	void
	evaluate_vector_field(	const DataPostprocessorInputs::Vector<spacedim>&	input_data,
							vector<Vector<double>>&								computed_quantities)
	const override
	{

		const Tensor<2, 3> I = unit_symmetric_tensor<3,double>();
		Tensor<2, 3> F, E, T;
		const auto current_cell = input_data.template get_cell<hp::DoFHandler<spacedim>>();
		const unsigned int material_id = current_cell->material_id();

		for (unsigned int dataset = 0; dataset < input_data.solution_gradients.size(); ++dataset)
		{

			F = 0.0;
			for (unsigned int d = 0; d < spacedim; ++d)
				for (unsigned int e = 0; e < spacedim; ++e)
					F[d][e] = input_data.solution_gradients[dataset][global_component_index_u + d][e];
			for (unsigned int d = 0; d < 3; ++d)
				F[d][d] += 1.0;

			double c = 0.0;
			for(const auto& global_component_index_c : global_component_indices_c[material_id])
				c += input_data.solution_values[dataset][global_component_index_c];
			E = 0.5 * (contract<0, 0>(F, F) - I) - deps[material_id]/3.0 * (c/c_ref[material_id] - 1.0) * I;
			T = lambda[material_id] * trace(E) * I + 2.0 * mu[material_id] * E;
			computed_quantities[dataset][0] = -trace(T)/3.0;
		}
	}
 };



// the main program
int main()
{

/********************
 * parameters, etc. *
 ********************/

	const unsigned int spacedim = 3;	// spatial dimensionality; this does only work for spacedim == 3

	// quantities used for normalization (in SI units)
	const double c_ast = 10000.0;
	const double D_ast = 1e-16;
	const double R_ast = 8.3144;
	const double T_ast = 293.15;
	const double F_ast = 96485.33;
	const double L_ast = 1e-6;

	// parameters (note: geometrical parameters must be consistent with mesh file, which is read in later)
	const unsigned int N_ap = 5;			// number of active particles
	const double L = N_ap * 6e-6 / L_ast;	// width of domain
	const double B = 1.5e-6 / L_ast;		// height of computational domain

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
	const double j_threshold_charging = 0.03;							// fraction of current magnitude at which constant voltage charging is stopped
	const double j_threshold_discharging = 0.03;						// fraction of current magnitude at which discharging is stopped
	const double j_bar = -5e-12 / (F_ast * c_ast * L_ast * D_ast);		// constant current charging current
	const double phi_bar = 4.0 * F_ast / (R_ast * T_ast);				// cut-off voltage
	const double R_el = fabs(phi_bar / j_bar);							// electrical resistance

	// numerical parameters
	const double eps_chemical = 1e-4;						// numerical parameter for regularization of chemical potential
	const unsigned int N_refinements_sing_edge = 2;			// number of refinements at edge with stress singularity
	const unsigned int N_refinements_global = 1;			// number of global mesh refinements
	const unsigned int solver_sym = 0;						// solver for method != 1: 0 - PARDISO, 1 - MA57, else - UMFPACK
	const unsigned int solver_unsym = 0;					// solver for method == 1: 0 - PARDISO, else - UMFPACK
	const unsigned int method = 1;							// numerical method (modified alpha-family)
	const double alpha = 0.5;								// time integration parameter alpha
	const unsigned int degree = 2;							// polynomial degree of finite elements
	const unsigned int cell_divisions = degree;				// cell divisions for output
	const double inc_0 = 0.1 * D_ast / (L_ast * L_ast);		// initial time increment (applied in constant current charging step and discharging step)
	const double inc_max = 500.0 * D_ast / (L_ast * L_ast);	// maximum time increment

	// mappings
	MappingQGeneric<spacedim, spacedim> mapping_domain(degree);			// FE mapping on domain
	MappingQGeneric<spacedim-1, spacedim> mapping_interface(degree);	// FE mapping on interfaces

	// global data object, used to transfer global data (like time step information) between different potential contributions and to define parameters for the Newton-Raphson algorithm, etc.
	GlobalDataIncrementalFE<spacedim> global_data;

	// define some parameters for the problem solution
	global_data.set_compute_sparsity_pattern(1);	// compute sparsity pattern only once and re-use for subsequent steps
	global_data.set_max_iter(20);					// maximum number of Newton-Raphson iterations allowed
	global_data.set_max_cutbacks(1);				// maximum number of cutbacks allowed for line search
	global_data.set_perform_line_search(false);		// do not perform line search
	global_data.set_scale_residual(false);			// do not scale the residual according to the matrix diagonals

/*****************************************************
 * grid, assignment of domain and interface portions *
 *****************************************************/

	// read in the mesh file
	// the mesh already contains the assignment of the domain portions by material id's:
	// 0 - solid electrolyte
	// 1 - active particles
	// attention: the region occupied by the mesh is 0 <= X <= B, 0 <= Y <= B, -0.5 * L <= Z <= 0.5 * L,  (i.e., the coordinate system is shifted along the Z axis compared to the manuscript)
	Triangulation<spacedim> tria_domain;
	GridIn<spacedim> grid_in;
	ifstream input_file("tria_domain_3d.vtk");
	grid_in.attach_triangulation(tria_domain);
	grid_in.read_vtk(input_file);
	input_file.close();

	// triangulation system and interface definition
	// 0  - Sigma_se,X=0
	// 1  - Sigma_ap,X=0
	// 2  - Sigma_se,X=B
	// 3  - Sigma_ap,X=B
	// 4  - Sigma_se,Y=0
	// 5  - Sigma_ap,Y=0
	// 6  - Sigma_se,Y=B
	// 7  - Sigma_ap,Y=B
	// 8  - Sigma_se,Li
	// 10 - Sigma_se_Z=L
	// 11 - Sigma_ap_Z=L
	// 12 - Sigma_se,ap
	dealii::GalerkinTools::TriangulationSystem<spacedim> tria_system(tria_domain, true);		// use automatic vertex correction to make sure that domain and interface mesh remain consistent
																								// (things would otherwise be slightly messed up since the transfinite interpolation manifold
																								// apparently handles things differently on the boundary then in the volume)

	//define interfaces
	for(const auto& cell : tria_domain.cell_iterators_on_level(0))
	{
		for(unsigned int face = 0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
		{
			if(cell->face(face)->at_boundary())
			{
				if(cell->face(face)->center()[0] < 1e-12)
				{
					if(cell->material_id() == 0)
						tria_system.add_interface_cell(cell, face, 0);
					else
						tria_system.add_interface_cell(cell, face, 1);
				}
				else if(cell->face(face)->center()[0] > B - 1e-12)
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
				else if(cell->face(face)->center()[2] < - 0.5 * L + 1e-12)
				{
					tria_system.add_interface_cell(cell, face, 8);
				}
				else if(cell->face(face)->center()[2] > 0.5 * L - 1e-12)
				{
					if(cell->material_id() == 0)
						tria_system.add_interface_cell(cell, face, 10);
					else
						tria_system.add_interface_cell(cell, face, 11);
				}
			}
			else
			{
				if( (cell->material_id() == 0) && (cell->neighbor(face)->material_id() == 1) )
					tria_system.add_interface_cell(cell, face, 12);
			}
		}
	}

	// attach manifolds, so that curved interface of active particles is correctly represented upon mesh refinement
	FlatManifold<spacedim> flat_manifold_domain;
	FlatManifold<spacedim-1, spacedim> flat_manifold_interface;
	Tensor<1,spacedim> direction_x, direction_y, direction_z;
	direction_x[0] = direction_y[1] = direction_z[2] = 1.0;
	CylindricalManifold<spacedim> cylindrical_manifold_z_domain(direction_z, Point<spacedim>());
	CylindricalManifold<spacedim-1, spacedim> cylindrical_manifold_z_interface(direction_z, Point<spacedim>());
	vector<SphericalManifold<spacedim>> spherical_manifold_domain;
	vector<SphericalManifold<spacedim-1, spacedim>> spherical_manifold_interface;
	vector<CylindricalManifold<spacedim>> cylindrical_manifold_x_domain, cylindrical_manifold_y_domain;
	vector<CylindricalManifold<spacedim-1, spacedim>> cylindrical_manifold_x_interface, cylindrical_manifold_y_interface;
	for(unsigned int n = 0; n < N_ap; ++n)
	{
		Point<spacedim> origin(0.0, 0.0, B + 2.0 * B * n);
		spherical_manifold_domain.push_back(SphericalManifold<spacedim>(origin));
		spherical_manifold_interface.push_back(SphericalManifold<spacedim-1, spacedim>(origin));
		cylindrical_manifold_x_domain.push_back(CylindricalManifold<spacedim>(direction_x, origin));
		cylindrical_manifold_y_domain.push_back(CylindricalManifold<spacedim>(direction_y, origin));
		cylindrical_manifold_x_interface.push_back(CylindricalManifold<spacedim-1, spacedim>(direction_x, origin));
		cylindrical_manifold_y_interface.push_back(CylindricalManifold<spacedim-1, spacedim>(direction_y, origin));
		tria_domain.set_manifold(2 + 3 * n, spherical_manifold_domain.back());
		tria_domain.set_manifold(3 + 3 * n, cylindrical_manifold_x_domain.back());
		tria_domain.set_manifold(4 + 3 * n, cylindrical_manifold_y_domain.back());
	}

	tria_domain.set_manifold(0, flat_manifold_domain);
	tria_domain.set_manifold(1, cylindrical_manifold_z_domain);

	for(unsigned int n = 0; n < N_ap; ++n)
	{
		tria_system.set_interface_manifold(2 + 3 * n, spherical_manifold_interface[n]);
		tria_system.set_interface_manifold(3 + 3 * n, cylindrical_manifold_x_interface[n]);
		tria_system.set_interface_manifold(4 + 3 * n, cylindrical_manifold_y_interface[n]);
	}
	tria_system.set_interface_manifold(0, flat_manifold_interface);
	tria_system.set_interface_manifold(1, cylindrical_manifold_z_interface);
	// note: don't assign a transfinite interpolation manifold to the interface here and let the automatic vertex correction fix things
	//       (this is only possible since the entire interface mesh which would require transfinite interpolation is in fact flat)

	// finish definition of geometry
	tria_system.close();

	// global mesh refinement
	tria_domain.refine_global(N_refinements_global);

	// mesh refinement at singular edge
	vector<pair<Point<spacedim>, unsigned int>> center_points;
	center_points.push_back(make_pair(Point<spacedim>(0.0, 0.0, 0.0), 2));
	for(unsigned int n = 0; n < N_ap; ++n)
	{
		center_points.push_back(make_pair(Point<spacedim>(0.0, B, B + n * 2.0 * B), 1));
		center_points.push_back(make_pair(Point<spacedim>(0.0, 0.0, 2.0 * B + n * 2.0 * B), 2));
		center_points.push_back(make_pair(Point<spacedim>(B, 0.0, B + n * 2.0 * B), 0));
	}
	for(unsigned int refinement_step = 0; refinement_step < N_refinements_sing_edge; ++refinement_step)
	{
		for(const auto& cell : tria_domain.active_cell_iterators())
		{
			for(unsigned int v = 0; v < GeometryInfo<spacedim>::vertices_per_cell; ++v)
			{
				for(const auto& center_point : center_points)
				{
					const unsigned int plane = center_point.second;
					const double plane_coord = center_point.first[center_point.second];
					const auto vertex = cell->vertex(v);
					if( ( fabs(vertex.distance(center_point.first) - B / 3.0) < 1e-4 ) && ( fabs(vertex[plane] - plane_coord) < 1e-12 ) )
						cell->set_refine_flag();
				}
			}
		}
		tria_domain.execute_coarsening_and_refinement();
	}

	tria_system.write_triangulations_vtk("tria_domain.vtk", "tria_interface.vtk");

/**************************************
 * unknowns and Dirichlet constraints *
 **************************************/

	dealii::Functions::ConstantFunction<spacedim> c_Li_initial(c_Li_ref);									// initial condition Lithium concentration in active particles
	dealii::Functions::ConstantFunction<spacedim> c_LiX_initial(c_LiX_ref);									// initial condition salt concentration in solid electrolyte
	dealii::Functions::ConstantFunction<spacedim> c_Lip_initial(c_Lip_ref);									// initial condition Lithium ion concentration in solid electrolyte
	RampFunction<spacedim> current_ramp(j_bar);																// define ramp function for current ramp for first loading step

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

	// define constraints for function spaces
	DirichletConstraint<spacedim> dc_u_x_bottom(u, 0, InterfaceSide::minus, {0, 1});						// normal displacement constraint on plane X=0
	DirichletConstraint<spacedim> dc_u_x_top(u, 0, InterfaceSide::minus, {2, 3}, nullptr, &u_N);			// normal displacement constraint on plane X=B
	DirichletConstraint<spacedim> dc_u_y_bottom(u, 1, InterfaceSide::minus, {4, 5});						// normal displacement constraint at plane Y=0
	DirichletConstraint<spacedim> dc_u_y_top(u, 1, InterfaceSide::minus, {6, 7}, nullptr, &u_N);			// normal displacement constraint at plane Y=B
	PointConstraint<spacedim, spacedim> dc_u_x(u, 2, Point<spacedim>(0.0, 0.0, -0.5 * L));					// lateral displacement constraint at single point
	PointConstraint<0, spacedim> dc_J(J, 0, Point<spacedim>(), &current_ramp); 								// current ramp constraint for first loading step

	// finally assemble the constraints into the constraints object
	Constraints<spacedim> constraints;
	constraints.add_dirichlet_constraint(dc_u_x_bottom);
	constraints.add_dirichlet_constraint(dc_u_x_top);
	constraints.add_dirichlet_constraint(dc_u_y_bottom);
	constraints.add_dirichlet_constraint(dc_u_y_top);
	constraints.add_point_constraint(dc_u_x);
	constraints.add_point_constraint(dc_J);

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
	F_xx.add_term(1.0);
	F_xx.add_term(1.0, u, 0, 0);
	F_yy.add_term(1.0);
	F_yy.add_term(1.0, u, 1, 1);
	F_zz.add_term(1.0);
	F_zz.add_term(1.0, u, 2, 2);
	F_xy.add_term(1.0, u, 0, 1);
	F_yx.add_term(1.0, u, 1, 0);
	F_yz.add_term(1.0, u, 1, 2);
	F_zy.add_term(1.0, u, 2, 1);
	F_zx.add_term(1.0, u, 2, 0);
	F_xz.add_term(1.0, u, 0, 2);

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
	eta_Li_z.add_term(1.0, eta_Li, 0, 2);

	// electrochemomechanical potential of Li+ ions in solid electrolyte and gradient thereof
	DependentField<spacedim, spacedim> eta_Lip_("eta_Lip_");
	DependentField<spacedim, spacedim> eta_Lip_x("eta_Lip_x");
	DependentField<spacedim, spacedim> eta_Lip_y("eta_Lip_y");
	DependentField<spacedim, spacedim> eta_Lip_z("eta_Lip_z");
	eta_Lip_.add_term(1.0, eta_Lip);
	eta_Lip_x.add_term(1.0, eta_Lip, 0, 0);
	eta_Lip_y.add_term(1.0, eta_Lip, 0, 1);
	eta_Lip_z.add_term(1.0, eta_Lip, 0, 2);

	// electrochemomechanical potential of X- ions in solid electrolyte and gradient thereof
	DependentField<spacedim, spacedim> eta_X_("eta_X_");
	DependentField<spacedim, spacedim> eta_X_x("eta_X_x");
	DependentField<spacedim, spacedim> eta_X_y("eta_X_y");
	DependentField<spacedim, spacedim> eta_X_z("eta_X_z");
	eta_X_.add_term(1.0, eta_X);
	eta_X_x.add_term(1.0, eta_X, 0, 0);
	eta_X_y.add_term(1.0, eta_X, 0, 1);
	eta_X_z.add_term(1.0, eta_X, 0, 2);

	// chemomechanical potential of LiX salt in solid electrolyte (computed from dissociation equilibrium, introduced for simpler implementation)
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
													{8},
													QGauss<spacedim-1>(degree + 1),
													global_data,
													i_0_se_Li * R * T / F, beta_se_Li, R * T, 20.0,
													method,
													alpha);
	TotalPotentialContribution<spacedim> omega_se_Li_tpc(omega_se_Li);

	// Omega on interface Sigma^se,ap - phi^se,ap
	OmegaDualButlerVolmer00<spacedim> omega_se_ap(	{deta_se_ap},
													{12},
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
	cout << "Number of active cells in mesh: " << tria_domain.n_active_cells() << endl;
	cout << "Number of unknowns: " << fe_model.get_assembly_helper().get_dof_handler_system().n_dofs() << endl << endl;

	// postprocessor computing the determinant of the deformation gradient
	PostprocessorJ<spacedim> pp_J("J", fe_model.get_assembly_helper().get_u_omega_global_component_index(u));
	fe_model.attach_data_postprocessor_domain(pp_J);

	// postprocessor computing the pressures
	vector<vector<unsigned int>> component_indices_c(2);
	component_indices_c[0].push_back(fe_model.get_assembly_helper().get_u_omega_global_component_index(c_Lip));
	component_indices_c[0].push_back(fe_model.get_assembly_helper().get_u_omega_global_component_index(c_LiX));
	component_indices_c[1].push_back(fe_model.get_assembly_helper().get_u_omega_global_component_index(c_Li));
	PressurePostprocessor<spacedim> postproc_P(	"P",
												fe_model.get_assembly_helper().get_u_omega_global_component_index(u),
												component_indices_c,
												{lambda_se, lambda_ap},
												{mu_se, mu_ap},
												{deps_LiX, deps_Li},
												{c_Lip_ref + c_LiX_ref, c_Li_ref});
	fe_model.attach_data_postprocessor_domain(postproc_P);

	// get the dof indices of phi and J (in order to be later able to read their values)
	const unsigned int dof_index_phi_ap = fe_model.get_assembly_helper().get_global_dof_index_C(&phi);
	const unsigned int dof_index_j_ap = fe_model.get_assembly_helper().get_global_dof_index_C(&J);

	// vector to store values (t, phi, J_dot)
	vector<tuple<double, double, double>> phi_j;

// first loading step
	electrical_loading_tpc.loading_type = 0;
	electrical_loading_tpc.j_bar = j_bar;

	double inc = inc_0;
	double pre_final_inc = inc_0;

	for(;;)
	{
		// predict cutoff time by linear extrapolation
		double cutoff_time = 1e16;
		const unsigned int N_data_sets = phi_j.size();
		if(N_data_sets > 1)
		{
			const double t_0__ = get<0>(phi_j[N_data_sets - 2]);
			const double t_1__ = get<0>(phi_j[N_data_sets - 1]);
			const double phi_0__ = get<1>(phi_j[N_data_sets - 2]);
			const double phi_1__ = get<1>(phi_j[N_data_sets - 1]);
			cutoff_time = ( ( ( phi_bar - phi_1__ ) * t_0__) - ( ( phi_bar - phi_0__ ) * t_1__) ) / ( phi_0__ - phi_1__ );
		}
		cout << "Cutoff time=" << cutoff_time << endl;

		double new_time = global_data.get_t() + inc;
		if(new_time >= cutoff_time)
		{
			pre_final_inc = inc;
			inc = cutoff_time - global_data.get_t();
			new_time = cutoff_time;
		}

		cout << "Old time=" << global_data.get_t() << endl;
		cout << "New time=" << new_time << endl;

		const int iter = fe_model.do_time_step(new_time);
		if(iter >= 0)
		{
			fe_model.write_output_independent_fields("results/domain", "results/interface", cell_divisions);
			phi_j.push_back(make_tuple(new_time - inc * (1.0 - alpha), fe_model.get_solution_vector()(dof_index_phi_ap), (fe_model.get_solution_vector()(dof_index_j_ap) - fe_model.get_solution_ref_vector()(dof_index_j_ap))/inc));

			if(new_time == cutoff_time)
				break;
			if(iter < 5)
			{
				inc = inc * 2.0;
				if(inc > inc_max)
					inc = inc_max;
			}
		}
		else
		{
			cout << "ERROR, CUTBACK necessary!" << endl;
			inc = inc / 2.0;
			continue;
		}
		cout << endl;
	}
	cout << "First step completed" << endl;

// second step
	electrical_loading_tpc.loading_type = 1;
	electrical_loading_tpc.phi = phi_bar;
	dc_J.set_constraint_is_active(false);	// deactivate constraint for time-integrated current

	// re-analyze sparsity pattern and matrix once as constraints have changed
	solver_wrapper_pardiso.analyze = 1;
	solver_wrapper_umfpack.analyze = 1;
	solver_wrapper_ma57.analyze = 1;
	global_data.set_compute_sparsity_pattern(1);

	// reset time increment size to the one used before the final increment in the constant current charging step
	inc = pre_final_inc;

	for(;;)
	{
		double new_time = global_data.get_t() + inc;
		cout << "Old time=" << global_data.get_t() << endl;
		cout << "New time=" << new_time << endl;

		const int iter = fe_model.do_time_step(new_time);
		if(iter >= 0)
		{
			fe_model.write_output_independent_fields("results/domain", "results/interface", cell_divisions);
			phi_j.push_back(make_tuple( new_time - inc * (1.0 - alpha), fe_model.get_solution_vector()(dof_index_phi_ap), (fe_model.get_solution_vector()(dof_index_j_ap) - fe_model.get_solution_ref_vector()(dof_index_j_ap))/inc ) );
			if( fabs( (fe_model.get_solution_vector()(dof_index_j_ap) - fe_model.get_solution_ref_vector()(dof_index_j_ap))/inc ) <= fabs(j_threshold_charging * j_bar) )
				break;
			if(iter < 5)
			{
				inc = inc * 2.0;
				if(inc > inc_max)
					inc = inc_max;
			}
		}
		else
		{
			cout << "ERROR, CUTBACK necessary!" << endl;
			inc = inc / 2.0;
			continue;
		}
		cout << endl;
	}
	cout << "Second step completed" << endl;

// third loading step
	electrical_loading_tpc.loading_type = 2;
	electrical_loading_tpc.R_el = R_el;
	dc_J.set_constraint_is_active(true);	// formally activate the constraint for time-integrated current again - the value to which it is constrained here is immaterial (the constraint is only needed to make the linear systems definite).

	// re-analyze sparsity pattern and matrix once as constraints have changed
	solver_wrapper_pardiso.analyze = 1;
	solver_wrapper_umfpack.analyze = 1;
	solver_wrapper_ma57.analyze = 1;
	global_data.set_compute_sparsity_pattern(1);
	global_data.set_max_iter(8);			// maximum number of Newton-Raphson iterations allowed (reduce in order to properly resolve final breakdown of electrical current)

	// reset time increment size to the initial one
	inc = inc_0;

	for(;;)
	{
		double new_time = global_data.get_t() + inc;
		cout << "Old time=" << global_data.get_t() << endl;
		cout << "New time=" << new_time << endl;

		const int iter = fe_model.do_time_step(new_time);
		if(iter >= 0)
		{
			fe_model.write_output_independent_fields("results/domain", "results/interface", cell_divisions);
			phi_j.push_back(make_tuple(new_time - inc * (1.0 - alpha), fe_model.get_solution_vector()(dof_index_phi_ap), -fe_model.get_solution_vector()(dof_index_phi_ap)/R_el));
			if( fabs(fe_model.get_solution_vector()(dof_index_phi_ap)/R_el) <= fabs(j_threshold_discharging * j_bar) )
				break;
			if(iter < 5)
			{
				inc = inc * 2.0;
				if(inc > inc_max)
					inc = inc_max;
			}
		}
		else
		{
			cout << "ERROR, CUTBACK necessary!" << endl;
			inc = inc / 2.0;
			continue;
		}
		cout << endl;
	}
	global_data.print_error_messages();

	if(tria_system.get_this_proc_n_procs().first == 0)
	{
		FILE* printout = fopen ("results/independent_scalars.dat", "w");
		for(const auto& phi_el : phi_j)
		{
			fprintf(printout, "%- 1.16e %- 1.16e %- 1.16e\n", get<0>(phi_el), get<1>(phi_el), get<2>(phi_el));
			printf("%- 10.2f %- 10.2f %- 10.2f\n", get<0>(phi_el), get<1>(phi_el), get<2>(phi_el));
		}
		fclose(printout);
	}

}
