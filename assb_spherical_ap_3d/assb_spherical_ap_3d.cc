#define PARALLEL

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

/**
 * postprocessor computing the pressure from the displacement field
 */
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
	 * Indices of concentration fields
	 */
	const vector<unsigned int>
	global_component_indices_c;

	/**
	 * Material id for which this produces non-zero graphical output
	 */
	const unsigned int
	material_id;

	/**
	 * Lame's constant \f$\lambda\f$
	 */
	const double
	lambda;

	/**
	 * Lame's constant \f$\lambda\f$
	 */
	const double
	mu;

	/**
	 * \f$\Delta \varepsilon\f$
	 */
	const double
	deps;

	/**
	 * \f$c^\mathrm{ref}\f$
	 */
	const double
	c_ref;

public:
	PressurePostprocessor(const string&				name,
						const unsigned int			global_component_index_u,
						const vector<unsigned int>	global_component_indices_c,
						const unsigned int			material_id,
						const double				lambda,
						const double				mu,
						const double				deps,
						const double				c_ref)
	:
	DataPostprocessorScalar<spacedim>(name, update_values | update_gradients),
	global_component_index_u(global_component_index_u),
	global_component_indices_c(global_component_indices_c),
	material_id(material_id),
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
		SymmetricTensor<2,3> sigma, s;
		const auto current_cell = input_data.template get_cell<hp::DoFHandler<spacedim>>();

		for (unsigned int dataset = 0; dataset < input_data.solution_gradients.size(); ++dataset)
		{

			if(current_cell->material_id() == material_id)
			{
				F = 0.0;
				for (unsigned int d = 0; d < spacedim; ++d)
					for (unsigned int e = 0; e < spacedim; ++e)
						F[d][e] = input_data.solution_gradients[dataset][global_component_index_u + d][e];
				for (unsigned int d = 0; d < 3; ++d)
					F[d][d] += 1.0;

				double c = 0.0;
				for(const auto& global_component_index_c : global_component_indices_c)
					c += input_data.solution_values[dataset][global_component_index_c];
				E = 0.5 * (contract<0, 0>(F, F) - I) - deps/3.0 * (c/c_ref - 1.0) * I;
				T = lambda * trace(E) * I + 2.0 * mu * E;
				sigma = symmetrize(T);
				computed_quantities[dataset][0] = -trace(sigma)/3.0;
			}
			else
			{
				computed_quantities[dataset][0] = 0.0;
			}
		}
	}
 };

template <int spacedim>
class GradUPostprocessor : public DataPostprocessorTensor<spacedim>
{

	/**
	 * Index of first component of displacement field
	 */
	const unsigned int
	global_component_index_u;


public:
	GradUPostprocessor(const unsigned int global_component_index_u)
	:
	DataPostprocessorTensor<spacedim> ("grad_u", update_gradients),
	global_component_index_u(global_component_index_u)
	{
	}

	void
	evaluate_vector_field(	const DataPostprocessorInputs::Vector<spacedim>&	input_data,
							vector<Vector<double>>&								computed_quantities)
	const
	{
		for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
		{
			for(unsigned int d = 0; d < spacedim; ++d)
				for (unsigned int e = 0; e < spacedim; ++e)
					computed_quantities[p][Tensor<2,spacedim>::component_to_unrolled_index(TableIndices<2>(d,e))] = input_data.solution_gradients[p][d + global_component_index_u][e];
      }
  }
};

// class defining an initial constant value for a scalar independent field
template<unsigned int spacedim>
class InitialConstant : public Function<spacedim>
{
private:
	const double
	c_0;

public:
	InitialConstant(const double c_0)
	:
	Function<spacedim>(1),
	c_0(c_0)
	{}

	double
	value(	const Point<spacedim>&	/*location*/,
			const unsigned int		/*component*/)
	const
	{
		return c_0;
	}
};

// class used for definition of electrical loading
template<unsigned int spacedim>
class ElectricalLoading : public TotalPotentialContribution<spacedim>
{
private:

	GlobalDataIncrementalFE<spacedim>&
	global_data;

public:

	double
	j_bar = 0.0;

	double
	phi_ap_bar = 0.0;

	double
	R_el = 0.0;

	/**
	 * 0 : prescribed current
	 * 1 : prescribed potential
	 * 2 : prescribed resistance
	 */
	unsigned int
	loading_type = 0;

	ElectricalLoading(	vector<const ScalarFunctional<spacedim, spacedim>*>		H_omega,
						vector<const ScalarFunctional<spacedim-1, spacedim>*>	H_sigma,
						vector<const IndependentField<0, spacedim>*>			C,
						GlobalDataIncrementalFE<spacedim>&						global_data)
	:
	TotalPotentialContribution<spacedim>(H_omega, H_sigma, C),
	global_data(global_data)
	{
	}

	bool get_potential_contribution(const Vector<double>&			H_omega_H_sigma_C,
									const vector<Vector<double>>&	/*C_ref_sets*/,
									double&							Pi,
									dealii::Vector<double>&			Pi_1,
									dealii::FullMatrix<double>&		Pi_2,
									const tuple<bool,bool,bool>&	requested_quantities)
	const
	{
		const double delta_t = global_data.get_t() - global_data.get_t_ref();
		const double phi_ap = H_omega_H_sigma_C[0];
		const double j_ap = H_omega_H_sigma_C[1];

		if(loading_type == 0)
		{
			if(get<0>(requested_quantities))
				Pi = -j_bar * phi_ap * delta_t + 0.5 * j_ap * j_ap;		// the part 0.5 j_ap * j_ap makes sure that j_ap = 0 during this type of loading (j_ap is not needed as unknown)

			if(get<1>(requested_quantities))
			{
				Pi_1[0] = -j_bar * delta_t;
				Pi_1[1] = j_ap;
			}

			if(get<2>(requested_quantities))
			{
				Pi_2(0,0) = 0.0;
				Pi_2(0,1) = Pi_2(1,0) = 0.0;
				Pi_2(1,1) = 1.0;
			}
		}
		else if(loading_type == 1)
		{
			if(get<0>(requested_quantities))
				Pi = -j_ap * (phi_ap - phi_ap_bar) * delta_t;	// enforce the boundary condition by the Lagrangian multiplier j_ap
																// (this has the advantage that the current corresponding to the potential is directly computed as part of the solution)

			if(get<1>(requested_quantities))
			{
				Pi_1[0] = -j_ap * delta_t;
				Pi_1[1] = - (phi_ap - phi_ap_bar) * delta_t;
			}

			if(get<2>(requested_quantities))
			{
				Pi_2(0,0) = 0.0;
				Pi_2(0,1) = Pi_2(1,0) = -delta_t;
				Pi_2(1,1) = 0.0;
			}
		}
		else if(loading_type == 2)
		{
			if(get<0>(requested_quantities))
				Pi = -0.5 / R_el * phi_ap * phi_ap * delta_t  + 0.5 * j_ap * j_ap;		// the part 0.5 j_ap * j_ap makes sure that j_ap = 0 during this type of loading (j_ap is not needed as unknown)

			if(get<1>(requested_quantities))
			{
				Pi_1[0] = -delta_t / R_el * phi_ap;
				Pi_1[1] = j_ap;
			}

			if(get<2>(requested_quantities))
			{
				Pi_2(0,0) = -delta_t / R_el;
				Pi_2(0,1) = Pi_2(1,0) = 0.0;
				Pi_2(1,1) = 1.0;
			}
		}
		else
		{
			Assert(false, ExcMessage("Unknown loading type!"));
		}


		return false;
	}
};


// main function
#ifdef PARALLEL
int main( int argc, char **argv )
#else
int main()
#endif
{

#ifdef PARALLEL
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
#endif

	const unsigned int spacedim = 3;	// this currently doesn't work for anything else than spacedim == 2

	const bool binary_se = false;		// if true, use relations for binary solid electrolyte

/**************
 * parameters *
 **************/

	// quantities used for normalization (in SI units)
	const double c_ast = 10000.0;
	const double D_ast = 1e-16;
	const double R_ast = 8.3144;
	const double T_ast = 293.15;
	const double F_ast = 96485.33;
	const double L_ast = 1e-6;

	// parameters of model (in SI units)
	const double L_ap = 15e-6 / L_ast;
	const double L_se = 15e-6 / L_ast;
	const double B_ap = 1.5e-6 / L_ast;
	const unsigned int N_ap = 5;

	const double dt_1 = 1800.0 * D_ast / (L_ast * L_ast);
	const double T = 293.15 / T_ast;

	const double c_Li_ref = 47500.0 / c_ast;
	const double c_Lip_ref = binary_se ? 750.0 / c_ast : 10000.0 / c_ast;
	const double c_LiX_ref = 750.0 / c_ast;

	const double lambda_se = binary_se ? 5e6 / (R_ast * T_ast * c_ast) : 57.7e9 / (R_ast * T_ast * c_ast);
	const double mu_se = binary_se ? 5e6 / (R_ast * T_ast * c_ast) : 38.5e9 / (R_ast * T_ast * c_ast);
	const double lambda_ap = 50.6e9 / (R_ast * T_ast * c_ast);
	const double mu_ap = 80e9 / (R_ast * T_ast * c_ast);
	const double deps_Li = -0.04;
	const double deps_LiX = 0.2;
	const double c_V = 50000.0 / c_ast;
	const double dmu_ap = 70000.0 / (R_ast * T_ast);
	const double D_Lip = binary_se ? 2.5e-13 / D_ast : 2.6e-12 / D_ast;
	const double D_X = 3.0e-13 / D_ast;
	const double D_Li = 5e-16 / D_ast;
	const double i_0_se_ap = 10.0 * L_ast / (F_ast * c_ast * D_ast);
	const double i_0_se_Li = 10.0 * L_ast / (F_ast * c_ast * D_ast);
	const double beta_se_ap = 0.5;
	const double beta_se_Li = 0.5;
	const double eta_bar_Li = 350000.0 / (R_ast * T_ast);
	const double R = 8.3144 / R_ast;
	const double F = 96485.33 / F_ast;

	const double R_ap = 0.5 * sqrt(10.0);
	const double h_ap = R_ap - B_ap;
	const double V_s = N_ap * 1.0/3.0 * numbers::PI * R_ap * R_ap * R_ap;
	const double V_kk = N_ap * 1.5 * numbers::PI / 3.0 * h_ap * h_ap * (3.0 * R_ap - h_ap);
	const double j_ap_bar = -1.0/dt_1 * (V_s - V_kk) * (c_Li_ref - (c_V - 1.0 * 0.5 * c_V)) * F;
	const double j_threshold_charging = 0.03;		// percentage of |j_ap_bar| at which constant voltage charging is stopped
	const double j_threshold_discharging = 0.03;	// percentage of |j_ap_bar| at which discharging is stopped
	const double phi_bar = eta_bar_Li - (R * T * log(0.5 * c_V / c_Li_ref) + dmu_ap * ( 0.5 - c_Li_ref / c_V ) ) / F;	// potential difference corresponding to 100 % charged state

	// numerical parameters
	const double eps_chemical = 1e-4;				// numerical parameter for regularization of chemical potential

	const unsigned int n_refinements_global = 2;	// number of global refinements of cathode part of mesh (2)
	const unsigned int n_refinements_sing_edge = 3;	// number of refinements at edge with stress singularity (3)
	const unsigned int n_refinements_anode = 2;		// number of refinements at anode side (required only for binary solid electrolyte to capture inhomogeneous ion distribution) (3)

	const double alpha = 0.5;						// time integration parameter alpha
	const unsigned int method = 2;					// time integration method (0: Miehe's method, 1: alpha family, 2: modified alpha family)
	const double inc_0 = dt_1 / 1000.0;				// initial time increment
	const double inc_max = dt_1 / 20.0;				// maximum time increment
	const double resistance_factor = 0.01;			// scaling factor for resistance

	const unsigned int degree = 1;					// degree of approximation of finite elements

	const unsigned int cell_divisions = 1;			// cell divisions for output

	// mappings
	MappingQGeneric<spacedim, spacedim> mapping_domain(1);
	MappingQGeneric<spacedim-1, spacedim> mapping_interface(1);

/**********************
 * independent fields *
 **********************/

	InitialConstant<spacedim> c_Li_initial(c_Li_ref);
	InitialConstant<spacedim> c_Lip_initial(c_Lip_ref);
	InitialConstant<spacedim> c_LiX_initial(c_LiX_ref);

	IndependentField<spacedim, spacedim> u("u", FE_Q<spacedim>(degree), spacedim, {0,1});					// displacement field (region 0 is solid electrolyte, region 1 is active material)
	IndependentField<spacedim, spacedim> c_Li("c_Li", FE_DGQ<spacedim>(degree), 1, {1}, &c_Li_initial);		// Lithium concentration in active material
	IndependentField<spacedim, spacedim> c_Lip("c_Lip", FE_DGQ<spacedim>(degree), 1, {0}, &c_Lip_initial);	// Lithium ion concentration in solid electrolyte
	IndependentField<spacedim, spacedim> c_LiX("c_LiX", FE_DGQ<spacedim>(degree), 1, {0}, &c_LiX_initial);	// salt concentration in solid electrolyte
	IndependentField<spacedim, spacedim> eta_Li("eta_Li", FE_Q<spacedim>(degree), 1, {1});					// chemical potential of Li in active material
	IndependentField<spacedim, spacedim> eta_Lip("eta_Lip", FE_Q<spacedim>(degree), 1, {0});				// electrochemical potential of Li+ ions in solid electrolyte
	IndependentField<spacedim, spacedim> eta_X("eta_X", FE_Q<spacedim>(degree), 1, {0});					// electrochemical potential of X- ions in solid electrolyte
	IndependentField<0, spacedim> phi_ap("phi_ap");															// electrical scalar potential of active particles
	IndependentField<0, spacedim> j_ap("j_ap");																// electrical current into active particles
	IndependentField<0, spacedim> constant_displacement("constant_displacement");							// constant displacement for periodic b.c.

/********************
 * dependent fields *
 ********************/

	// deformation gradient in solid electrolyte and active material
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
	if(spacedim == 3)
	{
		F_zz.add_term(1.0, u, 2, 2);
		F_yz.add_term(1.0, u, 1, 2);
		F_zy.add_term(1.0, u, 2, 1);
		F_zx.add_term(1.0, u, 2, 0);
		F_xz.add_term(1.0, u, 0, 2);
	}

	// Lithium ion concentration in solid electrolyte
	DependentField<spacedim, spacedim> c_Lip_("c_Lip_");
	if(binary_se)
		c_Lip_.add_term(1.0, c_Lip);
	else
		c_Lip_.add_term(c_Lip_ref);

	// X- concentration in solid electrolyte (determined by local electroneutrality)
	DependentField<spacedim, spacedim> c_X_("c_X_");
	c_X_.add_term(1.0, c_Lip);

	// salt concentration in solid electrolyte
	DependentField<spacedim, spacedim> c_LiX_("c_LiX_");
	c_LiX_.add_term(1.0, c_LiX);

	// sum of Lithium ion concentration and salt concentration in solid electrolyte
	DependentField<spacedim, spacedim> c_Lip_LipX_("c_Lip_LipX_");
	if(binary_se)
	{
		c_Lip_LipX_.add_term(1.0, c_LiX);
		c_Lip_LipX_.add_term(1.0, c_Lip);
	}
	else
		c_Lip_LipX_.add_term(c_Lip_ref);

	// Lithium concentration in active material
	DependentField<spacedim, spacedim> c_Li_("c_Li_");
	c_Li_.add_term(1.0, c_Li);

	// vacancy concentration in active material
	DependentField<spacedim, spacedim> c_V_("c_V");
	c_V_.add_term(-1.0, c_Li);
	c_V_.add_term(c_V);

	// chemical potential of Li in active material
	DependentField<spacedim, spacedim> eta_Li_("eta_Li_");
	DependentField<spacedim, spacedim> eta_Li_x("eta_Li_x");
	DependentField<spacedim, spacedim> eta_Li_y("eta_Li_y");
	DependentField<spacedim, spacedim> eta_Li_z("eta_Li_z");
	eta_Li_.add_term(1.0, eta_Li);
	eta_Li_x.add_term(1.0, eta_Li, 0, 0);
	eta_Li_y.add_term(1.0, eta_Li, 0, 1);
	if(spacedim == 3)
		eta_Li_z.add_term(1.0, eta_Li, 0, 2);

	// electrochemical potential of Li+ ions in solid electrolyte
	DependentField<spacedim, spacedim> eta_Lip_("eta_Lip_");
	DependentField<spacedim, spacedim> eta_Lip_x("eta_Lip_x");
	DependentField<spacedim, spacedim> eta_Lip_y("eta_Lip_y");
	DependentField<spacedim, spacedim> eta_Lip_z("eta_Lip_z");
	eta_Lip_.add_term(1.0, eta_Lip);
	eta_Lip_x.add_term(1.0, eta_Lip, 0, 0);
	eta_Lip_y.add_term(1.0, eta_Lip, 0, 1);
	if(spacedim == 3)
		eta_Lip_z.add_term(1.0, eta_Lip, 0, 2);

	// electrochemical potential of X- ions in solid electrolyte
	DependentField<spacedim, spacedim> eta_X_("eta_X_");
	DependentField<spacedim, spacedim> eta_X_x("eta_X_x");
	DependentField<spacedim, spacedim> eta_X_y("eta_X_y");
	DependentField<spacedim, spacedim> eta_X_z("eta_X_z");
	eta_X_.add_term(1.0, eta_X);
	eta_X_x.add_term(1.0, eta_X, 0, 0);
	eta_X_y.add_term(1.0, eta_X, 0, 1);
	if(spacedim == 3)
		eta_X_z.add_term(1.0, eta_X, 0, 2);

	// electrochemical potential of LiX salt in solid electrolyte (computed from dissociation equilibrium)
	DependentField<spacedim, spacedim> eta_LiX_("eta_LiX_");
	eta_LiX_.add_term(1.0, eta_Lip);
	eta_LiX_.add_term(1.0, eta_X);

	// potential difference on solid electrolyte - Lithium interface
	DependentField<spacedim-1, spacedim> deta_se_Li("deta_se_Li");
	deta_se_Li.add_term(-1.0, eta_Lip, 0, InterfaceSide::minus);
	deta_se_Li.add_term(eta_bar_Li);

	// potential difference on solid electrolyte - active material interface
	DependentField<spacedim-1, spacedim> deta_se_ap("deta_se_ap");
	deta_se_ap.add_term(-1.0, eta_Lip, 0, InterfaceSide::minus);
	deta_se_ap.add_term(1.0, eta_Li, 0, InterfaceSide::plus);
	deta_se_ap.add_term(F, phi_ap);

/********
 * grid *
 ********/

#ifdef PARALLEL
	dealii::parallel::distributed::Triangulation<spacedim> tria_domain_se (MPI_COMM_WORLD), tria_domain_ap(MPI_COMM_WORLD), tria_domain(MPI_COMM_WORLD);
#else
	dealii::Triangulation<spacedim> tria_domain_se, tria_domain_ap, tria_domain;
#endif

	GridIn<spacedim> grid_in;
	ifstream input_file("tria_domain_3d.vtk");
	grid_in.attach_triangulation(tria_domain);
	grid_in.read_vtk(input_file);
	input_file.close();

	// manifolds
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
		Point<spacedim> origin(0.0, 0.0, B_ap + 2.0 * B_ap * n);
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


	// mesh refinement

	for(unsigned int refinement_step = 0; refinement_step < n_refinements_global; ++refinement_step)
	{
		for(const auto& cell : tria_domain.active_cell_iterators())
		{
			if(cell->center()[2] > -B_ap)
				cell->set_refine_flag();
		}
		tria_domain.execute_coarsening_and_refinement();
	}

	for(unsigned int refinement_step = 0; refinement_step < n_refinements_anode; ++refinement_step)
	{
		for(const auto& cell : tria_domain.active_cell_iterators())
		{
			for(unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell; ++f)
			{
				if(cell->face(f)->center()[2] < -L_se + 1e-12)
					cell->set_refine_flag();
			}
		}
		tria_domain.execute_coarsening_and_refinement();
	}


	vector<pair<Point<spacedim>, unsigned int>> center_points;
	center_points.push_back(make_pair(Point<spacedim>(0.0, 0.0, 0.0), 2));
	for(unsigned int n = 0; n < N_ap; ++n)
	{
		center_points.push_back(make_pair(Point<spacedim>(0.0, B_ap, B_ap + n * 2.0 * B_ap), 1));
		center_points.push_back(make_pair(Point<spacedim>(0.0, 0.0, 2.0 * B_ap + n * 2.0 * B_ap), 2));
		center_points.push_back(make_pair(Point<spacedim>(B_ap, 0.0, B_ap + n * 2.0 * B_ap), 0));
	}
	for(unsigned int refinement_step = 0; refinement_step < n_refinements_sing_edge; ++refinement_step)
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
					if( ( fabs(vertex.distance(center_point.first) - B_ap / 3.0) < 1e-4 ) && ( fabs(vertex[plane] - plane_coord) < 1e-12 ) )
						cell->set_refine_flag();
				}
			}
		}
		tria_domain.execute_coarsening_and_refinement();
	}

	//triangulation system
#ifdef PARALLEL
	dealii::GalerkinTools::parallel::TriangulationSystem<spacedim> tria_system(tria_domain);
#else
	dealii::GalerkinTools::TriangulationSystem<spacedim> tria_system(tria_domain);
#endif

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
				else if(cell->face(face)->center()[0] > B_ap - 1e-12)
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
				else if(cell->face(face)->center()[1] > B_ap - 1e-12)
				{
					if(cell->material_id() == 0)
						tria_system.add_interface_cell(cell, face, 6);
					else
						tria_system.add_interface_cell(cell, face, 7);
				}
				else if(cell->face(face)->center()[2] < - L_se + 1e-12)
				{
					if(cell->material_id() == 0)
						tria_system.add_interface_cell(cell, face, 8);
					else
						tria_system.add_interface_cell(cell, face, 9);
				}
				else if(cell->face(face)->center()[2] > L_ap - 1e-12)
				{
					if(cell->material_id() == 0)
						tria_system.add_interface_cell(cell, face, 10);
					else
						tria_system.add_interface_cell(cell, face, 11);
				}
				else
				{
					Assert(false, ExcMessage("Not all cells included in boundary and interface mesh!"));
				}
			}
			else
			{
				if( (cell->material_id() == 0) && (cell->neighbor(face)->material_id() == 1) )
					tria_system.add_interface_cell(cell, face, 12);
			}
		}
	}

	//attach interface manifolds
	for(unsigned int n = 0; n < N_ap; ++n)
	{
		tria_system.set_interface_manifold(2 + 3 * n, spherical_manifold_interface[n]);
		tria_system.set_interface_manifold(3 + 3 * n, cylindrical_manifold_x_interface[n]);
		tria_system.set_interface_manifold(4 + 3 * n, cylindrical_manifold_y_interface[n]);
	}
	tria_system.set_interface_manifold(0, flat_manifold_interface);
	tria_system.set_interface_manifold(1, cylindrical_manifold_z_interface);

	tria_system.close();

/*************
 * potential *
 *************/

	// global data object for information transfer between different places
	GlobalDataIncrementalFE<spacedim> global_data;

	// psi_se
	KirchhoffMaterial00<spacedim> psi_se(	{F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz, c_Lip_LipX_},
											{0},
											QGauss<spacedim>(degree + 1),
											global_data,
											lambda_se,
											mu_se,
											binary_se ? deps_LiX : 0.0,
											binary_se ? c_Lip_ref + c_LiX_ref : c_Lip_ref,
											alpha);

	// psi_ap_e
	KirchhoffMaterial00<spacedim> psi_ap_e(	{F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz, c_Li_},
												{1},
												QGauss<spacedim>(degree + 1),
												global_data,
												lambda_ap,
												mu_ap,
												deps_Li,
												c_Li_ref,
												alpha);

	// psi_ap_c (part 1)
	PsiChemical00<spacedim> psi_ap_c_1(	{c_Li_},
										{1},
										QGauss<spacedim>(degree + 1),
										global_data,
										R*T, c_Li_ref, 0.0,
										alpha,
										eps_chemical);

	// psi_ap_c (part 2)
	PsiChemical00<spacedim> psi_ap_c_2(	{c_V_},
										{1},
										QGauss<spacedim>(degree + 1),
										global_data,
										R*T, c_V - c_Li_ref, 0.0,
										alpha,
										eps_chemical);

	// psi_ap_c (part 3)
	PsiChemical01<spacedim> psi_ap_c_3(	{c_Li_},
										{1},
										QGauss<spacedim>(degree + 1),
										global_data,
										dmu_ap/c_V, c_Li_ref,
										alpha);

	// psi_se_0_c (part 1)
	PsiChemical00<spacedim> psi_se_c_1(	{c_LiX_},
										{0},
										QGauss<spacedim>(degree + 1),
										global_data,
										R*T, c_LiX_ref, 0.0,
										alpha,
										eps_chemical);

	// psi_se_0_c (part 2)
	PsiChemical00<spacedim> psi_se_c_2(	{c_Lip_},
										{0},
										QGauss<spacedim>(degree + 1),
										global_data,
										2.0 * R*T, c_Lip_ref, 0.0,
										alpha,
										eps_chemical);

	// delta_se
	OmegaDualFluxDissipation00<spacedim> delta_se(	{eta_Lip_x, eta_Lip_y, eta_Lip_z, c_Lip_},
													{0},
													QGauss<spacedim>(degree + 1),
													global_data,
													D_Lip/(R*T),
													method,
													alpha);

	OmegaDualFluxDissipation00<spacedim> delta_se_X({eta_X_x, eta_X_y, eta_X_z, c_X_},
													{0},
													QGauss<spacedim>(degree + 1),
													global_data,
													D_X/(R*T),
													method,
													alpha);

	// delta_ap
	OmegaDualFluxDissipation00<spacedim> delta_ap(	{eta_Li_x, eta_Li_y, eta_Li_z, c_Li_},
													{1},
													QGauss<spacedim>(degree + 1),
													global_data,
													D_Li/(R*T),
													method,
													alpha);

	// mixed term for active material
	OmegaMixedTerm00<spacedim> omega_mixed_ap(	{c_Li_, eta_Li_},
												{1},
												QGauss<spacedim>(degree + 1),
												global_data,
												method,
												alpha);

	// mixed term for solid electrolyte (part 1)
	OmegaMixedTerm00<spacedim> omega_mixed_se_1(	{c_Lip_, eta_Lip_},
													{0},
													QGauss<spacedim>(degree + 1),
													global_data,
													method,
													alpha);

	// mixed term for solid electrolyte (part 2)
	OmegaMixedTerm00<spacedim> omega_mixed_se_2(	{c_X_, eta_X_},
													{0},
													QGauss<spacedim>(degree + 1),
													global_data,
													method,
													alpha);

	// mixed term for solid electrolyte (part 3)
	OmegaMixedTerm00<spacedim> omega_mixed_se_3(	{c_LiX_, eta_LiX_},
													{0},
													QGauss<spacedim>(degree + 1),
													global_data,
													method,
													alpha);

	// delta_se_Li
	OmegaDualButlerVolmer00<spacedim> delta_se_Li(	{deta_se_Li},
													{8},
													QGauss<spacedim-1>(degree + 1),
													global_data,
													i_0_se_Li * R * T / F, beta_se_Li, R * T, 20.0,
													method,
													alpha);

	// delta_se_ap
	OmegaDualButlerVolmer00<spacedim> delta_se_ap(	{deta_se_ap},
													{12},
													QGauss<spacedim-1>(degree + 1),
													global_data,
													i_0_se_ap * R * T / F, beta_se_ap, R * T, 20.0,
													method,
													alpha);

	// total potential contributions
	TotalPotentialContribution<spacedim> psi_se_tpc(psi_se);
	TotalPotentialContribution<spacedim> psi_ap_e_tpc(psi_ap_e);
	TotalPotentialContribution<spacedim> psi_ap_c_1_tpc(psi_ap_c_1);
	TotalPotentialContribution<spacedim> psi_ap_c_2_tpc(psi_ap_c_2);
	TotalPotentialContribution<spacedim> psi_ap_c_3_tpc(psi_ap_c_3);
	TotalPotentialContribution<spacedim> psi_se_c_1_tpc(psi_se_c_1);
	TotalPotentialContribution<spacedim> psi_se_c_2_tpc(psi_se_c_2);
	TotalPotentialContribution<spacedim> delta_se_tpc(delta_se);
	TotalPotentialContribution<spacedim> delta_se_X_tpc(delta_se_X);
	TotalPotentialContribution<spacedim> delta_ap_tpc(delta_ap);
	TotalPotentialContribution<spacedim> omega_mixed_ap_tpc(omega_mixed_ap);
	TotalPotentialContribution<spacedim> omega_mixed_se_1_tpc(omega_mixed_se_1);
	TotalPotentialContribution<spacedim> omega_mixed_se_2_tpc(omega_mixed_se_2);
	TotalPotentialContribution<spacedim> omega_mixed_se_3_tpc(omega_mixed_se_3);
	TotalPotentialContribution<spacedim> delta_se_Li_tpc(delta_se_Li);
	TotalPotentialContribution<spacedim> delta_se_ap_tpc(delta_se_ap);
	vector<const ScalarFunctional<spacedim,spacedim>*> H_omega_electrical_loading;
	vector<const ScalarFunctional<spacedim-1,spacedim>*> H_sigma_electrical_loading;
	vector<const IndependentField<0,spacedim>*> C_electrical_loading;
	C_electrical_loading.push_back(&phi_ap);
	C_electrical_loading.push_back(&j_ap);
	ElectricalLoading<spacedim> electrical_loading_tpc(H_omega_electrical_loading, H_sigma_electrical_loading, C_electrical_loading, global_data);

	// total potential
	TotalPotential<spacedim> total_potential;
 	total_potential.add_total_potential_contribution(psi_se_tpc);
 	total_potential.add_total_potential_contribution(psi_ap_e_tpc);
 	total_potential.add_total_potential_contribution(psi_ap_c_1_tpc);
 	total_potential.add_total_potential_contribution(psi_ap_c_2_tpc);
 	total_potential.add_total_potential_contribution(psi_ap_c_3_tpc);
 	total_potential.add_total_potential_contribution(delta_se_tpc);
	total_potential.add_total_potential_contribution(delta_ap_tpc);
	total_potential.add_total_potential_contribution(omega_mixed_ap_tpc);
	total_potential.add_total_potential_contribution(delta_se_Li_tpc);
	total_potential.add_total_potential_contribution(delta_se_ap_tpc);
	total_potential.add_total_potential_contribution(electrical_loading_tpc);
 	if(binary_se)
 	{
 	 	total_potential.add_total_potential_contribution(psi_se_c_1_tpc);
 	 	total_potential.add_total_potential_contribution(psi_se_c_2_tpc);
 		total_potential.add_total_potential_contribution(delta_se_X_tpc);
		total_potential.add_total_potential_contribution(omega_mixed_se_1_tpc);
		total_potential.add_total_potential_contribution(omega_mixed_se_2_tpc);
		total_potential.add_total_potential_contribution(omega_mixed_se_3_tpc);
 	}

	// add constraints
	// u_x
	DirichletConstraint<spacedim> dc_u_x_bottom(u, 0, InterfaceSide::minus, {0, 1});
	DirichletConstraint<spacedim> dc_u_x_top(u, 0, InterfaceSide::minus, {2, 3}, nullptr, &constant_displacement);

	// u_y
	DirichletConstraint<spacedim> dc_u_y_bottom(u, 1, InterfaceSide::minus, {4, 5});
	DirichletConstraint<spacedim> dc_u_y_top(u, 1, InterfaceSide::minus, {6, 7}, nullptr, &constant_displacement);

	Constraints<spacedim> constraints;
	constraints.add_dirichlet_constraint(dc_u_x_bottom);
	constraints.add_dirichlet_constraint(dc_u_x_top);
	constraints.add_dirichlet_constraint(dc_u_y_bottom);
	constraints.add_dirichlet_constraint(dc_u_y_top);

/***************************************************
 * set up finite element model and do computations *
 ***************************************************/

#ifdef PARALLEL
	SolverWrapperPETSc solver_wrapper;
	FEModel<spacedim, LinearAlgebra::distributed::Vector<double>, PETScWrappers::MPI::BlockVector, GalerkinTools::parallel::TwoBlockMatrix<PETScWrappers::MPI::SparseMatrix>> fe_model(total_potential, tria_system, mapping_domain, mapping_interface, global_data, constraints, solver_wrapper);
#else
	BlockSolverWrapperUMFPACK solver_wrapper;
	FEModel<spacedim, Vector<double>, BlockVector<double>, GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>> fe_model(total_potential, tria_system, mapping_domain, mapping_interface, global_data, constraints, solver_wrapper);
#endif

	AffineConstraints<double> custom_constraints(fe_model.get_assembly_helper().get_locally_relevant_indices());

	Point<spacedim> p_u_z;
	p_u_z(0) = 0.0;
	p_u_z(1) = 0.0;
	p_u_z(2) = -L_se;
	const unsigned int dof_index_u_z = fe_model.get_assembly_helper().get_dof_index_at_point_omega(&u, 2, p_u_z);
	if(dof_index_u_z != numbers::invalid_size_type)
	{
		custom_constraints.add_line(dof_index_u_z);
	}
	else
	{
#ifndef PARALLEL
		cout << "Not able to enforce displacement constraint in z-direction" << endl;
#endif
	}

	// postprocessors
	vector<unsigned int> component_indices_c_se;
	if(binary_se)
	{
		component_indices_c_se.push_back(fe_model.get_assembly_helper().get_u_omega_global_component_index(c_Lip));
		component_indices_c_se.push_back(fe_model.get_assembly_helper().get_u_omega_global_component_index(c_LiX));
	}
	PressurePostprocessor<spacedim> stress_postproc_se(	"sigma_se",
														fe_model.get_assembly_helper().get_u_omega_global_component_index(u),
														component_indices_c_se,
														0,
														lambda_se,
														mu_se,
														binary_se ? deps_LiX : 0.0,
														c_Lip_ref + c_LiX_ref);
	fe_model.attach_data_postprocessor_domain(stress_postproc_se);

	PressurePostprocessor<spacedim> stress_postproc_ap(	"sigma_ap",
														fe_model.get_assembly_helper().get_u_omega_global_component_index(u),
														{fe_model.get_assembly_helper().get_u_omega_global_component_index(c_Li)},
														1,
														lambda_ap,
														mu_ap,
														deps_Li,
														c_Li_ref);
	fe_model.attach_data_postprocessor_domain(stress_postproc_ap);

	GradUPostprocessor<spacedim> grad_u(fe_model.get_assembly_helper().get_u_omega_global_component_index(u));
	fe_model.attach_data_postprocessor_domain(grad_u);

	const unsigned int dof_index_phi_ap = fe_model.get_assembly_helper().get_global_dof_index_C(&phi_ap);
	const unsigned int dof_index_j_ap = fe_model.get_assembly_helper().get_global_dof_index_C(&j_ap);

	// vector for (t, phi_ap, j_ap)
	vector<tuple<double, double, double>> phi_j;

	// iteration settings
	global_data.set_max_iter(20);
	global_data.set_max_cutbacks(1);

// first step
	electrical_loading_tpc.loading_type = 0;
	electrical_loading_tpc.j_bar = j_ap_bar;
	double inc = inc_0;
	for(;;)
	{
		// predict cutoff time
		double cutoff_time = dt_1;
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
			new_time = cutoff_time;

		cout << "Old time=" << global_data.get_t() << endl;
		cout << "New time=" << new_time << endl;

		const int iter = fe_model.do_time_step(new_time, custom_constraints);
		if(iter >= 0)
		{
			fe_model.write_output_independent_fields("results/domain", "results/interface", cell_divisions);
			phi_j.push_back(make_tuple(new_time, fe_model.get_solution_vector()(dof_index_phi_ap), j_ap_bar));
			if(new_time == cutoff_time)
				break;
			if(iter < 4)
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
	electrical_loading_tpc.phi_ap_bar = phi_bar;
	for(;;)
	{
		double new_time = global_data.get_t() + inc;
		cout << "Old time=" << global_data.get_t() << endl;
		cout << "New time=" << new_time << endl;

		const int iter = fe_model.do_time_step(new_time, custom_constraints);
		if(iter >= 0)
		{
			fe_model.write_output_independent_fields("results/domain", "results/interface", cell_divisions);
			phi_j.push_back(make_tuple(new_time, fe_model.get_solution_vector()(dof_index_phi_ap), fe_model.get_solution_vector()(dof_index_j_ap)));
			if( fabs(fe_model.get_solution_vector()(dof_index_j_ap)) <= fabs(j_threshold_charging * j_ap_bar) )
				break;
			if(iter < 4)
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

// third step
	inc = inc_0 * resistance_factor;
	electrical_loading_tpc.loading_type = 2;
	electrical_loading_tpc.R_el = fabs(fe_model.get_solution_vector()(dof_index_phi_ap) / (j_ap_bar)) * resistance_factor;
	for(;;)
	{
		double new_time = global_data.get_t() + inc;
		cout << "Old time=" << global_data.get_t() << endl;
		cout << "New time=" << new_time << endl;

		const int iter = fe_model.do_time_step(new_time, custom_constraints);
		if(iter >= 0)
		{
			fe_model.write_output_independent_fields("results/domain", "results/interface", cell_divisions);
			phi_j.push_back(make_tuple(new_time, fe_model.get_solution_vector()(dof_index_phi_ap), -fe_model.get_solution_vector()(dof_index_phi_ap)/electrical_loading_tpc.R_el));
			if( fabs(fe_model.get_solution_vector()(dof_index_phi_ap)/electrical_loading_tpc.R_el) <= fabs(j_threshold_discharging * j_ap_bar) )
				break;
			if(iter < 4)
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
