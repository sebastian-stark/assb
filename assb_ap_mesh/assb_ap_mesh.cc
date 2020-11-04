#include <iostream>
#include <fstream>
#include <map>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

using namespace std;
using namespace dealii;

// class allowing transformation of solid electrolyte part
class Warp
{
private:

	/**
	 * old length
	 */
	const double
	L_old;

	/**
	 * new length
	 */
	const double
	L_new;


public:
	Warp(const double L_old,
		 const double L_new)
	:
	L_old(L_old),
	L_new(L_new)
	{
	}

	Point<3>
	operator()(const Point<3>& point)
	const
	{
		if(point(2) > 0)
			return point;
		else
		{
			double z = point(2);
			return Point<3>(point(0), point(1), z - (L_new - L_old) / L_old / L_old  * z * z);
		}
	}
};


void get_mesh_parts_3d(Triangulation<3>& tria_ap, Triangulation<3>& tria_se_0, Triangulation<3>& tria_se_1)
{
	const unsigned int spacedim = 3;

	vector<Triangulation<spacedim>> tria_parts;

// build the first part of the active particle triangulation (take this from the tethexed gmsh file)
	tria_parts.push_back(Triangulation<spacedim>(Triangulation<spacedim>::MeshSmoothing::none, true));
	GridIn<spacedim> grid_in;
	ifstream input_file("brick_split_in_5_tests_tethexed.msh");
	grid_in.attach_triangulation(tria_parts[0]);
	grid_in.read_msh(input_file);

	// shift vertices to surface of spherical inclusion
	const double new_pos_1 = 3.0/2.0/sqrt(2.0) - 0.5;
	const double new_pos_2 = (sqrt(10.0/3.0) - 1.0)*0.5;
	for(auto& cell : tria_parts[0].active_cell_iterators())
	{
		for(unsigned int v = 0; v < GeometryInfo<spacedim>::vertices_per_cell; ++v)
		{
			auto& p = cell->vertex(v);
			if( ( fabs(p[0] - 0.5) < 1e-8) && ( fabs(p[1] - 0.5) < 1e-8) && ( fabs(p[2] - 0.0) < 1e-8) )
				p[0] = p[1] = new_pos_1;
			else if( ( fabs(p[0] - 0.5) < 1e-8) && ( fabs(p[1] - 0.0) < 1e-8) && ( fabs(p[2] - 0.5) < 1e-8) )
				p[0] = p[2] = new_pos_1;
			else if( ( fabs(p[0] - 0.0) < 1e-8) && ( fabs(p[1] - 0.5) < 1e-8) && ( fabs(p[2] - 0.5) < 1e-8) )
				p[1] = p[2] = new_pos_1;
			else if( ( fabs(p[0] - 1.0/3.0) < 1e-8) && ( fabs(p[1] - 1.0/3.0) < 1e-8) && ( fabs(p[2] - 1.0/3.0) < 1e-8) )
				p[0] = p[1] = p[2] = new_pos_2;
		}
	}

// build the second, third, fourth part of the triangulation
	map<unsigned int, unsigned int> vertex_to_vertex;
	vector<Point<spacedim>> vertices_x, vertices_y, vertices_z;
	vector<CellData<spacedim>> cell_data_x, cell_data_y, cell_data_z;
	for(auto& cell : tria_parts[0].active_cell_iterators())
	{
		for(unsigned int f = 0; f <  GeometryInfo<spacedim>::faces_per_cell; ++f)
		{
			const auto& face = cell->face(f);
			if(face->at_boundary())
			{
				if(face->center()[0] < 1e-8)
				{
					CellData<spacedim> cell_data_;
					for(unsigned int v = 0; v < GeometryInfo<spacedim>::vertices_per_face; ++v)
					{
						if(vertex_to_vertex.find(face->vertex_index(v)) == vertex_to_vertex.end())
						{
							vertex_to_vertex.insert(make_pair(face->vertex_index(v), vertex_to_vertex.size()));
							vertices_x.push_back(face->vertex(v));
						}
						cell_data_.vertices[v] = vertex_to_vertex[face->vertex_index(v)];
					}
					cell_data_x.push_back(cell_data_);
				}
			}
		}
	}

	unsigned int vertex_count = vertices_x.size();
	for(unsigned int v = 0; v < vertex_count; ++v)
	{
		vertices_x.push_back(vertices_x[v]);
		vertices_x[v][0] -= 0.5;
	}

	for(auto& cell_data_ : cell_data_x)
		for(unsigned int v = 0; v < GeometryInfo<spacedim>::vertices_per_face; ++v)
			cell_data_.vertices[v + GeometryInfo<spacedim>::vertices_per_face] = cell_data_.vertices[v] + vertex_count;

	for(const auto& vertex : vertices_x)
	{
		const Point<spacedim> vertex_swapped(vertex[1], vertex[2], vertex[0]);
		vertices_y.push_back(Point<spacedim>(vertex[1], vertex[2], vertex[0]));
		vertices_z.push_back(Point<spacedim>(vertex[2], vertex[0], vertex[1]));
	}

	for(const auto& cell : cell_data_x)
	{
		CellData<spacedim> cell_data_;
		cell_data_.vertices[0] = cell.vertices[0];
		cell_data_.vertices[2] = cell.vertices[1];
		cell_data_.vertices[4] = cell.vertices[2];
		cell_data_.vertices[6] = cell.vertices[3];
		cell_data_.vertices[1] = cell.vertices[4];
		cell_data_.vertices[3] = cell.vertices[5];
		cell_data_.vertices[5] = cell.vertices[6];
		cell_data_.vertices[7] = cell.vertices[7];
		cell_data_y.push_back(cell_data_);

		cell_data_.vertices[0] = cell.vertices[0];
		cell_data_.vertices[4] = cell.vertices[1];
		cell_data_.vertices[1] = cell.vertices[2];
		cell_data_.vertices[5] = cell.vertices[3];
		cell_data_.vertices[2] = cell.vertices[4];
		cell_data_.vertices[6] = cell.vertices[5];
		cell_data_.vertices[3] = cell.vertices[6];
		cell_data_.vertices[7] = cell.vertices[7];
		cell_data_z.push_back(cell_data_);
	}

	tria_parts.push_back(Triangulation<spacedim>(Triangulation<spacedim>::MeshSmoothing::none, true));
	tria_parts.back().create_triangulation(vertices_x, cell_data_x, SubCellData());

	tria_parts.push_back(Triangulation<spacedim>(Triangulation<spacedim>::MeshSmoothing::none, true));
	tria_parts.back().create_triangulation(vertices_y, cell_data_y, SubCellData());

	tria_parts.push_back(Triangulation<spacedim>(Triangulation<spacedim>::MeshSmoothing::none, true));
	tria_parts.back().create_triangulation(vertices_z, cell_data_z, SubCellData());

// build the remaining parts of the triangulation
	tria_parts.push_back(Triangulation<spacedim>(Triangulation<spacedim>::MeshSmoothing::none, true));
	const vector<unsigned int> repetitions_x = {3, 1, 1};
	GridGenerator::subdivided_hyper_rectangle(tria_parts.back(), repetitions_x, Point<spacedim>(-0.5, -0.5, -0.5), Point<spacedim>(1.0, 0.0, 0.0));

	tria_parts.push_back(Triangulation<spacedim>(Triangulation<spacedim>::MeshSmoothing::none, true));
	const vector<unsigned int> repetitions_y = {1, 2, 1};
	GridGenerator::subdivided_hyper_rectangle(tria_parts.back(), repetitions_y, Point<spacedim>(-0.5, 0.0, -0.5), Point<spacedim>(0.0, 1.0, 0.0));

	tria_parts.push_back(Triangulation<spacedim>(Triangulation<spacedim>::MeshSmoothing::none, true));
	const vector<unsigned int> repetitions_z = {1, 1, 2};
	GridGenerator::subdivided_hyper_rectangle(tria_parts.back(), repetitions_z, Point<spacedim>(-0.5, -0.5, 0.0), Point<spacedim>(0.0, 0.0, 1.0));

//m erge the parts together to form one eighth of the active particle surrounded by solid electrolyte
	Triangulation<spacedim> tria_1(Triangulation<spacedim>::MeshSmoothing::none, true);
	GridGenerator::merge_triangulations({&tria_parts[0], &tria_parts[1], &tria_parts[2], &tria_parts[3], &tria_parts[4], &tria_parts[5], &tria_parts[6]}, tria_1);

// shift origin such that one corner of mesh coincides with (0, 0, 0)
	Tensor<1,spacedim> shift_vector;
	shift_vector[0] = 0.5;
	shift_vector[1] = 0.5;
	shift_vector[2] = 0.5;
	GridTools::shift(shift_vector, tria_1);

// make some further adjustments to vertex positions in order to make the shape of the active particle spherical
	const double new_pos_3 = 0.5/sqrt(2.0);
	for(auto& cell : tria_1.active_cell_iterators())
	{
		for(unsigned int v = 0; v < GeometryInfo<spacedim>::vertices_per_cell; ++v)
		{
			auto& p = cell->vertex(v);
			if( ( fabs(p[0] - 0.5) < 1e-8) && ( fabs(p[1] - 0.5) < 1e-8) && ( fabs(p[2] - 1.5) < 1e-8) )
				p[0] = p[1] = new_pos_3;
			else if( ( fabs(p[0] - 0.5) < 1e-8) && ( fabs(p[1] - 1.5) < 1e-8) && ( fabs(p[2] - 0.5) < 1e-8) )
				p[0] = p[2] = new_pos_3;
			else if( ( fabs(p[0] - 1.5) < 1e-8) && ( fabs(p[1] - 0.5) < 1e-8) && ( fabs(p[2] - 0.5) < 1e-8) )
				p[1] = p[2] = new_pos_3;
		}
	}

	const double new_pos_4 = sqrt(5.0)/2.0;
	for(auto& cell : tria_1.active_cell_iterators())
	{
		for(unsigned int v = 0; v < GeometryInfo<spacedim>::vertices_per_cell; ++v)
		{
			auto& p = cell->vertex(v);
			if( ( fabs(p[0] - 0.0) < 1e-8) && ( fabs(p[1] - 3.0/2.0/sqrt(2.0)) < 1e-8) && ( fabs(p[2] - 3.0/2.0/sqrt(2.0)) < 1e-8) )
			{
				p[1] = p[2] = new_pos_4;
			}
			else if( ( fabs(p[0] - 3.0/2.0/sqrt(2.0)) < 1e-8) && ( fabs(p[1] - 0.0) < 1e-8) && ( fabs(p[2] - 3.0/2.0/sqrt(2.0)) < 1e-8) )
			{
				p[0] = p[2] = new_pos_4;
			}
			else if( ( fabs(p[0] - 3.0/2.0/sqrt(2.0)) < 1e-8) && ( fabs(p[1] - 3.0/2.0/sqrt(2.0)) < 1e-8) && ( fabs(p[2] - 0.0) < 1e-8) )
			{
				p[0] = p[1] = new_pos_4;
			}
		}
	}

// now make a copy of the octant, rotate and merge in order to get one quarter of the active particle
	Triangulation<spacedim> tria_2;
	tria_2.copy_triangulation(tria_1);
	GridTools::rotate(1.5*numbers::PI, 0, tria_2);

	GridGenerator::merge_triangulations({&tria_1, &tria_2}, tria_ap, 1e-12, true);

	// shift triangulation to origin
	shift_vector[0] = 0.0;
	shift_vector[1] = 0.0;
	shift_vector[2] = 1.5;
	GridTools::shift(shift_vector, tria_ap);

// now make mesh of solid electrolyte
	vertex_to_vertex.clear();
	vertices_z.clear();
	cell_data_z.clear();
	for(auto& cell : tria_ap.active_cell_iterators())
	{
		for(unsigned int f = 0; f <  GeometryInfo<spacedim>::faces_per_cell; ++f)
		{
			const auto& face = cell->face(f);
			if(face->at_boundary())
			{
				if(face->center()[2] < 1e-8)
				{
					CellData<spacedim> cell_data_;
					for(unsigned int v = 0; v < GeometryInfo<spacedim>::vertices_per_face; ++v)
					{
						if(vertex_to_vertex.find(face->vertex_index(v)) == vertex_to_vertex.end())
						{
							vertex_to_vertex.insert(make_pair(face->vertex_index(v), vertex_to_vertex.size()));
							vertices_z.push_back(face->vertex(v));
						}
						cell_data_.vertices[v] = vertex_to_vertex[face->vertex_index(v)];
					}
					cell_data_z.push_back(cell_data_);
				}
			}
		}
	}

	vertex_count = vertices_z.size();
	for(unsigned int v = 0; v < vertex_count; ++v)
	{
		vertices_z.push_back(vertices_z[v]);
		vertices_z.back()[2] += 0.5;
	}

	for(auto& cell_data_ : cell_data_z)
		for(unsigned int v = 0; v < GeometryInfo<spacedim>::vertices_per_face; ++v)
			cell_data_.vertices[v + GeometryInfo<spacedim>::vertices_per_face] = cell_data_.vertices[v] + vertex_count;


	Triangulation<spacedim> tria_se(Triangulation<spacedim>::MeshSmoothing::none, false);
	tria_se.create_triangulation(vertices_z, cell_data_z, SubCellData());

	for(auto& cell : tria_se.active_cell_iterators())
	{
		for(unsigned int v = 0; v < GeometryInfo<spacedim>::vertices_per_cell; ++v)
		{
			auto& p = cell->vertex(v);
			if( ( fabs(p[0] - 0.5/sqrt(2.0)) < 1e-8) && ( fabs(p[1] - 0.5/sqrt(2.0)) < 1e-8 ) && (fabs(p[2]) < 1e-8) )
				p[0] = p[1] = 0.5;
		}
	}

	shift_vector[0] = 0.0;
	shift_vector[1] = 0.0;
	shift_vector[2] = -0.5;
	GridTools::shift(shift_vector, tria_se);

	tria_se.set_all_manifold_ids(0);
	for(const auto& cell : tria_se.active_cell_iterators())
	{
		cell->set_material_id(0);
		for(unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell; ++f)
		{
			if( (cell->face(f)->center()[2] > - 1e-8) && (cell->face(f)->center()[0] < 0.5) && (cell->face(f)->center()[1] < 0.5))
				for(unsigned int l = 0; l < GeometryInfo<spacedim>::lines_per_face; ++l)
					if( ( cell->face(f)->line(l)->center()[0] > 0.26 - 1e-8) || ( cell->face(f)->line(l)->center()[1] > 0.26 - 1e-8) )
						cell->face(f)->line(l)->set_manifold_id(4);
		}
	}

	tria_se_0.copy_triangulation(tria_se);

	tria_se.set_all_manifold_ids(0);
	for(auto& cell : tria_se.active_cell_iterators())
	{
		for(unsigned int v = 0; v < GeometryInfo<spacedim>::vertices_per_cell; ++v)
		{
			auto& p = cell->vertex(v);
			if( ( fabs(p[0] - 0.5/sqrt(2.0)) < 1e-8) && ( fabs(p[1] - 0.5/sqrt(2.0)) < 1e-8 ) && (fabs(p[2]) < 1e-8) )
			{
				p[0] = p[1] = 0.5;
			}
		}
	}
	shift_vector[0] = 0.0;
	shift_vector[1] = 0.0;
	shift_vector[2] = -0.5;
	GridTools::shift(shift_vector, tria_se);

	tria_se_1.copy_triangulation(tria_se);

}

void attach_material_ids_manifold_ids_ap_3d(Triangulation<3>& tria_ap, const unsigned int n)
{
	const unsigned int spacedim = 3;

	const double z_offset = 1.5 + 3.0 * n;

	// shift such that active particle is centered at origin
	Tensor<1,spacedim> shift_vector;
	shift_vector[0] = 0.0;
	shift_vector[1] = 0.0;
	shift_vector[2] = -z_offset;
	GridTools::shift(shift_vector, tria_ap);

	// material id's
	Point<spacedim> origin;
	const double r = sqrt(10)/2.0;
	for(const auto& cell : tria_ap.active_cell_iterators())
	{
		if(cell->center().distance(origin) < r)
			cell->set_material_id(1);
		else
			cell->set_material_id(0);
	}

	// start with setting all manifolds of the active particle to a spherical manifold
	tria_ap.set_all_manifold_ids(0);
	for(const auto& cell : tria_ap.active_cell_iterators())
		if(cell->material_id() == 1)
			cell->set_all_manifold_ids(2 + n*3);

	// set manifold id's of some inner parts of active particle to cylindrical manifold
	for(const auto& cell : tria_ap.active_cell_iterators())
	{
		if(cell->material_id() == 1)
		{
			for(unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell; ++f)
			{
				if( (cell->face(f)->center()[0] < 1e-8) && (cell->face(f)->center().distance(origin) > 1.0/sqrt(2.0) ) && ( fabs( cell->face(f)->center()[1] - fabs(cell->face(f)->center()[2]) ) < 1e-8 ) )
					cell->set_all_manifold_ids(3+n*3);

				if( (cell->face(f)->center()[1] < 1e-8) && (cell->face(f)->center().distance(origin) > 1.0/sqrt(2.0) ) && ( fabs( cell->face(f)->center()[0] - fabs(cell->face(f)->center()[2]) ) < 1e-8 ) )
					cell->set_all_manifold_ids(4+n*3);

				if( (fabs(cell->face(f)->center()[2]) < 1e-8) && (cell->face(f)->center().distance(origin) > 1.0/sqrt(2.0) ) && ( fabs( cell->face(f)->center()[0] - cell->face(f)->center()[1] ) < 1e-8 ) )
					cell->set_all_manifold_ids(1);
			}
		}
	}

	// set manifold id's of some inner parts of active particles to flat manifold
	for(const auto& cell : tria_ap.active_cell_iterators())
	{
		if(cell->material_id() == 1)
		{
			const auto& center = cell->center();
			if( ( (center[0] < 0.5) && (center[1] < 0.5) ) || ( (center[0] < 0.5) && (fabs(center[2]) < 0.5) ) || ( (center[1] < 0.5) && (fabs(center[2]) < 0.5) ) )
				cell->set_all_manifold_ids(0);
		}
	}

	// set manifold id's of cutting surface of active particle to cylindrical manifold
	for(const auto& cell : tria_ap.active_cell_iterators())
	{
		if(cell->material_id() == 1)
		{
			for(unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell; ++f)
			{
				if(cell->face(f)->center()[0] > 1.5 - 1e-8)
					for(unsigned int l = 0; l < GeometryInfo<spacedim>::lines_per_face; ++l)
						if( ( cell->face(f)->line(l)->center()[1] > 0.26 - 1e-8) || ( fabs(cell->face(f)->line(l)->center()[2]) > 0.26 - 1e-8) )
							cell->face(f)->line(l)->set_manifold_id(3+n*3);

				if(cell->face(f)->center()[1] > 1.5 - 1e-8)
					for(unsigned int l = 0; l < GeometryInfo<spacedim>::lines_per_face; ++l)
						if( ( cell->face(f)->line(l)->center()[0] > 0.26 - 1e-8) || ( fabs(cell->face(f)->line(l)->center()[2]) > 0.26 - 1e-8) )
							cell->face(f)->line(l)->set_manifold_id(4+n*3);

				if(fabs(cell->face(f)->center()[2]) > 1.5 - 1e-8)
					for(unsigned int l = 0; l < GeometryInfo<spacedim>::lines_per_face; ++l)
						if( ( cell->face(f)->line(l)->center()[0] > 0.26 - 1e-8) || ( cell->face(f)->line(l)->center()[1] > 0.26 - 1e-8) )
							cell->face(f)->line(l)->set_manifold_id(1);
			}
		}
	}

	// shift active particle back to original position
	shift_vector[2] = z_offset;
	GridTools::shift(shift_vector, tria_ap);

}

void attach_material_ids_manifold_ids_se_0_3d(Triangulation<3>& tria_se_0)
{
	const unsigned int spacedim = 3;

	tria_se_0.set_all_manifold_ids(0);

	for(const auto& cell : tria_se_0.active_cell_iterators())
	{
		cell->set_material_id(0);
		for(unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell; ++f)
		{
			if( (cell->face(f)->center()[2] > - 1e-8) && (cell->face(f)->center()[0] < 0.5) && (cell->face(f)->center()[1] < 0.5))
				for(unsigned int l = 0; l < GeometryInfo<spacedim>::lines_per_face; ++l)
					if( ( cell->face(f)->line(l)->center()[0] > 0.26 - 1e-8) || ( cell->face(f)->line(l)->center()[1] > 0.26 - 1e-8) )
						cell->face(f)->line(l)->set_manifold_id(1);
		}
	}
}

double make_mesh_3d(const unsigned int 	N_ap,
					const unsigned int	N_se,
					const double 		L_ap,
					const double 		L_se,
					Triangulation<3>&	tria_domain_3d)
{
	const unsigned int spacedim = 3;

	Triangulation<spacedim> tria_ap, tria_se_0, tria_se_1;
	vector<Triangulation<spacedim>> tria_parts;

	get_mesh_parts_3d(tria_ap, tria_se_0, tria_se_1);

	// active particles
	GridIn<spacedim> grid_in_1;
	for(unsigned int n = 0; n < N_ap; ++n)
	{
		// add active particle
		tria_parts.push_back(Triangulation<spacedim>());
		tria_parts.back().copy_triangulation(tria_ap);

		// shift active particle to correct position
		Tensor<1,spacedim> shift_vector;
		shift_vector[0] = 0.0;
		shift_vector[1] = 0.0;
		shift_vector[2] = 3.0 * n;
		GridTools::shift(shift_vector, tria_parts.back());

		// assign manifold id's
		attach_material_ids_manifold_ids_ap_3d(tria_parts.back(), n);

	}

	// solid electrolyte transition layer
	tria_parts.push_back(Triangulation<spacedim>());
	tria_parts.back().copy_triangulation(tria_se_0);
	// assign manifold id's
	attach_material_ids_manifold_ids_se_0_3d(tria_parts.back());

	// solid electrolyte extra layers
	for(unsigned int n = 1; n < N_se; ++n)
	{
		//add solid electrolyte layer
		tria_parts.push_back(Triangulation<spacedim>());
		tria_parts.back().copy_triangulation(tria_se_1);

		//shift solid electrolyte layer to correct position
		Tensor<1,spacedim> shift_vector;
		shift_vector[0] = 0.0;
		shift_vector[1] = 0.0;
		shift_vector[2] = -0.5 * (n - 1);
		GridTools::shift(shift_vector, tria_parts.back());

		//assign manifold id's
		tria_parts.back().set_all_manifold_ids(0);
	}

	// now merge everything
	for(const auto& tria_part : tria_parts)
		GridGenerator::merge_triangulations(tria_part, tria_domain_3d, tria_domain_3d, 1e-12, true);

	// now warp and scale to get the right dimensions
	const double L_se_old = N_se * 0.5;
	const double L_se_new = N_ap * 3.0 * L_se / L_ap;
	Warp warp(L_se_old, L_se_new);
	GridTools::transform(warp, tria_domain_3d);
	const double L_warped = L_se_new + N_ap * 3.0;
	const double scale_factor = (L_se + L_ap)/L_warped;
	GridTools::scale(scale_factor, tria_domain_3d);

	return scale_factor;
}

int main()
{
// 2d or 3d mesh
	const unsigned int mesh_dimension = 3;

// make 3d mesh, which is used for extraction of 2d mesh
	const unsigned int N_ap = 5;
	const unsigned int N_se = 20;
	const double L_ap = 15.0;
	const double L_se = 15.0;
	Triangulation<3>	tria_domain_3d;
	const double scale_factor = make_mesh_3d(N_ap, N_se, L_ap, L_se, tria_domain_3d);

	if(mesh_dimension == 3)
	{
		ofstream output_file("tria_domain_3d.vtk");
		GridOut grid_out;
		grid_out.write_vtk(tria_domain_3d, output_file);
		return 0;
	}

// extract the 2d mesh if necessary
	const unsigned int spacedim = 3;

	for(const auto& cell : tria_domain_3d.active_cell_iterators())
	{
		for(unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell; ++f)
		{
			if(cell->face(f)->at_boundary())
			{
				if(cell->face(f)->center()[0] < 1e-12)
					cell->face(f)->set_all_boundary_ids(1);
				else
					cell->face(f)->set_all_boundary_ids(0);
			}
		}
	}

	dealii::Triangulation<spacedim-1, spacedim> tria_domain_3d_boundary;
	GridGenerator::extract_boundary_mesh(tria_domain_3d, tria_domain_3d_boundary, {1});
	GridTools::rotate(numbers::PI/2.0, 1, tria_domain_3d_boundary);

	dealii::Triangulation<spacedim-1, spacedim-1> tria_domain;
	GridGenerator::flatten_triangulation(tria_domain_3d_boundary, tria_domain);

	tria_domain.set_all_manifold_ids(0);
	for(const auto& cell : tria_domain.active_cell_iterators())
		cell->set_material_id(0);
	Point<spacedim-1> center(1.5 * scale_factor, 0.0);
	for(unsigned int m = 0; m < N_ap; ++m)
	{
		const double r = sqrt(10)/2.0 * scale_factor;
		for(const auto& cell : tria_domain.active_cell_iterators())
		{
			if(cell->center().distance(center) < r)
			{
				cell->set_material_id(1);
				cell->set_all_manifold_ids(m + 1);
			}
		}
		for(const auto& cell : tria_domain.active_cell_iterators())
		{
			if( (cell->center()[1] < 0.5 * scale_factor) || (fabs(cell->center()[0] - center[0]) < 0.5 * scale_factor) )
				cell->set_all_manifold_ids(0);
		}


		center[0] += 3.0 * scale_factor;
	}

// attach manifolds
	vector<SphericalManifold<spacedim-1>> spherical_manifold_domain;
	vector<SphericalManifold<spacedim-2, spacedim-1>> spherical_manifold_interface;
	FlatManifold<spacedim-1> flat_manifold_domain;
	for(unsigned int n = 0; n < N_ap; ++n)
	{
		Point<spacedim-1> origin((1.5 + 3.0 * n) * scale_factor, 0.0);
		spherical_manifold_domain.push_back(SphericalManifold<spacedim-1>(origin));
		spherical_manifold_interface.push_back(SphericalManifold<spacedim-2, spacedim-1>(origin));
		tria_domain.set_manifold(n + 1, spherical_manifold_domain.back());
	}
	tria_domain.set_manifold(0, flat_manifold_domain);

	ofstream output_file("tria_domain_2d.vtk");
	GridOut grid_out;
	grid_out.write_vtk(tria_domain, output_file);
}
