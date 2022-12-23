use crate::texture;
use std::ops::Range;

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

pub trait DrawModel<'a> {
	fn draw_mesh(&mut self, mesh: &'a Mesh);
	fn draw_mesh_instanced(
		&mut self,
		mesh: &'a Mesh,
		instances: Range<u32>,
	);
}

impl <'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where 
	'b : 'a,
{
	fn draw_mesh(&mut self, mesh: &'b Mesh)	{
		self.draw_mesh_instanced(mesh, 0..1);
	}

	fn draw_mesh_instanced(
			&mut self,
			mesh: &'a Mesh,
			instances: Range<u32>,
		) {
		self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
		self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
		self.draw_indexed(0..mesh.num_elements, 0, instances);
	}
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

impl ModelVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3];
}

pub struct Material {
	pub name: String,
	pub diffuse_texture: texture::Texture,
	pub bind_group: wgpu::BindGroup,
}

pub struct  Mesh {
	pub name: String,
	pub vertex_buffer: wgpu::Buffer,
	pub index_buffer: wgpu::Buffer,
	pub num_elements: u32,
	pub material: usize,
}

pub struct Model{
	pub meshes: Vec<Mesh>,
	pub materials: Vec<Material>,
}



impl Vertex for ModelVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}
