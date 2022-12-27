// Vertex shader

struct Camera {
	view_pos: vec4<f32>,
	view_proj: mat4x4<f32>,
};

@group(1) @binding(0)
var <uniform> camera: Camera;

struct Light {
	position: vec3<f32>,
	color: vec3<f32>,
}

@group(2) @binding(0)
var<uniform>  light: Light;

struct VetexInput {
	@location(0) position: vec3<f32>,
	@location(1) tex_coords: vec2<f32>,
	@location(2) normal: vec3<f32>,
	@location(3) tangent: vec3<f32>,
	@location(4) bitangent: vec3<f32>
};

struct InstanceInput {
	@location(5) model_matrix_0: vec4<f32>,
	@location(6) model_matrix_1: vec4<f32>,
	@location(7) model_matrix_2: vec4<f32>,
	@location(8) model_matrix_3: vec4<f32>,
	@location(9) normal_matrix_0: vec3<f32>,
	@location(10) normal_matrix_1: vec3<f32>,
	@location(11) normal_matrix_2: vec3<f32>,
}

struct VertexOutput {
	@builtin(position) clip_position : vec4<f32>,
	@location(0) tex_coords: vec2<f32>,
	@location(1) tangent_position: vec3<f32>,
	@location(2) tangent_light_position: vec3<f32>,
	@location(3) tangent_view_position: vec3<f32>,
};


@vertex
fn vs_main(
	model: VetexInput,
	instance: InstanceInput,
) -> VertexOutput {
	let model_matrix = mat4x4<f32>(
		instance.model_matrix_0,
		instance.model_matrix_1,
		instance.model_matrix_2,
		instance.model_matrix_3,
	);

	let nomal_matrx = mat3x3<f32>(
		instance.normal_matrix_0,
		instance.normal_matrix_1,
		instance.normal_matrix_2,
	);

	let world_normal = normalize(nomal_matrx * model.normal);
	let wold_tangent = normalize(nomal_matrx * model.tangent);
	let world_bitangent = normalize(nomal_matrx * model.bitangent);
	let tantent_matrix = transpose(mat3x3<f32>(
		wold_tangent,
		world_bitangent,
		world_normal,
	));

	let world_position = model_matrix * vec4<f32>(model.position, 1.0);

	var out: VertexOutput;
	out.clip_position = camera.view_proj * world_position;
	out.tex_coords = model.tex_coords;
	out.tangent_position = tantent_matrix * world_position.xyz;
	out.tangent_view_position = tantent_matrix * camera.view_pos.xyz;
	out.tangent_light_position = tantent_matrix * light.position;

	return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(0) @binding(2)
var t_normal: texture_2d<f32>;
@group(0) @binding(3)
var s_normal: sampler;


// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);
	let object_normla: vec4<f32> = textureSample(t_normal, s_normal, in.tex_coords);

	let ambient_strenght = 0.1;
	let ambient_color  = light.color * ambient_strenght;

	let tangent_normal = object_normla.xyz * 2.0 - 1.0;
	let light_dir = normalize(in.tangent_light_position - in.tangent_position);
	let view_dir = normalize(in.tangent_light_position - in.tangent_position);
	let half_dir = normalize(view_dir + light_dir);

	let diffuse_strenght = max(dot(tangent_normal, light_dir), 0.0);
	let diffuse_color = light.color * diffuse_strenght;

	let specular_strenght = pow(max(dot(tangent_normal, half_dir), 0.0), 32.0);
	let speculer_color = specular_strenght * light.color;

	let result = (ambient_color + diffuse_color + speculer_color) * object_color.xyz;

	return vec4<f32>(result, object_color.a);
}