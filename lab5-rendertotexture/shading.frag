#version 420

// required by GLSL spec Sect 4.5.3 (though nvidia does not, amd does)
precision highp float;

///////////////////////////////////////////////////////////////////////////////
// Material
///////////////////////////////////////////////////////////////////////////////
uniform vec3 material_color;
uniform float material_metalness;
uniform float material_fresnel;
uniform float material_shininess;
uniform vec3 material_emission;

const vec3 black_color = vec3(0,0,0);

uniform int has_color_texture;
layout(binding = 0) uniform sampler2D colorMap;
uniform int has_emission_texture;
layout(binding = 5) uniform sampler2D emissiveMap;

///////////////////////////////////////////////////////////////////////////////
// Environment
///////////////////////////////////////////////////////////////////////////////
layout(binding = 6) uniform sampler2D environmentMap;
layout(binding = 7) uniform sampler2D irradianceMap;
layout(binding = 8) uniform sampler2D reflectionMap;
uniform float environment_multiplier;

///////////////////////////////////////////////////////////////////////////////
// Light source
///////////////////////////////////////////////////////////////////////////////
uniform vec3 point_light_color = vec3(1.0, 1.0, 1.0);
uniform float point_light_intensity_multiplier = 50.0;

///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////
#define PI 3.14159265359

///////////////////////////////////////////////////////////////////////////////
// Input varyings from vertex shader
///////////////////////////////////////////////////////////////////////////////
in vec2 texCoord;
in vec3 viewSpaceNormal;
in vec3 viewSpacePosition;

///////////////////////////////////////////////////////////////////////////////
// Input uniform variables
///////////////////////////////////////////////////////////////////////////////
uniform mat4 viewInverse;
uniform vec3 viewSpaceLightPosition;

///////////////////////////////////////////////////////////////////////////////
// Output color
///////////////////////////////////////////////////////////////////////////////
layout(location = 0) out vec4 fragmentColor;


vec3 calculateDirectIllumiunation(vec3 wo, vec3 n, vec3 base_color)
{
	vec3 direct_illum = base_color;
	
	///////////////////////////////////////////////////////////////////////////
	// Task 1.2 - Calculate the radiance Li from the light, and the direction
	//            to the light. If the light is backfacing the triangle,
	//            return vec3(0);
	///////////////////////////////////////////////////////////////////////////

	float d = length(viewSpaceLightPosition - viewSpacePosition);
	vec3 li = (point_light_intensity_multiplier * point_light_color) / (d * d);

	///////////////////////////////////////////////////////////////////////////
	// Task 1.3 - Calculate the diffuse term and return that as the result
	///////////////////////////////////////////////////////////////////////////
	vec3 wi = normalize(viewSpaceLightPosition - viewSpacePosition);
	if(dot(n, wi) <= 0) return black_color;

	vec3 diffuse_term = (direct_illum * length(dot(n, wi)) * li) / PI;

	///////////////////////////////////////////////////////////////////////////
	// Task 2 - Calculate the Torrance Sparrow BRDF and return the light
	//          reflected from that instead
	///////////////////////////////////////////////////////////////////////////

	float ro = material_fresnel;
	vec3 wh = normalize(wi + wo);
	float dot_n_wh =  max(0.0001, dot(n, wh));
	float dot_n_wo =  max(0.0001,dot(n, wo));
	float dot_n_wi =  max(0.0001,dot(n, wi));
	float dot_wo_wh = max(0.0001,dot(wo, wh));

	float F = ro + (1.0 - ro) * pow((1.0 - dot_wo_wh), 5.0);

	float s = material_shininess;
	float D = ((s + 2) / (2.0 * PI)) * pow(dot_n_wh, s);

	

	float G = min(1.0, min(2.0 * ((dot_n_wh * dot_n_wo) / dot_wo_wh), 2.0 * (dot_n_wh * dot_n_wi) / dot_wo_wh));
	float denominator = (4.0 * clamp(dot_n_wo * dot_n_wi, 0.0001, 1.0));
	float brdf = (F * D * G)  / denominator;

	//return brdf * dot_n_wi * li;
	//return vec3(F);

	///////////////////////////////////////////////////////////////////////////
	// Task 3 - Make your shader respect the parameters of our material model.
	///////////////////////////////////////////////////////////////////////////

	vec3 dielectric_term = brdf * dot_n_wi * li + (1 - F) * diffuse_term;
	vec3 metal_term = brdf * base_color * dot_n_wi * li;

	direct_illum = material_metalness * metal_term + (1 - material_metalness) * dielectric_term;

	return direct_illum;
}

vec3 calculateIndirectIllumination(vec3 wo, vec3 n, vec3 base_color)
{
	vec3 indirect_illum = vec3(0.f);
	///////////////////////////////////////////////////////////////////////////
	// Task 5 - Lookup the irradiance from the irradiance map and calculate
	//          the diffuse reflection
	///////////////////////////////////////////////////////////////////////////
	// Calculate the spherical coordinates of the direction
	vec3 world_normal = vec3(viewInverse * vec4(viewSpaceNormal, 0.0));
	float theta = acos(max(-1.0f, min(1.0f, world_normal.y)));
	float phi = atan(world_normal.z, world_normal.x);
	if(phi < 0.0f)
	{
		phi = phi + 2.0f * PI;
	}

	// Use these to lookup the color in the environment map
	vec2 lookup = vec2(phi / (2.0 * PI), 1 - theta / PI);
	vec3 irradiance = texture(irradianceMap, lookup).rgb;

	indirect_illum = base_color * (1.0 / PI) * irradiance;
	///////////////////////////////////////////////////////////////////////////
	// Task 6 - Look up in the reflection map from the perfect specular
	//          direction and calculate the dielectric and metal terms.
	///////////////////////////////////////////////////////////////////////////
	vec3 wi = reflect(wo, n);
	vec3 world_wi = vec3(viewInverse * vec4(wi, 0.0));

	theta = acos(max(-1.0f, min(1.0f, world_wi.y)));
	phi = atan(world_wi.z, world_wi.x);
	if(phi < 0.0f)
	{
		phi = phi + 2.0f * PI;
	}

	// Use these to lookup the color in the environment map
	lookup = vec2(phi / (2.0 * PI), 1 - theta / PI);
	float s = material_shininess;
	float roughness = sqrt(sqrt(2.0 / (s + 2.0)));

	vec3 li = environment_multiplier * textureLod(reflectionMap, lookup, roughness * 7.0).rgb;
	vec3 diffuse_term = (base_color * length(dot(n, wi)) * li) / PI;

	float ro = material_fresnel;
	vec3 wh = normalize(wi + wo);
	float dot_n_wh =  max(0.0001, dot(n, wh));
	float dot_n_wo =  max(0.0001,dot(n, wo));
	float dot_n_wi =  max(0.0001,dot(n, wi));
	float dot_wo_wh = max(0.0001,dot(wo, wh));

	float F = ro + (1.0 - ro) * pow((1.0 - dot_wo_wh), 5.0);

	vec3 dielectric_term = F * li + (1 - F) * diffuse_term;

	vec3 metal_term = F * base_color * li;

	indirect_illum = material_metalness * metal_term + (1 - material_metalness) * dielectric_term;
	return indirect_illum;
}


void main()
{
	vec3 wo = -normalize(viewSpacePosition);
	vec3 n = normalize(viewSpaceNormal);

	vec3 base_color = material_color;
	if(has_color_texture == 1)
	{
		base_color *= texture(colorMap, texCoord).rgb;
	}

	// Direct illumination
	vec3 direct_illumination_term = calculateDirectIllumiunation(wo, n, base_color);

	// Indirect illumination
	vec3 indirect_illumination_term = calculateIndirectIllumination(wo, n, base_color);

	///////////////////////////////////////////////////////////////////////////
	// Add emissive term. If emissive texture exists, sample this term.
	///////////////////////////////////////////////////////////////////////////
	vec3 emission_term = material_emission;
	if(has_emission_texture == 1)
	{
		emission_term *= texture(emissiveMap, texCoord).rgb;
	}

	fragmentColor.rgb = direct_illumination_term + indirect_illumination_term + emission_term;
}
