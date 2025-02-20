#include "Pathtracer.h"
#include <memory>
#include <iostream>
#include <map>
#include <algorithm>
#include "material.h"
#include "embree.h"
#include "sampling.h"
#include "labhelper.h"

using namespace std;
using namespace glm;
using namespace labhelper;

namespace pathtracer
{
///////////////////////////////////////////////////////////////////////////////
// Global variables
///////////////////////////////////////////////////////////////////////////////
Settings settings;
Environment environment;
Image rendered_image;
PointLight point_light;
std::vector<DiscLight> disc_lights;

///////////////////////////////////////////////////////////////////////////
// Restart rendering of image
///////////////////////////////////////////////////////////////////////////
void restart()
{
	// No need to clear image,
	rendered_image.number_of_samples = 0;
}

int getSampleCount()
{
	return std::max(rendered_image.number_of_samples - 1, 0);
}

///////////////////////////////////////////////////////////////////////////
// On window resize, window size is passed in, actual size of pathtraced
// image may be smaller (if we're subsampling for speed)
///////////////////////////////////////////////////////////////////////////
void resize(int w, int h)
{
	rendered_image.width = w / settings.subsampling;
	rendered_image.height = h / settings.subsampling;
	rendered_image.data.resize(rendered_image.width * rendered_image.height);
	restart();
}

///////////////////////////////////////////////////////////////////////////
/// Return the radiance from a certain direction wi from the environment
/// map.
///////////////////////////////////////////////////////////////////////////
vec3 Lenvironment(const vec3& wi)
{
	const float theta = acos(std::max(-1.0f, std::min(1.0f, wi.y)));
	float phi = atan(wi.z, wi.x);
	if(phi < 0.0f)
		phi = phi + 2.0f * M_PI;
	vec2 lookup = vec2(phi / (2.0 * M_PI), 1 - theta / M_PI);
	return environment.multiplier * environment.map.sample(lookup.x, lookup.y);
}

///////////////////////////////////////////////////////////////////////////
/// Calculate the radiance going from one point (r.hitPosition()) in one
/// direction (-r.d), through path tracing.
///////////////////////////////////////////////////////////////////////////
vec3 Li(Ray& primary_ray)
{
	vec3 L = vec3(0.0f);
	vec3 path_throughput = vec3(1.0);
	Ray current_ray = primary_ray;
	Ray shadowRay;
	for (size_t i = 0; i < settings.max_bounces; i++)
	{


		///////////////////////////////////////////////////////////////////
		//The intersection information from the ray
		///////////////////////////////////////////////////////////////////

		Intersection hit = getIntersection(current_ray);

		///////////////////////////////////////////////////////////////////
		// A Material tree for evaluating brdfs and calculating
		// sample directions.
		///////////////////////////////////////////////////////////////////

		
		Diffuse diffuse(hit.material->m_color);
		MicrofacetBRDF microfacet(hit.material->m_shininess);
		DielectricBSDF dielectric(&microfacet, &diffuse, hit.material->m_fresnel);
		MetalBSDF metal(&microfacet, hit.material->m_color, hit.material->m_fresnel);
		BSDFLinearBlend metal_blend(hit.material->m_metalness, &metal, &dielectric);
		BSDF& mat = metal_blend;
		///////////////////////////////////////////////////////////////////
		// Calculate Direct Illumination from light.
		///////////////////////////////////////////////////////////////////
		{
			const float distance_to_light = length(point_light.position - hit.position);
			const float falloff_factor = 1.0f / (distance_to_light * distance_to_light);
			vec3 Li = point_light.intensity_multiplier * point_light.color * falloff_factor;
			vec3 wi = normalize(point_light.position - hit.position);



			///////////////////////////////////////////////////////////////////
			// Check if the point is in shadow
			///////////////////////////////////////////////////////////////////

			shadowRay = Ray(hit.position + hit.geometry_normal * EPSILON, normalize(point_light.position - hit.position));

			if (!occluded(shadowRay))
			{
				L += path_throughput * mat.f(wi, hit.wo, hit.shading_normal) * Li * std::max(0.0f, dot(wi, hit.shading_normal));
			}
			
		}
		for (const DiscLight& discLight : disc_lights) {
			// Sample a point on the disc light
			// Create random point in local disc coordinates (square root ensures unifor distribution over the area of the disc)
			float r = sqrt(randf());
			float theta = 2.0f * M_PI * randf();
			// Create a point on the disc by converting coordinates (r, theta) to coordinate (x,y) on the disc
			vec3 localPoint = vec3(r * cos(theta), r * sin(theta), 0.0f);

			// Transform to world space
			// Convert local disc coordinates to world coordinates, aligning the disc with its direction
			mat3 tbn = tangentSpace(discLight.direction);
			// Transforms the local point to world space and scales it by the disc's radius, then translates it to the disc's position
			vec3 light_sample = discLight.position + (tbn * localPoint) * discLight.radius;

			vec3 wi = normalize(light_sample - hit.position);
			float distance_to_light = length(light_sample - hit.position);
			float falloff_factor = 1.0f / (distance_to_light * distance_to_light);
			vec3 Li = discLight.intensity_multiplier * discLight.color * falloff_factor;

			Ray shadowRay;
			shadowRay.o = hit.position + hit.geometry_normal * EPSILON;
			shadowRay.d = wi;

			if (!occluded(shadowRay))
			{
				float cos_theta = std::max(0.0f, dot(wi, hit.shading_normal));
				float cos_theta_light = std::max(0.0f, dot(-wi, discLight.direction));
				float solid_angle = (cos_theta_light * discLight.radius * discLight.radius) / (distance_to_light * distance_to_light);
				L += path_throughput * mat.f(wi, hit.wo, hit.shading_normal) * Li * cos_theta * solid_angle;
			}
		}

		L += path_throughput * hit.material->m_emission;

		///////////////////////////////////////////////////////////////////


		float pdf;
		vec3 wi;
		vec3 f;
		// Sample an incoming direction (and the brdf and pdf value for that direction)
		auto brdf = mat.sample_wi(hit.wo, hit.shading_normal);
		

		pdf = brdf.pdf;
		wi = brdf.wi;
		f = brdf.f;

		// if the pdf is zero, it means that the current path is extremely unlikely
		if (pdf < EPSILON) return L;

		auto cosine_term = abs(dot(wi, hit.shading_normal));
		path_throughput = path_throughput * (f * cosine_term) / pdf;
		
		//If path_throughput is zero, we can stop the path tracing
		if (path_throughput == vec3(0.0f))
			return L;

		Ray next_ray;
		//Create next ray on the path (existing instance can not be reused)
		next_ray = Ray(hit.position + hit.shading_normal*EPSILON, wi);
		
		if (!intersect(next_ray))
		{
			return L + path_throughput * Lenvironment(next_ray.d);

		}
		current_ray = next_ray;
	}
	return L;
}

///////////////////////////////////////////////////////////////////////////
/// Used to homogenize points transformed with projection matrices
///////////////////////////////////////////////////////////////////////////
inline static glm::vec3 homogenize(const glm::vec4& p)
{
	return glm::vec3(p * (1.f / p.w));
}

///////////////////////////////////////////////////////////////////////////
/// Trace one path per pixel and accumulate the result in an image
///////////////////////////////////////////////////////////////////////////
void tracePaths(const glm::mat4& V, const glm::mat4& P)
{
	// Stop here if we have as many samples as we want
	if((int(rendered_image.number_of_samples) > settings.max_paths_per_pixel)
	   && (settings.max_paths_per_pixel != 0))
	{
		return;
	}
	vec3 camera_pos = vec3(glm::inverse(V) * vec4(0.0f, 0.0f, 0.0f, 1.0f));
	// Trace one path per pixel (the omp parallel stuf magically distributes the
	// pathtracing on all cores of your CPU).
	int num_rays = 0;
	vector<vec4> local_image(rendered_image.width * rendered_image.height, vec4(0.0f));

#pragma omp parallel for
	for(int y = 0; y < rendered_image.height; y++)
	{
		for(int x = 0; x < rendered_image.width; x++)
		{
			vec3 color;
			Ray primaryRay;
			primaryRay.o = camera_pos;
			primaryRay.d = vec3(randf(), randf(), randf()); //Task One - Randomize the direction of the primary ray

			
			// Create a ray that starts in the camera position and points toward
			// the current pixel on a virtual screen.
			vec2 screenCoord = vec2((float(x) / float(rendered_image.width)),
				(float(y) / float(rendered_image.height)));
			// Calculate direction
			vec4 viewCoord = vec4(screenCoord.x * 2.0f - 1.0f, screenCoord.y * 2.0f - 1.0f, 1.0f, 1.0f);
			vec3 p = homogenize(inverse(P * V) * viewCoord);
			primaryRay.d = normalize(p - camera_pos);

			// Intersect ray with scene
			if(intersect(primaryRay))
			{
				// If it hit something, evaluate the radiance from that point
				color = Li(primaryRay);
			}
			else
			{
				// Otherwise evaluate environment
				color = Lenvironment(primaryRay.d);
			}
			// Accumulate the obtained radiance to the pixels color
			float n = float(rendered_image.number_of_samples);
			rendered_image.data[y * rendered_image.width + x] =
			    rendered_image.data[y * rendered_image.width + x] * (n / (n + 1.0f))
			    + (1.0f / (n + 1.0f)) * color;
		}
	}
	rendered_image.number_of_samples += 1;
}
}; // namespace pathtracer
