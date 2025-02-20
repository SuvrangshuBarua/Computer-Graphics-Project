#include "material.h"
#include "sampling.h"
#include "labhelper.h"

using namespace labhelper;

namespace pathtracer
{
WiSample sampleHemisphereCosine(const vec3& wo, const vec3& n)
{
	mat3 tbn = tangentSpace(n);
	vec3 sample = cosineSampleHemisphere();
	WiSample r;
	r.wi = tbn * sample;
	if(dot(r.wi, n) > 0.0f)
		r.pdf = max(0.0f, dot(r.wi, n)) / M_PI;
	return r;
}

///////////////////////////////////////////////////////////////////////////
// A Lambertian (diffuse) material
///////////////////////////////////////////////////////////////////////////
vec3 Diffuse::f(const vec3& wi, const vec3& wo, const vec3& n) const
{
	if(dot(wi, n) <= 0.0f)
		return vec3(0.0f);
	if(!sameHemisphere(wi, wo, n))
		return vec3(0.0f);
	return (1.0f / M_PI) * color;
}

WiSample Diffuse::sample_wi(const vec3& wo, const vec3& n) const
{
	WiSample r = sampleHemisphereCosine(wo, n);
	r.f = f(r.wi, wo, n);
	return r;
}

vec3 MicrofacetBRDF::f(const vec3& wi, const vec3& wo, const vec3& n) const
{
	
	vec3 wh = normalize(wi + wo);
	float dot_n_wh = max(0.0001f, dot(n, wh));
	float dot_n_wo = max(0.0001f, dot(n, wo));
	float dot_n_wi = max(0.0001f, dot(n, wi));
	float dot_wo_wh = max(0.0001f, dot(wo, wh));
	float s = pow(dot_n_wh, shininess);

	float D = (shininess + 2.0f) / (2.0f * M_PI) * s;

	float G = min(1.0, min(2.0 * ((dot_n_wh * dot_n_wo) / dot_wo_wh), 2.0 * (dot_n_wh * dot_n_wi) / dot_wo_wh));
	float denominator = (4.0 * clamp(dot_n_wo * dot_n_wi, 0.0001f, 1.0f));
	float brdf = ( D * G) / denominator;

	return vec3(brdf);
}

WiSample MicrofacetBRDF::sample_wi(const vec3& wo, const vec3& n) const
{
	WiSample r;
	vec3 tangent = normalize(perpendicular(n));
	vec3 bitangent = normalize(cross(tangent, n));
	float phi = 2.0f * M_PI * randf();
	float cos_theta = pow(randf(), 1.0f / (shininess + 1.0f));
	float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
	vec3 wh = normalize(sin_theta * cos(phi) * tangent + 
						sin_theta * sin(phi) * bitangent + 
						cos_theta * n);
	
	
	float p_wh = ((shininess + 1.0f) * max(0.0f, pow(dot(n, wh), shininess)) / (2.0f * M_PI));
	float p_wi = p_wh / max(0.001f, (4.0f * dot(wo, wh)));

	r.pdf = p_wi;
	r.wi = -reflect(wo, wh); // reflect the outgoing direction around the half vector
	r.f = f(r.wi, wo, n);
	return r;
}


float BSDF::fresnel(const vec3& wi, const vec3& wo) const
{

	vec3 wh = normalize(wi + wo);
	float dot_wh_wi = max(0.0001f, dot(wo, wh));
	float f = R0 + (1.0f - R0) * pow(1.0f - dot_wh_wi, 5.0f);
	return f;
	
}


vec3 DielectricBSDF::f(const vec3& wi, const vec3& wo, const vec3& n) const
{

	vec3 diffuse = reflective_material->f(wi, wo, n);
	vec3 specular = transmissive_material->f(wi, wo, n);

	return fresnel(wi, wo) * diffuse + (1.0f - fresnel(wi, wo)) * specular;

}

WiSample DielectricBSDF::sample_wi(const vec3& wo, const vec3& n) const
{

	WiSample r;

	if (randf() < 0.5f) {

		r = reflective_material->sample_wi(wo, n);
		r.f = f(r.wi, wo, n);
		r.pdf *= 0.5f;
		float F = fresnel(r.wi, wo);
		r.f *= F;
	}
	else {

		r = transmissive_material->sample_wi(wo, n);
		r.f = f(r.wi, wo, n);
		r.pdf *= 0.5f;
		float F = fresnel(r.wi, wo);
		r.f *= 1.0f - F;
	}
	return r;
}

vec3 MetalBSDF::f(const vec3& wi, const vec3& wo, const vec3& n) const
{

	float F = fresnel(wi, wo);
	vec3 specular_term = reflective_material->f(wi, wo, n);
	return F * specular_term * color;
}

WiSample MetalBSDF::sample_wi(const vec3& wo, const vec3& n) const
{
	WiSample r = reflective_material->sample_wi(wo, n); // Importance sampling from BRDF
	r.f = r.f * fresnel(r.wi, wo) * color; // Apply Fresnel and color factor
	return r;
}


vec3 BSDFLinearBlend::f(const vec3& wi, const vec3& wo, const vec3& n) const
{
	return w * bsdf0->f(wi, wo, n) + (1-w) * bsdf1->f(wi,wo, n);
}

WiSample BSDFLinearBlend::sample_wi(const vec3& wo, const vec3& n) const
{
	WiSample r;

	if (randf() < w) {
		r = bsdf0->sample_wi(wo, n);
	}
	else {
		r = bsdf1->sample_wi(wo, n);
	}
	return r;
}


#if SOLUTION_PROJECT == PROJECT_REFRACTIONS
///////////////////////////////////////////////////////////////////////////
// A perfect specular refraction.
///////////////////////////////////////////////////////////////////////////
vec3 GlassBTDF::f(const vec3& wi, const vec3& wo, const vec3& n) const
{
	if(sameHemisphere(wi, wo, n))
	{
		return vec3(0);
	}
	else
	{
		return vec3(1);
	}
}

WiSample GlassBTDF::sample_wi(const vec3& wo, const vec3& n) const
{
	WiSample r;

	float eta;
	glm::vec3 N;
	if(dot(wo, n) > 0.0f)
	{
		N = n;
		eta = 1.0f / ior;
	}
	else
	{
		N = -n;
		eta = ior;
	}

	// Alternatively:
	// d = dot(wo, N)
	// k = d * d (1 - eta*eta)
	// wi = normalize(-eta * wo + (d * eta - sqrt(k)) * N)

	// or

	// d = dot(n, wo)
	// k = 1 - eta*eta * (1 - d * d)
	// wi = - eta * wo + ( eta * d - sqrt(k) ) * N

	float w = dot(wo, N) * eta;
	float k = 1.0f + (w - eta) * (w + eta);
	if(k < 0.0f)
	{
		// Total internal reflection
		r.wi = reflect(-wo, n);
	}
	else
	{
		k = sqrt(k);
		r.wi = normalize(-eta * wo + (w - k) * N);
	}
	r.pdf = abs(dot(r.wi, n));
	r.f = vec3(1.0f, 1.0f, 1.0f);

	return r;
}

vec3 BTDFLinearBlend::f(const vec3& wi, const vec3& wo, const vec3& n) const
{
	return w * btdf0->f(wi, wo, n) + (1.0f - w) * btdf1->f(wi, wo, n);
}

WiSample BTDFLinearBlend::sample_wi(const vec3& wo, const vec3& n) const
{
	if(randf() < w)
	{
		WiSample r = btdf0->sample_wi(wo, n);
		return r;
	}
	else
	{
		WiSample r = btdf1->sample_wi(wo, n);
		return r;
	}
}

#endif
} // namespace pathtracer
