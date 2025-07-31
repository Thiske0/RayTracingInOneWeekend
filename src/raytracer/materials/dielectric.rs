use crate::raytracer::{
    color::Color,
    hitable::HitRecord,
    materials::Material,
    ray::Ray,
    vec3::{Real, Vec3},
};

pub struct Dielectric {
    refraction_index: Real,
}

impl Dielectric {
    pub fn new(refraction_index: Real) -> Self {
        Dielectric { refraction_index }
    }

    // Abuse vec3::random to generate a random number in the range [0, 1)
    fn random_real() -> Real {
        Vec3::random(0.0..1.0).x
    }

    // Use Schlick's approximation for reflectance.
    fn reflectance(cosine: Real, refraction_index: Real) -> Real {
        let r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
        let r0 = r0 * r0;
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}
impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<(Ray, Color)> {
        let is_front_face = hit.normal.dot(ray.direction) < 0.0;
        let (ri, normal) = if is_front_face {
            (1.0 / self.refraction_index, hit.normal)
        } else {
            (self.refraction_index, -hit.normal)
        };

        let unit_direction = ray.direction.normalize();

        let cos_theta = Real::min(-unit_direction.dot(normal), 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let cannot_refract = ri * sin_theta > 1.0;
        let direction = if cannot_refract {
            // Reflect
            unit_direction.reflect(normal)
        } else {
            // Refract
            unit_direction.refract(normal, ri)
        };

        let scattered = Ray::new(hit.p, direction);
        Some((scattered, Color::white()))
    }
}
