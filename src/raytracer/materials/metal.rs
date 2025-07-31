use crate::raytracer::{
    color::Color,
    hitable::HitRecord,
    materials::Material,
    ray::Ray,
    vec3::{Real, Vec3},
};

pub struct Metal {
    albedo: Color,
    fuzziness: Real,
}

impl Metal {
    pub fn new(albedo: Color, fuzziness: Real) -> Self {
        Metal { albedo, fuzziness }
    }
}
impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<(Ray, Color)> {
        let direction =
            ray.direction.reflect(hit.normal).normalize() + Vec3::random_unit() * self.fuzziness;
        if direction.near_zero() || direction.dot(hit.normal) < 0.0 {
            return None; // Ray is absorbed
        }
        let new_ray = Ray::new(hit.p, direction);
        Some((new_ray, self.albedo))
    }
}
