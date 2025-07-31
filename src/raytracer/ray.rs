use crate::raytracer::{
    color::Color,
    hitable::Hitable,
    vec3::{Point3, Real, Vec3},
};

pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        Ray { origin, direction }
    }

    pub fn origin(&self) -> Point3 {
        self.origin
    }

    pub fn direction(&self) -> Vec3 {
        self.direction
    }

    pub fn at(&self, t: Real) -> Point3 {
        self.origin + self.direction * t
    }

    pub fn color<T: Hitable>(&self, depth: usize, hitable: &T) -> Color {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if depth <= 0 {
            return Color::black();
        }

        if let Some(hit) = hitable.hit(self, &(1e-12..Real::INFINITY)) {
            let direction = hit.normal + Vec3::random_unit();
            let new_ray = Ray::new(hit.p, direction);
            return new_ray.color(depth - 1, hitable) * 0.5;
        }

        let unit_direction = self.direction.normalize();
        let blue = Color::new(0.5, 0.7, 1.0);
        let white = Color::new(1.0, 1.0, 1.0);
        let t = 0.5 * (unit_direction.y + 1.0);
        white.lerp(blue, t)
    }
}
