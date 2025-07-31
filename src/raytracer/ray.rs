use crate::raytracer::{
    color::Color,
    hitable::Hitable,
    vec3::{Point3, Vec3},
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

    pub fn at(&self, t: f32) -> Point3 {
        self.origin + self.direction * t
    }

    pub fn color<T: Hitable>(&self, hitable: &T) -> Color {
        if let Some(hit) = hitable.hit(self, &(0.0..f32::INFINITY)) {
            return ((hit.normal + 1.0) * 0.5).to_color();
        }

        let unit_direction = self.direction.normalize();
        let blue = Color::new(0.5, 0.7, 1.0);
        let white = Color::new(1.0, 1.0, 1.0);
        let t = 0.5 * (unit_direction.y + 1.0);
        white.lerp(blue, t)
    }
}
