use crate::raytracer::{
    color::Color,
    sphere::Sphere,
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

    pub fn color(&self) -> Color {
        let sphere = Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5);
        if let Some(t) = sphere.hit(self) {
            let normal = (self.at(t) - sphere.center).normalize();
            return ((normal + 1.0) * 0.5).to_color();
        }

        let unit_direction = self.direction.normalize();
        let blue = Color::new(0.5, 0.7, 1.0);
        let white = Color::new(1.0, 1.0, 1.0);
        let t = 0.5 * (unit_direction.y + 1.0);
        white.lerp(blue, t)
    }
}
