use crate::vec3::Vec3;

pub struct ONB {
    axis: [Vec3; 3],
}

impl ONB {
    pub fn new_from_normal(n: &Vec3) -> Self {
        let w = n.normalize();
        let a = if w.x.abs() > 0.9 {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let v = w.cross(&a).normalize();
        let u = w.cross(&v);
        ONB { axis: [u, v, w] }
    }

    pub fn u(&self) -> &Vec3 {
        &self.axis[0]
    }

    pub fn v(&self) -> &Vec3 {
        &self.axis[1]
    }

    pub fn w(&self) -> &Vec3 {
        &self.axis[2]
    }

    pub fn to_world(&self, a: &Vec3) -> Vec3 {
        &self.axis[0] * a.x + &self.axis[1] * a.y + &self.axis[2] * a.z
    }
}
