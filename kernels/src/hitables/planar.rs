use gpu_builder::{DeviceCopyBuilder, derive_builder};

use crate::{
    hitables::{BoundingBox, HitKind, HitRecord, Hitable, IntoBoundingBox},
    materials::{MaterialKind, MaterialKindDevice},
    random::Random,
    ray::Ray,
    vec3::{Point3, Real, Vec3},
};
use core::ops::Range;

#[cfg(not(target_os = "cuda"))]
use cust::memory::DeviceCopy;

#[cfg_attr(not(target_os = "cuda"), derive(Clone, Copy, DeviceCopy))]
#[repr(C)]
#[derive(DeviceCopyBuilder)]
pub struct Plane {
    d: Real,
    normal: Vec3,
    w: Vec3,
}

pub struct PlaneHit {
    pub p: Point3,
    pub normal: Vec3,
    pub t: Real,
    pub is_front_face: bool,
}

impl Plane {
    pub fn from_uv(point: &Vec3, u: &Vec3, v: &Vec3) -> Self {
        let n = u.cross(v);
        let normal = n.normalize();
        let d = normal.dot(point);
        let w = (&n) / n.length_squared();
        Plane { normal, d, w }
    }

    pub fn get_uv_coords(
        &self,
        intersection: &Vec3,
        origin: &Vec3,
        u: &Vec3,
        v: &Vec3,
    ) -> (Real, Real) {
        let planar_hitpt_vector = intersection - origin;
        let u_coord = self.w.dot(&planar_hitpt_vector.cross(v));
        let v_coord = self.w.dot(&u.cross(&planar_hitpt_vector));
        (u_coord, v_coord)
    }

    fn hit<'a>(&self, ray: &Ray, t_range: &Range<Real>) -> Option<PlaneHit> {
        let denom = self.normal.dot(&ray.direction);
        if denom.abs() < 1e-6 {
            return None; // Ray is parallel to the plane
        }

        let t = (self.d - self.normal.dot(&ray.origin)) / denom;
        if !t_range.contains(&t) {
            return None; // Intersection is out of range
        }

        let point = ray.at(t);
        let normal = if denom < 0.0 {
            self.normal.clone()
        } else {
            -&self.normal
        };

        Some(PlaneHit {
            t,
            p: point,
            normal,
            is_front_face: denom < 0.0,
        })
    }
}

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
#[derive_builder('a)]
pub struct Triangle<'a> {
    origin: Point3,
    u: Vec3,
    v: Vec3,
    plane: Plane,
    mat: MaterialKind<'a>,
}

impl<'a> Triangle<'a> {
    pub fn new(p1: Point3, p2: Point3, p3: Point3, mat: MaterialKind<'a>) -> HitKind<'a> {
        let u = p2 - &p1;
        let v = p3 - &p1;
        let plane = Plane::from_uv(&p1, &u, &v);
        Triangle {
            origin: p1,
            u,
            v,
            plane,
            mat,
        }
        .into()
    }
}

impl<'b> Hitable for Triangle<'b> {
    fn hit<'a>(
        &'a self,
        ray: &Ray,
        t_range: &Range<Real>,
        _rng: &mut Random,
    ) -> Option<HitRecord<'a>> {
        if let Some(hit) = self.plane.hit(ray, t_range) {
            let (u, v) = self
                .plane
                .get_uv_coords(&hit.p, &self.origin, &self.u, &self.v);
            if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
                return Some(HitRecord {
                    p: hit.p,
                    normal: hit.normal,
                    t: hit.t,
                    is_front_face: hit.is_front_face,
                    u,
                    v,
                    mat: &self.mat,
                });
            }
        }
        None
    }
}

impl<'a> IntoBoundingBox for Triangle<'a> {
    fn boundingbox(&self) -> BoundingBox {
        // Compute the bounding box of all three vertices.
        let bbox = BoundingBox::new_point(self.origin.clone());
        let bbox = bbox.merge(&BoundingBox::new_point((&self.origin) + &self.u));
        bbox.merge(&BoundingBox::new_point((&self.origin) + &self.v))
    }
}

#[repr(C)]
#[cfg_attr(not(target_os = "cuda"), derive(Clone))]
#[derive_builder('a)]
pub struct Quad<'a> {
    origin: Point3,
    u: Vec3,
    v: Vec3,
    plane: Plane,
    mat: MaterialKind<'a>,
}

impl<'a> Quad<'a> {
    pub fn new(origin: Point3, u: Vec3, v: Vec3, mat: MaterialKind<'a>) -> HitKind<'a> {
        let plane = Plane::from_uv(&origin, &u, &v);
        Quad {
            origin,
            u,
            v,
            plane,
            mat,
        }
        .into()
    }
}

impl<'b> Hitable for Quad<'b> {
    fn hit<'a>(
        &'a self,
        ray: &Ray,
        t_range: &Range<Real>,
        _rng: &mut Random,
    ) -> Option<HitRecord<'a>> {
        if let Some(hit) = self.plane.hit(ray, t_range) {
            let (u, v) = self
                .plane
                .get_uv_coords(&hit.p, &self.origin, &self.u, &self.v);
            if u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0 {
                return Some(HitRecord {
                    p: hit.p,
                    normal: hit.normal,
                    t: hit.t,
                    is_front_face: hit.is_front_face,
                    u,
                    v,
                    mat: &self.mat,
                });
            }
        }
        None
    }
}

impl<'a> IntoBoundingBox for Quad<'a> {
    fn boundingbox(&self) -> BoundingBox {
        // Compute the bounding box of all four vertices.
        let bbox = BoundingBox::new_point(self.origin.clone());
        let bbox = bbox.merge(&BoundingBox::new_point((&self.origin) + &self.u));
        let bbox = bbox.merge(&BoundingBox::new_point((&self.origin) + &self.v));
        bbox.merge(&BoundingBox::new_point((&self.origin) + &self.u + &self.v))
    }
}

#[cfg(not(target_os = "cuda"))]
use crate::hitables::hitable_list_builder::HitableListBuilder;

#[cfg(not(target_os = "cuda"))]
pub fn make_box<'a>(a: Point3, b: Point3, mat: MaterialKind<'a>) -> HitableListBuilder<'a> {
    // Returns the 3D box (six sides) that contains the two opposite vertices a & b.

    let mut sides = HitableListBuilder::new();

    // Construct the two opposite vertices with the minimum and maximum coordinates.
    let min = Point3::new(
        Real::min(a.x, b.x),
        Real::min(a.y, b.y),
        Real::min(a.z, b.z),
    );
    let max = Point3::new(
        Real::max(a.x, b.x),
        Real::max(a.y, b.y),
        Real::max(a.z, b.z),
    );

    let dx = Vec3::new(max.x - min.x, 0.0, 0.0);
    let dy = Vec3::new(0.0, max.y - min.y, 0.0);
    let dz = Vec3::new(0.0, 0.0, max.z - min.z);

    sides.add(Quad::new(
        Point3::new(min.x, min.y, max.z),
        dx,
        dy,
        mat.clone(),
    )); // front
    sides.add(Quad::new(
        Point3::new(max.x, min.y, max.z),
        -dz,
        dy,
        mat.clone(),
    )); // right
    sides.add(Quad::new(
        Point3::new(max.x, min.y, min.z),
        -dx,
        dy,
        mat.clone(),
    )); // back
    sides.add(Quad::new(
        Point3::new(min.x, min.y, min.z),
        dz,
        dy,
        mat.clone(),
    )); // left
    sides.add(Quad::new(
        Point3::new(min.x, max.y, max.z),
        dx,
        -dz,
        mat.clone(),
    )); // top
    sides.add(Quad::new(Point3::new(min.x, min.y, min.z), dx, dz, mat)); // bottom

    sides
}
