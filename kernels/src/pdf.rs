use crate::{
    hitables::HitKind,
    onb::ONB,
    random::{Random, RandomRange},
    vec3::{Point3, Real, Vec3},
};
use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait PDF {
    fn generate(&self, rng: &mut Random) -> Vec3;
    fn value(&self, direction: &Vec3, rng: &mut Random) -> Real;
}

#[enum_dispatch(PDF)]
pub enum PDFKind<'b, 'a> {
    Sphere(SpherePDF),
    Cosine(CosinePDF),
    Hitable(HitablePDF<'b, 'a>),
}

pub struct SpherePDF {}

impl SpherePDF {
    pub fn new() -> PDFKind<'static, 'static> {
        SpherePDF {}.into()
    }
}

impl PDF for SpherePDF {
    fn generate(&self, rng: &mut Random) -> Vec3 {
        Vec3::random_unit(rng)
    }

    fn value(&self, _direction: &Vec3, _rng: &mut Random) -> Real {
        1.0 / (4.0 * (core::f32::consts::PI as Real))
    }
}

pub struct CosinePDF {
    uvw: ONB,
}

impl CosinePDF {
    pub fn new(normal: &Vec3) -> PDFKind<'static, 'static> {
        let uvw = ONB::new_from_normal(normal);
        CosinePDF { uvw }.into()
    }
}

impl PDF for CosinePDF {
    fn generate(&self, rng: &mut Random) -> Vec3 {
        let dir = Vec3::random_cosine_direction(rng);
        self.uvw.to_world(&dir)
    }

    fn value(&self, direction: &Vec3, _rng: &mut Random) -> Real {
        let cosine = direction.normalize().dot(&self.uvw.w());
        if cosine <= 0.0 {
            0.0
        } else {
            cosine / (core::f32::consts::PI as Real)
        }
    }
}

pub struct HitablePDF<'b, 'a> {
    origin: Point3,
    hitable: &'b HitKind<'a>,
}

impl<'b, 'a> HitablePDF<'b, 'a> {
    pub fn new(origin: Point3, hitable: &'b HitKind<'a>) -> PDFKind<'b, 'a> {
        HitablePDF { origin, hitable }.into()
    }
}

impl PDF for HitablePDF<'_, '_> {
    fn generate(&self, rng: &mut Random) -> Vec3 {
        self.hitable.random(&self.origin, rng)
    }

    fn value(&self, direction: &Vec3, rng: &mut Random) -> Real {
        self.hitable.pdf_value(&self.origin, direction, rng)
    }
}

pub struct MixturePDF<'a, PDF1: PDF, PDF2: PDF> {
    chance_pdf1: Real,
    pdf1: &'a PDF1,
    pdf2: &'a PDF2,
}

impl<'a, PDF1: PDF, PDF2: PDF> MixturePDF<'a, PDF1, PDF2> {
    pub fn new_with_chance(chance_pdf1: Real, pdf1: &'a PDF1, pdf2: &'a PDF2) -> Self {
        MixturePDF {
            chance_pdf1,
            pdf1,
            pdf2,
        }
    }

    pub fn new(pdf1: &'a PDF1, pdf2: &'a PDF2) -> Self {
        MixturePDF::new_with_chance(0.5, pdf1, pdf2)
    }
}

impl<PDF1: PDF, PDF2: PDF> PDF for MixturePDF<'_, PDF1, PDF2> {
    fn generate(&self, rng: &mut Random) -> Vec3 {
        if rng.random_range(0.0..1.0) < self.chance_pdf1 {
            self.pdf1.generate(rng)
        } else {
            self.pdf2.generate(rng)
        }
    }

    fn value(&self, direction: &Vec3, rng: &mut Random) -> Real {
        self.chance_pdf1 * self.pdf1.value(direction, rng)
            + (1.0 - self.chance_pdf1) * self.pdf2.value(direction, rng)
    }
}
