use crate::{
    hitables::{hitable_list_builder::HitableListBuilder, planar::Triangle},
    materials::MaterialKind,
    vec3::{Real, Vec3},
};
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

pub fn parse_obj<'a>(
    path: &str,
    material: MaterialKind<'a>,
) -> Result<HitableListBuilder<'a>, Box<dyn std::error::Error>> {
    // Implementation for parsing .obj files
    let file = BufReader::new(File::open(path)?);
    let mut vertices = Vec::new();
    let mut result = HitableListBuilder::new();

    for line in file.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "#" => continue,
            "v" => {
                let x: Real = parts[1].parse()?;
                let y: Real = parts[2].parse()?;
                let z: Real = parts[3].parse()?;
                vertices.push(Vec3::new(x, y, z));
            }
            "f" => {
                let v1: usize = parts[1].split('/').next().unwrap().parse()?;
                let v1 = vertices[v1 - 1];
                let v2: usize = parts[2].split('/').next().unwrap().parse()?;
                let v2 = vertices[v2 - 1];
                let v3: usize = parts[3].split('/').next().unwrap().parse()?;
                let v3 = vertices[v3 - 1];

                let triangle = Triangle::new(v1, v2, v3, material.clone());
                result.add(triangle);
            }
            _ => {
                return Err(format!("Unsupported line type: {}", parts[0]).into());
            }
        }
    }
    Ok(result)
}
