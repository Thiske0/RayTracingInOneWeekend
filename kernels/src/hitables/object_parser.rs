use crate::{
    hitables::{
        hitable_list_builder::HitableListBuilder,
        planar::{NormedTriangle, Triangle},
    },
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
    let mut vertex_normals = Vec::new();
    let mut result = HitableListBuilder::new();

    for line in file.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "#" => continue,
            "o" => continue,
            "s" => continue,
            "v" => {
                let x: Real = parts[1].parse()?;
                let y: Real = parts[2].parse()?;
                let z: Real = parts[3].parse()?;
                vertices.push(Vec3::new(x, y, z));
            }
            "vn" => {
                let x: Real = parts[1].parse()?;
                let y: Real = parts[2].parse()?;
                let z: Real = parts[3].parse()?;
                vertex_normals.push(Vec3::new(x, y, z));
            }
            "f" => {
                if parts.len() != 4 {
                    return Err("Only triangular faces are supported".into());
                }
                let splitted_parts: Vec<Vec<&str>> = vec![
                    parts[1].split('/').collect::<Vec<_>>(),
                    parts[2].split('/').collect::<Vec<_>>(),
                    parts[3].split('/').collect::<Vec<_>>(),
                ];
                let v1: usize = splitted_parts[0][0].parse()?;
                let v1 = vertices[v1 - 1];
                let v2: usize = splitted_parts[1][0].parse()?;
                let v2 = vertices[v2 - 1];
                let v3: usize = splitted_parts[2][0].parse()?;
                let v3 = vertices[v3 - 1];

                let triangle = if splitted_parts[0].len() < 3 || splitted_parts[0][2].is_empty() {
                    Triangle::new(v1, v2, v3, material.clone())
                } else {
                    let vn1_index: usize = splitted_parts[0][2].parse()?;
                    let vn1 = vertex_normals[vn1_index - 1];
                    let vn2_index: usize = splitted_parts[1][2].parse()?;
                    let vn2 = vertex_normals[vn2_index - 1];
                    let vn3_index: usize = splitted_parts[2][2].parse()?;
                    let vn3 = vertex_normals[vn3_index - 1];
                    NormedTriangle::new(v1, v2, v3, vn1, vn2, vn3, material.clone())
                };
                result.add(triangle);
            }
            _ => {
                return Err(format!("Unsupported line type: {}", parts[0]).into());
            }
        }
    }
    Ok(result)
}
