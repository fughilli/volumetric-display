use bevy::prelude::*;

#[derive(Component)]
pub struct FireParticle {
    pub velocity: bevy::math::Vec3,
    pub lifetime: f32,
    pub max_lifetime: f32,
    pub size: f32,
}

#[derive(Component)]
pub struct FireEmitter {
    pub spawn_rate: f32,
    pub spawn_timer: f32,
    pub particle_lifetime: f32,
    pub initial_velocity: bevy::math::Vec3,
    pub velocity_variance: bevy::math::Vec3,
    pub size: f32,
    pub size_variance: f32,
}

impl Default for FireEmitter {
    fn default() -> Self {
        Self {
            spawn_rate: 30.0, // particles per second
            spawn_timer: 0.0,
            particle_lifetime: 2.0,
            initial_velocity: bevy::math::Vec3::new(0.0, 2.0, 0.0),
            velocity_variance: bevy::math::Vec3::new(0.5, 1.0, 0.5),
            size: 0.1,
            size_variance: 0.05,
        }
    }
}

pub fn spawn_fire_particles(
    mut commands: Commands,
    mut emitters: Query<(Entity, &mut FireEmitter, &Transform)>,
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (emitter_entity, mut emitter, transform) in emitters.iter_mut() {
        emitter.spawn_timer += time.delta_seconds_f64() as f32;

        let particles_to_spawn = (emitter.spawn_rate * emitter.spawn_timer) as u32;
        if particles_to_spawn > 0 {
            emitter.spawn_timer = 0.0;

            for _ in 0..particles_to_spawn {
                // Calculate random velocity using simple hash-based random
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                (emitter_entity, particles_to_spawn).hash(&mut hasher);
                let seed = hasher.finish();
                let r1 = (seed & 0xFFFF) as f32 / 65536.0;
                let r2 = ((seed >> 16) & 0xFFFF) as f32 / 65536.0;
                let r3 = ((seed >> 32) & 0xFFFF) as f32 / 65536.0;

                let vel_x = emitter.initial_velocity.x + (r1 - 0.5) * emitter.velocity_variance.x;
                let vel_y = emitter.initial_velocity.y + r2 * emitter.velocity_variance.y;
                let vel_z = emitter.initial_velocity.z + (r3 - 0.5) * emitter.velocity_variance.z;

                // Calculate random size
                let size = emitter.size + (r1 - 0.5) * emitter.size_variance;

                // Spawn position at emitter location with some randomness
                let spawn_offset = bevy::math::Vec3::new((r2 - 0.5) * 0.2, 0.0, (r3 - 0.5) * 0.2);

                // Create a simple quad mesh for the particle using Bevy's Quad primitive
                let mesh = meshes.add(Rectangle::new(size * 2.0, size * 2.0));

                // Create fire-colored material (orange to red gradient)
                let material = StandardMaterial {
                    base_color: Color::srgb(1.0, 0.5, 0.0),
                    emissive: LinearRgba::from(Color::srgb(1.0, 0.3, 0.0)),
                    unlit: true,
                    alpha_mode: bevy::prelude::AlphaMode::Blend,
                    ..default()
                };

                commands.spawn((
                    MaterialMeshBundle {
                        mesh: mesh.clone(),
                        material: materials.add(material),
                        transform: Transform::from_translation(
                            transform.translation + spawn_offset,
                        ),
                        ..default()
                    },
                    FireParticle {
                        velocity: bevy::math::Vec3::new(vel_x, vel_y, vel_z),
                        lifetime: emitter.particle_lifetime,
                        max_lifetime: emitter.particle_lifetime,
                        size,
                    },
                ));
            }
        }
    }
}

pub fn update_fire_particles(
    mut commands: Commands,
    mut particles: Query<(
        Entity,
        &mut FireParticle,
        &mut Transform,
        &mut Handle<StandardMaterial>,
    )>,
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (entity, mut particle, mut transform, material_handle) in particles.iter_mut() {
        particle.lifetime -= time.delta_seconds_f64() as f32;

        if particle.lifetime <= 0.0 {
            commands.entity(entity).despawn();
            continue;
        }

        // Update position
        transform.translation += particle.velocity * time.delta_seconds_f64() as f32;

        // Apply upward acceleration (simulate buoyancy)
        particle.velocity.y += 1.0 * time.delta_seconds_f64() as f32;

        // Apply drag
        particle.velocity *= 0.98;

        // Update color based on lifetime (orange -> red -> dark red)
        let lifetime_ratio = particle.lifetime / particle.max_lifetime;
        if let Some(material) = materials.get_mut(material_handle.id()) {
            if lifetime_ratio > 0.5 {
                // Orange phase
                let t = (lifetime_ratio - 0.5) * 2.0;
                material.base_color = Color::srgb(1.0, 0.3 + t * 0.2, t * 0.2);
                material.emissive = LinearRgba::from(Color::srgb(1.0, 0.3 + t * 0.2, t * 0.2));
            } else {
                // Red phase
                let t = lifetime_ratio * 2.0;
                material.base_color = Color::srgb(1.0, t * 0.3, 0.0);
                material.emissive = LinearRgba::from(Color::srgb(1.0, t * 0.3, 0.0));
            }
            // Update alpha based on lifetime
            let rgba = material.base_color.to_srgba();
            material.base_color = Color::srgba(rgba.red, rgba.green, rgba.blue, lifetime_ratio);
        }

        // Update size (shrink as particle ages)
        let scale = lifetime_ratio * 0.5 + 0.5; // Scale from 1.0 to 0.5
        transform.scale = bevy::math::Vec3::splat(scale * particle.size);
    }
}
