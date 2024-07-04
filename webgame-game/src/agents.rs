use std::time::Duration;

use bevy::{prelude::*, sprite::Mesh2dHandle};
use bevy_rapier2d::control::KinematicCharacterController;

use crate::gridworld::{ShouldRun, GRID_CELL_SIZE};

/// Plugin for agent stuff.
pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                move_agents,
                visualize_agent::<PursuerAgent>(Color::RED),
                visualize_agent::<PlayerAgent>(Color::GREEN),
            )
                .run_if(resource_exists::<ShouldRun>),
        );
    }
}

/// Adds playable functionality for agents.
pub struct AgentPlayPlugin;

impl Plugin for AgentPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            set_player_action.run_if(resource_exists::<ShouldRun>),
        );
    }
}

/// State used by all agents.
#[derive(Component, Clone, Copy)]
pub struct Agent {
    /// The direction the agent is currently looking at.
    pub dir: Vec2,
}

impl Default for Agent {
    fn default() -> Self {
        Self { dir: Vec2::X }
    }
}

// Indicates the Pursuer agent.
#[derive(Component)]
pub struct PursuerAgent;

/// Indicates the Player agent;
#[derive(Component)]
pub struct PlayerAgent;

/// The child of an `Agent` that contains its visuals.
#[derive(Component)]
pub struct AgentVisuals;

/// Adds a visual to newly created agents.
fn visualize_agent<T: Component>(
    color: Color,
) -> impl Fn(
    Commands<'_, '_>,
    Query<'_, '_, Entity, Added<T>>,
    ResMut<Assets<Mesh>>,
    ResMut<Assets<ColorMaterial>>,
) {
    move |mut commands: Commands,
          agent_query: Query<Entity, Added<T>>,
          mut meshes: ResMut<Assets<Mesh>>,
          mut materials: ResMut<Assets<ColorMaterial>>| {
        for e in agent_query.iter() {
            commands.entity(e).insert((
                Mesh2dHandle(meshes.add(Circle::new(GRID_CELL_SIZE * 0.25))),
                materials.add(color),
                Visibility::Visible,
                InheritedVisibility::default(),
                ViewVisibility::default(),
            ));
        }
    }
}

const AGENT_SPEED: f32 = GRID_CELL_SIZE * 2.;

/// Holds the next action for an agent.
#[derive(Default, Component)]
pub struct NextAction {
    /// Which direction the agent will move in.
    pub dir: Vec2,
    /// Whether the agent should toggle nearby objects this frame.
    pub toggle_objs: bool,
}

/// Allows the player to set the Player's next action.
fn set_player_action(
    inpt: Res<ButtonInput<KeyCode>>,
    mut player_query: Query<&mut NextAction, With<PlayerAgent>>,
) {
    let mut dir = Vec2::ZERO;
    if inpt.pressed(KeyCode::KeyW) {
        dir.y += 1.;
    }
    if inpt.pressed(KeyCode::KeyS) {
        dir.y -= 1.;
    }
    if inpt.pressed(KeyCode::KeyA) {
        dir.x -= 1.;
    }
    if inpt.pressed(KeyCode::KeyD) {
        dir.x += 1.;
    }
    let mut next_action = player_query.single_mut();
    next_action.dir = dir;
    next_action.toggle_objs = false;
    if inpt.just_pressed(KeyCode::KeyF) {
        next_action.toggle_objs = true;
    }
}

/// Moves agents around.
pub fn move_agents(
    mut agent_query: Query<(
        Entity,
        &mut Agent,
        &mut KinematicCharacterController,
        &NextAction,
        &Children,
    )>,
    child_query: Query<(Entity, Option<&Name>, Option<&Children>)>,
    mut vis_query: Query<&mut Transform, With<AgentVisuals>>,
    mut anim_query: Query<&mut AnimationPlayer>,
    time: Res<Time>,
    asset_server: Res<AssetServer>,
) {
    for (agent_e, mut agent, mut controller, next_action, children) in agent_query.iter_mut() {
        let dir = next_action.dir;
        let anim_e = get_entity(&agent_e, &["", "", "Root"], &child_query);
        if dir.length_squared() > 0.1 {
            let dir = dir.normalize();
            agent.dir = dir;
            controller.translation = Some(dir * AGENT_SPEED * time.delta_seconds());
            for child in children.iter() {
                if let Ok(mut xform) = vis_query.get_mut(*child) {
                    xform.look_to(-dir.extend(0.), Vec3::Z);
                    if let Ok(mut anim) = anim_query.get_mut(anim_e.unwrap()) {
                        anim.play_with_transition(
                            asset_server.load("characters/cyborgFemaleA.glb#Animation1"),
                            Duration::from_secs_f32(0.2),
                        )
                        .repeat();
                    }
                    break;
                }
            }
        } else if let Some(anim_e) = anim_e {
            if let Ok(mut anim) = anim_query.get_mut(anim_e) {
                anim.play_with_transition(
                    asset_server.load("characters/cyborgFemaleA.glb#Animation0"),
                    Duration::from_secs_f32(0.2),
                )
                .repeat();
            }
        }
    }
}

/// Returns the entity at this path.
fn get_entity(
    parent: &Entity,
    path: &[&str],
    child_query: &Query<(Entity, Option<&Name>, Option<&Children>)>,
) -> Option<Entity> {
    if path.is_empty() {
        return Some(*parent);
    }
    if let Ok((_, _, Some(children))) = child_query.get(*parent) {
        for child in children {
            let (_, name, _) = child_query.get(*child).unwrap();
            if (name.is_none() && path[0].is_empty()) || (name.unwrap().as_str() == path[0]) {
                let e = get_entity(child, &path[1..], child_query);
                if e.is_some() {
                    return e;
                }
            }
        }
    }
    None
}
